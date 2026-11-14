"""Tests CPU du worker vace (0 credit) : python -m pytest -q tests"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clips  # noqa: E402
import workflow  # noqa: E402

PORTRAIT = (480, 832)


def by_class(g, cls):
    return {k: v for k, v in g.items() if v["class_type"] == cls}


def source(g, link):
    """Noeud amont d'un lien [id, sortie]."""
    return g[link[0]]


# --- cablage du graphe --------------------------------------------------------------------------------

def test_control_mask_ref_wiring():
    files = {"controls": {"depth": "vace_x_depth-0.mkv"}, "mask": "vace_x_mask-0.mkv", "ref": "vace_x_refs.png"}
    g = workflow.build_workflow({"prompt": "a rainy street"}, files, PORTRAIT, 81, seed=7)
    (vid, vace), = by_class(g, "WanVaceToVideo").items()
    i = vace["inputs"]
    # controle -> control_video
    comp = source(g, i["control_video"])
    assert comp["class_type"] == "GetVideoComponents"
    assert source(g, comp["inputs"]["video"]) == {"class_type": "LoadVideo", "inputs": {"file": "vace_x_depth-0.mkv"},
                                                  "_meta": {"title": "control depth"}}
    # masque -> control_masks (canal rouge de la video masque)
    tomask = source(g, i["control_masks"])
    assert tomask["class_type"] == "ImageToMask" and tomask["inputs"]["channel"] == "red"
    assert source(g, source(g, tomask["inputs"]["image"])["inputs"]["video"])["inputs"]["file"] == "vace_x_mask-0.mkv"
    # references -> reference_image
    assert source(g, i["reference_image"]) == {"class_type": "LoadImage", "inputs": {"image": "vace_x_refs.png"}}
    assert (i["width"], i["height"], i["length"]) == (480, 832, 81)
    assert source(g, i["positive"])["inputs"]["text"] == "a rainy street"
    # echantillonnage et decoupe sur CE noeud VACE
    assert g["40"]["inputs"]["latent_image"] == [vid, 2] and g["40"]["inputs"]["positive"] == [vid, 0]
    assert g["42"]["inputs"]["trim_amount"] == [vid, 3]
    assert g[workflow.OUTPUT_NODE]["class_type"] == "SaveVideo"


def test_without_mask_or_ref_inputs_are_absent():
    g = workflow.build_workflow({}, {"controls": {"clay": "c.mkv"}}, PORTRAIT, 33)
    (vace,) = by_class(g, "WanVaceToVideo").values()
    assert "control_masks" not in vace["inputs"] and "reference_image" not in vace["inputs"]
    assert not by_class(g, "LoadImage") and not by_class(g, "ImageToMask")


def test_several_controls_are_chained_with_their_strength():
    files = {"controls": {"pose": "p.mkv", "depth": "d.mkv", "clay": "c.mkv"}, "ref": "r.png", "mask": "m.mkv"}
    g = workflow.build_workflow({"strength": {"clay": 0.5}}, files, PORTRAIT, 81)
    vaces = by_class(g, "WanVaceToVideo")
    order = sorted(vaces)
    roles = [source(g, source(g, vaces[k]["inputs"]["control_video"])["inputs"]["video"])["inputs"]["file"] for k in order]
    assert roles == ["d.mkv", "c.mkv", "p.mkv"]                              # ordre fixe : depth, clay, pose
    assert [vaces[k]["inputs"]["strength"] for k in order] == [1.0, 0.5, 1.0]
    assert vaces[order[0]]["inputs"]["positive"] == ["5", 0]
    for prev, cur in zip(order, order[1:]):                                  # contextes VACE empiles
        assert vaces[cur]["inputs"]["positive"] == [prev, 0] and vaces[cur]["inputs"]["negative"] == [prev, 1]
    for k in order:                                                          # meme masque et meme planche partout
        assert vaces[k]["inputs"]["control_masks"] == ["17", 0] and vaces[k]["inputs"]["reference_image"] == ["12", 0]
    assert g["40"]["inputs"]["latent_image"] == [order[-1], 2]


def test_two_experts_and_presets():
    files = {"controls": {"depth": "d.mkv"}}
    draft = workflow.build_workflow({}, files, (832, 480), 81)
    hi, lo = draft["40"]["inputs"], draft["41"]["inputs"]
    assert source(draft, source(draft, hi["model"])["inputs"]["model"])["inputs"]["lora_name"] == workflow.DISTILL_LORAS["high"]
    assert draft["1"]["inputs"]["unet_name"] == workflow.UNETS["high"] and draft["2"]["inputs"]["unet_name"] == workflow.UNETS["low"]
    assert (hi["start_at_step"], hi["end_at_step"], hi["add_noise"], hi["return_with_leftover_noise"]) == (0, 3, "enable", "enable")
    assert (lo["start_at_step"], lo["add_noise"], lo["latent_image"]) == (3, "disable", ["40", 0])
    assert hi["cfg"] == 1.0 and hi["steps"] == 6
    final = workflow.build_workflow({"preset": "final"}, files, (1280, 720), 81)
    assert not by_class(final, "LoraLoaderModelOnly")
    assert final["40"]["inputs"]["end_at_step"] == 10 and final["40"]["inputs"]["cfg"] == 3.5
    assert final["9"]["inputs"]["shift"] == 8.0


@pytest.mark.parametrize("params,files,err", [
    ({}, {"controls": {}}, "control pass"),
    ({}, {"controls": {"normals": "n.mkv"}}, "control pass"),
    ({"preset": "ultra"}, {"controls": {"depth": "d"}}, "preset"),
    ({"split": 6}, {"controls": {"depth": "d"}}, "split"),
    ({"strength": {"depth": 3}}, {"controls": {"depth": "d"}}, "strength"),
    ({"distill": 1}, {"controls": {"depth": "d"}}, "type"),
])
def test_build_refuses(params, files, err):
    with pytest.raises(ValueError, match=err):
        workflow.build_workflow(params, files, PORTRAIT, 81)


def test_graph_links_point_to_existing_nodes():
    files = {"controls": {"depth": "d", "clay": "c", "pose": "p"}, "mask": "m", "ref": "r"}
    for preset in workflow.PRESETS:
        g = workflow.build_workflow({"preset": preset}, files, PORTRAIT, 81)
        for node in g.values():
            for v in node["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    assert v[0] in g


# --- fenetres -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("total", [1, 33, 81, 82, 160, 240, 241])
def test_windows_cover_every_frame(total):
    plan = workflow.windows(total)
    covered = []
    for w in plan:
        assert (w["length"] - 1) % 4 == 0 and w["length"] >= w["count"] and w["length"] <= workflow.CAPACITY
        covered.extend(range(w["start"] + w["overlap"], w["start"] + w["count"]))
    assert covered == list(range(total))
    assert all(w["overlap"] == workflow.OVERLAP for w in plan[1:])
    # chaque fenetre suivante commence sur les OVERLAP dernieres images de la precedente
    for a, b in zip(plan, plan[1:]):
        assert b["start"] == a["start"] + a["count"] - workflow.OVERLAP


# --- entrees d'une fenetre (pur numpy) ------------------------------------------------------------------

def frames(n, value, hw=(8, 16)):
    return [np.full((*hw, 3), value, np.uint8) for _ in range(n)]


def test_window_inputs_plain():
    ready, mask = clips.window_inputs({"depth": frames(31, 50)}, 33)
    assert mask is None and len(ready["depth"]) == 33 and (ready["depth"][-1] == 50).all()


def test_window_inputs_mask_composites_base():
    m = frames(5, 0)
    for f in m:
        f[:, 8:] = 255                       # moitie droite a repeindre
    ready, mask = clips.window_inputs({"depth": frames(5, 200)}, 5, m, frames(5, 10))
    f = ready["depth"][0]
    assert (f[:, :8] == 10).all() and (f[:, 8:] == 200).all()      # hors masque : couche amont ; dedans : controle
    assert (mask[0][:, :8] == 0).all() and (mask[0][:, 8:] == 255).all()


def test_window_inputs_carry_keeps_previous_frames():
    carry = frames(3, 99)
    ready, mask = clips.window_inputs({"depth": frames(9, 200), "pose": frames(9, 40)}, 9, carry=carry)
    for role in ("depth", "pose"):
        assert all((f == 99).all() for f in ready[role][:3]) and (ready[role][3] != 99).all()
    assert all((k == 0).all() for k in mask[:3]) and all((k == 255).all() for k in mask[3:])


def test_window_inputs_refuses_mask_without_base_and_mismatched_passes():
    with pytest.raises(ValueError, match="base"):
        clips.window_inputs({"depth": frames(5, 1)}, 5, mask=frames(5, 255))
    with pytest.raises(ValueError, match="differs"):
        clips.window_inputs({"depth": frames(5, 1), "clay": frames(4, 1)}, 5)


def test_reference_board_size_and_content():
    refs = [np.full((100, 50, 3), 0, np.uint8), np.full((40, 40, 3), 128, np.uint8)]
    board = clips.reference_board(refs, (480, 832))
    assert board.shape == (832, 480, 3)
    assert (board == 0).any() and (board == 128).any() and (board == 255).any()


def test_assemble_drops_carried_frames():
    plan = workflow.windows(100)
    chunks = [[np.full((2, 2, 3), i * 100 + j, np.uint8) for j in range(w["length"])] for i, w in enumerate(plan)]
    out = clips.assemble(chunks, plan, 100)
    assert len(out) == 100 and out[81][0, 0, 0] == 100 + workflow.OVERLAP


# --- handler : refus avant tout telechargement ----------------------------------------------------------

@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setitem(sys.modules, "runpod", types.ModuleType("runpod"))
    import handler as h
    return h


@pytest.mark.parametrize("roles,params,err", [
    ({"ref0"}, {}, "control pass"),
    ({"depth", "mask"}, {}, "base"),
    ({"depth", "base"}, {}, "only used with"),
    ({"depth", "normals"}, {}, "unknown inputs"),
    ({"depth"}, {"max_frames": 500}, "max_frames"),
    ({"depth"}, {"seed": -1}, "seed"),
])
def test_check_request_refuses(handler, roles, params, err):
    with pytest.raises(ValueError, match=err):
        handler.check_request(params, roles)


def test_check_request_ok(handler):
    s = handler.check_request({"preset": "final", "steps": 24, "split": 12}, {"depth", "clay", "mask", "base", "ref0", "ref1"})
    assert (s["steps"], s["split"], s["distill"]) == (24, 12, False)


def test_unknown_action_is_an_error_event(handler):
    events = list(handler.handler({"id": "j", "input": {"params": {"action": "mask"}}}))
    assert events[0]["type"] == "version" and events[0]["worker"] == "vace"
    assert events[-1]["type"] == "error" and "unknown action" in events[-1]["message"]
