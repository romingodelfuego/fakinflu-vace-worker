"""Graphe ComfyUI (format API) de la Couche decor : Wan 2.2 VACE Fun A14B, noeuds ComfyUI natifs.

Fonctions PURES (aucun fichier lu ni ecrit, ni torch ni av) : testables sur CPU, et appelees telles quelles par
tools/comfy_workflow.py (bouton i de l'interface) et tools/comfy_graph_check.py (CI, sans poids).

Cablage (WanVaceToVideo, comfy_extras/nodes_wan.py de ComfyUI v0.34.0) :
  - chaque Passe de controle (depth, clay, pose) -> son propre WanVaceToVideo, `control_video` ; les noeuds sont
    CHAINES (positive/negative de l'un -> l'autre) : ComfyUI ajoute un contexte VACE par noeud (append=True,
    vace_frames/vace_mask/vace_strength en listes) et le modele les somme, chacun avec sa force ;
  - le masque (zones a repeindre, blanc = 1) -> `control_masks` de chaque noeud ; 0 = pixels de control_video gardes
    (decor amont hors masque, images reportees de la fenetre precedente) ;
  - la planche de references -> `reference_image` de chaque noeud (le noeud natif ne lit que la 1re image du lot :
    plusieurs references sont assemblees en une planche par le worker, clips.reference_board). Tous les noeuds la
    recoivent : les contextes VACE doivent avoir la meme longueur pour etre empiles.
Wan 2.2 A14B = deux experts : haut bruit (etapes 0..split) puis bas bruit (split..fin), KSamplerAdvanced x2.
"""

OUTPUT_NODE = "45"
FPS = 16
SIZES = ((480, 832), (832, 480), (720, 1280), (1280, 720))
CAPACITY = 81        # fenetre Wan (4n+1), entrainement 81 images a 16 i/s
OVERLAP = 9          # images reportees d'une fenetre a l'autre (1 + 2x4 : 3 latents entiers)
MAX_FRAMES = 241     # plan de 15 s a 16 i/s (+1 pour le 4n+1)
CONTROLS = ("depth", "clay", "pose")   # Passes de la Maquette lues comme controle (ordre fixe des noeuds)
MAX_REFS = 4

UNETS = {"high": "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors",
         "low": "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors"}
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
DISTILL_LORAS = {"high": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                 "low": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"}
NEGATIVE = ("blurry, distorted, deformed, extra limbs, watermark, text, subtitles, low quality, jpeg artifacts, "
            "static frame, overexposed")
SAMPLERS = ("euler", "euler_ancestral", "uni_pc", "uni_pc_bh2", "dpmpp_2m", "dpmpp_sde", "lcm", "res_multistep")
SCHEDULERS = ("simple", "normal", "beta", "sgm_uniform", "karras", "exponential")

# draft : LoRA lightx2v 4 etapes (Wan2.2-Lightning T2V v1.1), CFG 1 ; final : template Wan 2.2 T2V 14B (20 etapes,
# bascule a 10, CFG 3,5, shift 8). Tout se remplace par job (params).
PRESETS = {
    "draft": {"distill": True, "steps": 6, "split": 3, "cfg": 1.0, "shift": 5.0, "sampler": "euler", "scheduler": "simple",
              "strength": {"depth": 1.0, "clay": 0.8, "pose": 1.0}},
    "final": {"distill": False, "steps": 20, "split": 10, "cfg": 3.5, "shift": 8.0, "sampler": "euler", "scheduler": "simple",
              "strength": {"depth": 1.0, "clay": 0.8, "pose": 1.0}},
}
TUNABLE = {"distill": bool, "steps": int, "split": int, "cfg": (int, float), "shift": (int, float), "sampler": str, "scheduler": str}


def settings(params=None):
    """Reglages resolus du job : preset (draft par defaut), puis params non nuls. ValueError si hors bornes."""
    params = params or {}
    name = params.get("preset") or "draft"
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r} (available: {', '.join(PRESETS)})")
    s = {**PRESETS[name], "strength": dict(PRESETS[name]["strength"]), "preset": name}
    for k, t in TUNABLE.items():
        v = params.get(k)
        if v is None:
            continue
        if isinstance(v, bool) != (t is bool) or not isinstance(v, t):
            raise ValueError(f"params.{k} has the wrong type")
        s[k] = v
    for k, v in (params.get("strength") or {}).items():
        if k not in CONTROLS or isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 2:
            raise ValueError(f"params.strength.{k} must be a number in [0, 2] for one of {', '.join(CONTROLS)}")
        s["strength"][k] = float(v)
    if not 1 <= s["steps"] <= 60:
        raise ValueError("params.steps must be in [1, 60]")
    if not 1 <= s["split"] < s["steps"]:
        raise ValueError("params.split must be in [1, steps - 1] (step where the low-noise expert takes over)")
    if not 0 <= s["cfg"] <= 20 or not 0 <= s["shift"] <= 50:
        raise ValueError("params.cfg must be in [0, 20] and params.shift in [0, 50]")
    if s["sampler"] not in SAMPLERS or s["scheduler"] not in SCHEDULERS:
        raise ValueError(f"params.sampler in {SAMPLERS}, params.scheduler in {SCHEDULERS}")
    return s


def windows(total, capacity=CAPACITY, overlap=OVERLAP):
    """Fenetres couvrant `total` images ; la derniere est completee jusqu'a 4n+1 (images clonees).
    Les `overlap` premieres images d'une fenetre > 0 sont les dernieres images GENEREES de la precedente."""
    if total < 1 or capacity < 9 or (capacity - 1) % 4 or not 0 < overlap < capacity:
        raise ValueError("Need frames>0, a 4n+1 window>=9, and 0<overlap<window")
    out, start = [], 0
    while True:
        count = min(capacity, total - start)
        out.append({"start": start, "count": count, "length": max(1, ((count - 1 + 3) // 4) * 4 + 1),
                    "overlap": 0 if start == 0 else overlap})
        if start + count >= total:
            return out
        start += capacity - overlap


def build_workflow(params, files, size, length, seed=0, prefix="fakinflu/run"):
    """Graphe d'une fenetre.

    params : params du job (prompt, negative, preset, steps, split, cfg, shift, sampler, scheduler, distill, strength).
    files  : noms des fichiers deja copies dans input/ de ComfyUI :
             {"controls": {"depth"|"clay"|"pose": video}, "mask": video (option), "ref": image (option)}.
    size   : (largeur, hauteur) d'un graph.SIZES ; length : images de la fenetre (4n+1).
    """
    s = settings(params)
    controls = [r for r in CONTROLS if r in (files.get("controls") or {})]
    if not controls:
        raise ValueError(f"at least one control pass required ({', '.join(CONTROLS)})")
    unknown = set(files.get("controls") or {}) - set(CONTROLS)
    if unknown:
        raise ValueError(f"unknown control pass(es): {', '.join(sorted(unknown))}")
    if tuple(size) not in SIZES:
        raise ValueError(f"size {size} not in {SIZES}")
    if length < 1 or (length - 1) % 4:
        raise ValueError("length must be 4n+1")
    w, h = size
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNETS["high"], "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": UNETS["low"], "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": (params.get("prompt") or "").strip()}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0],
                                                        "text": NEGATIVE if params.get("negative") is None else params["negative"]}},
    }
    models = {"high": ["1", 0], "low": ["2", 0]}
    if s["distill"]:
        for key, node in (("high", "7"), ("low", "8")):
            g[node] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": models[key], "lora_name": DISTILL_LORAS[key], "strength_model": 1.0}}
            models[key] = [node, 0]
    for key, node in (("high", "9"), ("low", "10")):
        g[node] = {"class_type": "ModelSamplingSD3", "inputs": {"model": models[key], "shift": s["shift"]}}
        models[key] = [node, 0]
    if files.get("ref"):
        g["12"] = {"class_type": "LoadImage", "inputs": {"image": files["ref"]}}
    if files.get("mask"):
        g["15"] = {"class_type": "LoadVideo", "inputs": {"file": files["mask"]}}
        g["16"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["15", 0]}}
        g["17"] = {"class_type": "ImageToMask", "inputs": {"image": ["16", 0], "channel": "red"}}
    positive, negative, vace = ["5", 0], ["6", 0], None
    for i, role in enumerate(controls):
        load, comp, vace = f"2{i}", f"2{i}1", f"3{i}"
        g[load] = {"class_type": "LoadVideo", "inputs": {"file": files["controls"][role]}, "_meta": {"title": f"control {role}"}}
        g[comp] = {"class_type": "GetVideoComponents", "inputs": {"video": [load, 0]}}
        inputs = {"positive": positive, "negative": negative, "vae": ["4", 0], "width": w, "height": h, "length": length,
                  "batch_size": 1, "strength": s["strength"][role], "control_video": [comp, 0]}
        if files.get("mask"):
            inputs["control_masks"] = ["17", 0]
        if files.get("ref"):
            inputs["reference_image"] = ["12", 0]
        g[vace] = {"class_type": "WanVaceToVideo", "inputs": inputs, "_meta": {"title": f"VACE {role}"}}
        positive, negative = [vace, 0], [vace, 1]
    common = {"noise_seed": seed, "steps": s["steps"], "cfg": s["cfg"], "sampler_name": s["sampler"],
              "scheduler": s["scheduler"], "positive": positive, "negative": negative}
    g["40"] = {"class_type": "KSamplerAdvanced", "_meta": {"title": "high noise"},
               "inputs": {**common, "model": models["high"], "add_noise": "enable", "latent_image": [vace, 2],
                          "start_at_step": 0, "end_at_step": s["split"], "return_with_leftover_noise": "enable"}}
    g["41"] = {"class_type": "KSamplerAdvanced", "_meta": {"title": "low noise"},
               "inputs": {**common, "model": models["low"], "add_noise": "disable", "latent_image": ["40", 0],
                          "start_at_step": s["split"], "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
    g["42"] = {"class_type": "TrimVideoLatent", "inputs": {"samples": ["41", 0], "trim_amount": [vace, 3]}}
    g["43"] = {"class_type": "VAEDecode", "inputs": {"samples": ["42", 0], "vae": ["4", 0]}}
    g["44"] = {"class_type": "CreateVideo", "inputs": {"images": ["43", 0], "fps": float(FPS)}}
    g[OUTPUT_NODE] = {"class_type": "SaveVideo", "inputs": {"video": ["44", 0], "filename_prefix": prefix, "format": "mp4"}}
    return g


def weights(params=None):
    """Poids nommes par le graphe de ces params (garde-fou de build : check_weights.py)."""
    s = settings(params)
    names = [UNETS["high"], UNETS["low"], TEXT_ENCODER, VAE]
    return names + (list(DISTILL_LORAS.values()) if s["distill"] else [])
