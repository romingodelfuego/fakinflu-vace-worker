#!/usr/bin/env python3
"""fakinflu — worker vace : Couche decor de la Maquette (Wan 2.2 VACE Fun A14B, ComfyUI natif).

Remplace le /handler.py de runpod/worker-comfyui ; /start.sh lance ComfyUI (127.0.0.1:8188).
Handler GENERATEUR : progression et apercus via /stream (return_aggregate_stream = False) ; en batch RunPod
(defaut fakinflu) les sorties passent par R2 et outputs.json. Squelette (ComfyUI, websocket, OOM, sonde, VramMeter)
repris du worker openjutsu.

Entree (contrat transfer.py, contracts/vace.input.schema.json) :
  {"run_id", "r2": bool,
   "inputs":  {"depth" | "clay" | "pose": Passes de la Maquette (au moins une ; 16 i/s, graph.SIZES, memes tailles),
               "mask": zones a repeindre (blanc), "base": couche gardee hors du masque (requise avec mask),
               "ref0".."ref3": references globales (images)},
   "outputs": {"video", "preview", "probe"}, "outputs_json": {put_url, key} | null,
   "params":  {"action": "probe|generate", "prompt", "negative", "seed", "max_frames", "preset": "draft|final",
               "steps", "split", "cfg", "shift", "sampler", "scheduler", "distill", "strength": {role: 0..2}}}

Actions :
  probe    : VRAM, RAM du conteneur, disque, poids presents, version (aucun modele charge).
  generate : fenetres de 81 images (9 reportees), graphe workflow.build_workflow par fenetre -> result.mp4 (sans audio).
Sortie : evenement done avec "gpu" = {name, vram_total_gb, vram_start_gb, vram_peak_gb} et les durees ; le COUT est
calcule cote client (duree facturee x tarif du pool), jamais ici.
"""
import glob, json, os, queue, shutil, subprocess, tempfile, threading, time, traceback, urllib.error, urllib.request, uuid

import runpod

import clips
import workflow
from transfer import Outputs, fetch_inputs, preview_event, version_event

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = os.environ.get("COMFY_DIR", "/comfyui")
JOB_TIMEOUT = int(os.environ.get("VACE_JOB_TIMEOUT_S", "3000"))   # par fenetre
ACTIONS = ("probe", "generate")
REFS = tuple(f"ref{i}" for i in range(workflow.MAX_REFS))
ROLES = (*workflow.CONTROLS, "mask", "base", *REFS)


def _http(method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def wait_comfy(max_s=600):
    t = time.time()
    while time.time() - t < max_s:
        try:
            _http("GET", "/system_stats", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1)
    raise RuntimeError(f"ComfyUI unreachable at {COMFY} after {max_s}s")


def _ws_progress(client_id, q, stop):
    try:
        import websocket
        ws = websocket.create_connection(COMFY.replace("http", "ws", 1) + f"/ws?clientId={client_id}", timeout=5)
    except Exception:
        return
    ws.settimeout(1)
    while not stop.is_set():
        try:
            msg = ws.recv()
            m = json.loads(msg) if isinstance(msg, str) else None
        except Exception:
            continue
        if m and m.get("type") in ("progress", "executing"):
            q.put(m)
    ws.close()


def run_workflow(wf):
    """Generateur : soumet le graphe, emet la progression, renvoie (secondes) via StopIteration."""
    client_id = uuid.uuid4().hex
    q, stop = queue.Queue(), threading.Event()
    threading.Thread(target=_ws_progress, args=(client_id, q, stop), daemon=True).start()
    try:
        try:
            pid = _http("POST", "/prompt", {"prompt": wf, "client_id": client_id})["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the graph (HTTP {e.code}): {e.read().decode()[:1500]}")
        t0, last_poll = time.time(), 0.0
        names = {k: v.get("_meta", {}).get("title") or v["class_type"] for k, v in wf.items()}
        while True:
            if time.time() - t0 > JOB_TIMEOUT:
                _http("POST", "/interrupt", {})
                raise RuntimeError(f"ComfyUI timeout ({JOB_TIMEOUT}s)")
            try:
                m = q.get(timeout=1)
                d = m.get("data", {})
                if m["type"] == "executing" and d.get("node"):
                    yield {"step": names.get(d["node"], d["node"])}
                elif m["type"] == "progress" and d.get("max"):
                    yield {"step": names.get(d.get("node"), "compute"), "pct": round(100 * d["value"] / d["max"])}
            except queue.Empty:
                pass
            if time.time() - last_poll < 2:
                continue
            last_poll = time.time()
            try:
                h = _http("GET", f"/history/{pid}")
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError as e:
                raise RuntimeError(f"ComfyUI stopped mid-compute ({e.reason}): {oom_report()}") from e
            if pid not in h:
                continue
            st = h[pid].get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[-1][1] if msgs else st
                if isinstance(detail, dict):
                    detail = {k: detail.get(k) for k in ("node_id", "node_type", "exception_message")}
                raise RuntimeError(f"ComfyUI execution error: {json.dumps(detail, ensure_ascii=False)[:800]}")
            if st.get("completed"):
                return time.time() - t0
    finally:
        stop.set()


def _cgroup(name):
    for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        try:
            with open(os.path.join(base, name)) as f:
                return f.read()
        except OSError:
            continue
    return None


def memory_limit_bytes():
    for name in ("memory.max", "memory.limit_in_bytes"):
        v = (_cgroup(name) or "").strip()
        if v.isdigit() and int(v) < 1 << 60:
            return int(v)
    return None


def _gb(b):
    return round(int(b) / 1024 ** 3, 2) if b not in (None, "") and str(b).strip().isdigit() else None


def oom_report():
    events = _cgroup("memory.events") or ""
    kills = next((int(l.split()[1]) for l in events.splitlines() if l.startswith("oom_kill ")), None)
    peak = (_cgroup("memory.peak") or _cgroup("memory.max_usage_in_bytes") or "").strip()
    mem = f"peak {_gb(peak)} GB, limit {_gb(memory_limit_bytes())} GB"
    if kills:
        return f"killed by the kernel out of memory (oom_kill={kills}, {mem}). Check COMFY_EXTRA_ARGS ({os.environ.get('COMFY_EXTRA_ARGS', '')!r})"
    return f"no oom_kill in the cgroup ({mem}): likely native crash (worker RunPod logs)"


class VramMeter:
    """Pic de VRAM du GPU pendant le job (nvidia-smi chaque seconde ; memoire reservee : majorant du besoin)."""

    QUERY = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"]

    def __init__(self, every=1.0):
        self.every, self.stop = every, threading.Event()
        self.name = self.total = self.first = self.peak = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(self.QUERY, capture_output=True, text=True, timeout=5).stdout
                name, used, total = [x.strip() for x in out.splitlines()[0].split(",")]
                self.name, self.total = name, int(total)
                self.first = int(used) if self.first is None else self.first
                self.peak = max(self.peak or 0, int(used))
            except Exception:
                pass
            self.stop.wait(self.every)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=6)

    def report(self):
        if self.peak is None:
            return None
        gb = lambda mib: round(mib / 1024, 1)
        return {"name": self.name, "vram_total_gb": gb(self.total), "vram_start_gb": gb(self.first), "vram_peak_gb": gb(self.peak)}


def probe_report():
    try:
        stats = _http("GET", "/system_stats", timeout=10)
    except Exception as e:
        stats = {"error": f"{type(e).__name__}: {e}"}
    disk = {}
    for path in ("/", COMFY_DIR):
        try:
            st = os.statvfs(path)
            disk[path] = {"free_gb": round(st.f_bavail * st.f_frsize / 1024 ** 3, 2), "total_gb": round(st.f_blocks * st.f_frsize / 1024 ** 3, 2)}
        except OSError:
            pass
    system = stats.get("system") or {}
    return {
        "cgroup": {"limit_gb": _gb(memory_limit_bytes()), "peak_gb": _gb((_cgroup("memory.peak") or "").strip())},
        "comfyui": {"version": system.get("comfyui_version"), "pytorch": system.get("pytorch_version"), "argv": system.get("argv"),
                    "ram_total_gb": _gb(system.get("ram_total")), "error": stats.get("error"),
                    "devices": [{k: d.get(k) for k in ("name", "type", "vram_total", "vram_free")} for d in stats.get("devices") or []]},
        "models": {os.path.relpath(p, COMFY_DIR): _gb(os.path.getsize(p))
                   for p in sorted(glob.glob(os.path.join(COMFY_DIR, "models", "*", "*.safetensors")))},
        "disk": disk,
        "env": {"COMFY_EXTRA_ARGS": os.environ.get("COMFY_EXTRA_ARGS", "")},
    }


def check_request(params, roles):
    """Tout ce qui se verifie AVANT le telechargement des entrees (aucun GPU paye pour une requete invalide)."""
    unknown = sorted(set(roles) - set(ROLES))
    if unknown:
        raise ValueError(f"unknown inputs {unknown} (expected {', '.join(ROLES)})")
    if not any(r in roles for r in workflow.CONTROLS):
        raise ValueError(f"at least one control pass required in inputs: {', '.join(workflow.CONTROLS)}")
    if "mask" in roles and "base" not in roles:
        raise ValueError("inputs.mask needs inputs.base (layer kept outside the mask)")
    if "base" in roles and "mask" not in roles:
        raise ValueError("inputs.base is only used with inputs.mask")
    mf = params.get("max_frames")
    if mf is not None and (isinstance(mf, bool) or not isinstance(mf, int) or not 1 <= mf <= workflow.MAX_FRAMES):
        raise ValueError(f"params.max_frames must be an integer in [1, {workflow.MAX_FRAMES}]")
    seed = params.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("params.seed must be a non-negative integer")
    for k in ("prompt", "negative"):
        v = params.get(k)
        if v is not None and (not isinstance(v, str) or len(v) > 4000):
            raise ValueError(f"params.{k} must be a string (4000 characters at most)")
    return workflow.settings(params)


def check_videos(paths):
    """Passes, masque et couche amont : normalises par le client (16 i/s, workflow.SIZES), meme taille."""
    infos = {r: clips.probe(paths[r]) for r in (*workflow.CONTROLS, "mask", "base") if r in paths}
    size = None
    for role, info in infos.items():
        wh = (info["width"], info["height"])
        if wh not in workflow.SIZES or abs(info["fps"] - clips.FPS) > 0.05 or not info["frames"]:
            raise ValueError(f"{role} not normalized: {wh[0]}x{wh[1]} at {info['fps']:.2f} fps, {info['frames']} frames "
                             f"(expected {' or '.join(f'{w}x{h}' for w, h in workflow.SIZES)} at {clips.FPS} fps)")
        if size and wh != size:
            raise ValueError(f"{role} is {wh[0]}x{wh[1]}, other inputs are {size[0]}x{size[1]}")
        size = wh
    frames = min(i["frames"] for r, i in infos.items() if r in workflow.CONTROLS)
    short = [r for r in ("mask", "base") if r in infos and infos[r]["frames"] < frames]
    if short:
        raise ValueError(f"{', '.join(short)} shorter than the control passes ({frames} frames)")
    return size, frames


def thumb(path, dst, at=0.5):
    info = clips.probe(path)
    clips.ffmpeg("-ss", f"{max(0.0, at * info['frames'] / clips.FPS):.2f}", "-i", path, "-frames:v", 1, "-q:v", 5, dst)
    with open(dst, "rb") as f:
        return f.read()


class Comfy:
    """Fichiers deposes dans input/ de ComfyUI sous des noms uniques au job ; nettoyes a la fin."""

    def __init__(self, tag):
        self.tag, self.copied = tag, []

    def put(self, path, name):
        dst = os.path.join(COMFY_DIR, "input", f"vace_{self.tag}_{name}")
        shutil.copy(path, dst)
        self.copied.append(dst)
        return os.path.basename(dst)

    def cleanup(self):
        for p in self.copied:
            if os.path.exists(p):
                os.remove(p)


def do_generate(params, s, paths, outputs, work, prefix, comfy):
    size, available = check_videos(paths)
    total = min(available, params["max_frames"]) if params.get("max_frames") else min(available, workflow.MAX_FRAMES)
    plan = workflow.windows(total)
    roles = [r for r in workflow.CONTROLS if r in paths]
    refs = [r for r in REFS if r in paths]
    seed = params.get("seed") if params.get("seed") is not None else 0
    yield {"type": "progress", "pct": 2,
           "step": f"{s['preset']} {size[0]}x{size[1]}, {total} frames, {len(plan)} window(s), controls {roles}"
                   + (f", {len(refs)} reference(s)" if refs else "") + (", mask" if "mask" in paths else "")}
    files = {}
    if refs:
        from PIL import Image
        board = clips.reference_board([Image.open(paths[r]) for r in refs], size)
        p = os.path.join(work, "refs.png")
        Image.fromarray(board).save(p)
        files["ref"] = comfy.put(p, "refs.png")
        with open(p, "rb") as f:
            yield preview_event(f.read(), "image/png")
    wait_comfy()

    chunks, timings, carry = [], [], None
    for i, w in enumerate(plan):
        controls = {r: clips.read_range(paths[r], w["start"], w["count"]) for r in roles}
        mask = clips.read_range(paths["mask"], w["start"], w["count"]) if "mask" in paths else None
        base = clips.read_range(paths["base"], w["start"], w["count"]) if "base" in paths else None
        ready, masks = clips.window_inputs(controls, w["length"], mask, base, carry)
        files["controls"] = {}
        for r, frames in ready.items():
            p = os.path.join(work, f"{r}-{i}.mkv")
            clips.write_frames(frames, p, lossless=True)
            files["controls"][r] = comfy.put(p, f"{r}-{i}.mkv")
        files.pop("mask", None)
        if masks is not None:
            p = os.path.join(work, f"mask-{i}.mkv")
            clips.write_frames(masks, p, lossless=True)
            files["mask"] = comfy.put(p, f"mask-{i}.mkv")
        wf = workflow.build_workflow(params, files, size, w["length"], seed, f"{prefix}/chunk-{i}")
        label = f"window {i + 1}/{len(plan)}"
        gen = run_workflow(wf)
        while True:
            try:
                ev = next(gen)
            except StopIteration as stop:
                seconds = stop.value
                break
            yield {"type": "progress", "step": f"{label}: {ev['step']}",
                   "pct": round(5 + 90 * (i + ev.get("pct", 0) / 100) / len(plan))}
        hits = sorted(glob.glob(os.path.join(COMFY_DIR, "output", prefix, f"chunk-{i}_*.mp4")))
        if not hits:
            raise RuntimeError(f"{label}: ComfyUI completed without a video output (node {workflow.OUTPUT_NODE})")
        chunk = os.path.join(work, f"chunk-{i}.mp4")
        shutil.move(hits[-1], chunk)
        frames = clips.read_range(chunk, 0, w["length"])
        chunks.append(frames)
        carry = frames[w["count"] - workflow.OVERLAP:w["count"]] if i + 1 < len(plan) else None
        timings.append(round(seconds, 1))
        print(f"[vace] {label}: {w['length']} frames in {seconds:.1f}s", flush=True)
        yield preview_event(thumb(chunk, chunk + ".jpg"), "image/jpeg", segment=i, total=len(plan))

    result = os.path.join(work, "result.mp4")
    clips.write_frames(clips.assemble(chunks, plan, total), result)
    preview = os.path.join(work, "preview.jpg")
    thumb(result, preview)
    yield from outputs.emit("video", result, "result.mp4")
    yield from outputs.emit("preview", preview, "preview.jpg")
    outputs.write_summary()
    comfy_s = sum(timings)
    summary = {"preset": s["preset"], "controls": roles, "references": len(refs), "mask": "mask" in paths,
               "frames": total, "windows": len(plan),
               **{k: s[k] for k in ("distill", "steps", "split", "cfg", "shift", "sampler", "scheduler", "strength")}}
    print(f"[vace] done: {json.dumps(summary)}", flush=True)
    yield {"type": "progress", "pct": 100, "step": f"done: {json.dumps(summary)}"}
    # evenement done strict (contracts/stream-event.schema.json) : seed, durees, gpu
    yield {"type": "done", "seed": seed,
           "timings": {"comfy_s": round(comfy_s, 1), "windows_s": timings, "s_per_frame": round(comfy_s / total, 2)}}


def handler(job):
    inp = job.get("input") or {}
    params = inp.get("params") or {}
    action = params.get("action") or "generate"
    work = tempfile.mkdtemp(prefix="vace_")
    prefix = f"fakinflu/{inp.get('run_id') or job.get('id', 'run')}"
    comfy = Comfy(uuid.uuid4().hex[:8])
    t0 = time.time()
    meter = VramMeter().start()
    yield version_event("vace")
    try:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r} ({'|'.join(ACTIONS)})")
        outputs = Outputs(inp)
        if action == "probe":
            wait_comfy(120)
            report = probe_report()
            print(f"[vace] probe: {json.dumps(report)}", flush=True)
            path = os.path.join(work, "probe.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            yield from outputs.emit("probe", path, "probe.json")
            yield {"type": "done", "timings": {"total_s": round(time.time() - t0, 1)}}
            return
        s = check_request(params, set(inp.get("inputs") or {}))
        paths, events = fetch_inputs(inp, os.path.join(work, "in"), outputs)
        yield from events
        for ev in do_generate(params, s, paths, outputs, work, prefix, comfy):
            if ev.get("type") == "done":
                meter.close()
                ev = {**ev, "gpu": meter.report()}
                print(f"[vace] gpu: {json.dumps(ev['gpu'])}", flush=True)
            yield ev
    except Exception as e:
        traceback.print_exc()
        meter.close()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}", "gpu": meter.report()}
    finally:
        meter.close()
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(os.path.join(COMFY_DIR, "output", prefix), ignore_errors=True)
        comfy.cleanup()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
