"""Images des fenetres de la Couche decor : lecture, composition des entrees VACE, planche de references, assemblage.

E/S video reprises d'openjutsu/serverless/openjutsu/video.py (SydFloyd/openjutsu @ f783a001, Apache-2.0).
window_inputs et reference_board sont pures (numpy) : testables sur CPU sans video.
"""
from fractions import Fraction
import subprocess

import numpy as np

FPS = 16


def ffmpeg(*args):
    import imageio_ffmpeg
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *map(str, args)],
                       capture_output=True, text=True)
    if r.returncode:
        raise ValueError(r.stderr.strip())


def probe(path):
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        count = st.frames or sum(1 for _ in c.decode(video=0))
        return {"width": st.width, "height": st.height, "frames": count, "fps": float(st.average_rate or FPS)}


def read_range(path, start, count):
    """Images [start, start+count) en RGB uint8 ; la video est decodee depuis le debut (clips <= 15 s)."""
    import av
    out = []
    with av.open(str(path)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i >= start + count:
                break
            if i >= start:
                out.append(frame.to_ndarray(format="rgb24"))
    return out


def write_frames(frames, path, fps=FPS, lossless=False):
    import av
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as c:
        st = c.add_stream("ffv1" if lossless else "libx264", rate=Fraction(str(fps)))
        st.width, st.height = w, h
        st.pix_fmt = "bgr0" if lossless else "yuv420p"
        if not lossless:
            st.options = {"crf": "18"}
        for image in frames:
            for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def _pad(frames, length):
    """Complete jusqu'a `length` en clonant la derniere image (fenetre finale, 4n+1)."""
    return list(frames) + [frames[-1]] * (length - len(frames))


def window_inputs(controls, length, mask=None, base=None, carry=None):
    """Entrees VACE d'une fenetre (images RGB uint8 HxWx3).

    controls : {role: [images]} des Passes (count images) ; mask : [images] blanc = zone a repeindre ;
    base : [images] gardees hors du masque (couche amont) ; carry : dernieres images GENEREES de la fenetre
    precedente, reportees en tete et gardees telles quelles (masque 0) : continuite d'une fenetre a l'autre.

    Renvoie ({role: [images]}, [images masque] ou None). Sans masque ni report : masque None (tout est genere).
    """
    if mask is not None and base is None:
        raise ValueError("a mask needs a base layer (pixels kept outside the mask)")
    first = next(iter(controls.values()))
    shape = first[0].shape
    for role, frames in controls.items():
        if len(frames) != len(first) or frames[0].shape != shape:
            raise ValueError(f"control {role}: frame count or size differs from the other passes")
    count = len(first)
    m = None
    if mask is not None:
        if len(mask) < count or len(base) < count:
            raise ValueError("mask and base must cover the window")
        m = [(f[:, :, :1].astype(np.float32) / 255.0) for f in mask[:count]]
    out = {}
    for role, frames in controls.items():
        frames = list(frames)
        if m is not None:
            frames = [np.rint(b * (1 - k) + c * k).astype(np.uint8) for b, c, k in zip(base, frames, m)]
        if carry:
            frames[:len(carry)] = list(carry)
        out[role] = _pad(frames, length)
    if m is None and not carry:
        return out, None
    masks = [np.repeat((k * 255).astype(np.uint8), 3, axis=2) for k in m] if m is not None \
        else [np.full(shape, 255, np.uint8) for _ in range(count)]
    for i in range(len(carry or ())):
        masks[i] = np.zeros(shape, np.uint8)
    return out, _pad(masks, length)


def reference_board(images, size, background=255):
    """Une seule image de reference (le noeud natif ne lit que la 1re) : les references cote a cote, chacune
    reduite pour tenir dans sa case, centrees sur fond blanc, a la taille du plan (largeur, hauteur)."""
    from PIL import Image
    if not images:
        raise ValueError("no reference image")
    w, h = size
    board = Image.new("RGB", (w, h), (background,) * 3)
    n = len(images)
    horizontal = w >= h
    cell = (w // n, h) if horizontal else (w, h // n)
    for i, img in enumerate(images):
        im = Image.fromarray(img[:, :, :3]) if isinstance(img, np.ndarray) else img.convert("RGB")
        im.thumbnail(cell, Image.LANCZOS)
        ox = (i * cell[0] if horizontal else 0) + (cell[0] - im.width) // 2
        oy = (0 if horizontal else i * cell[1]) + (cell[1] - im.height) // 2
        board.paste(im, (ox, oy))
    return np.asarray(board)


def assemble(chunks, plan, total):
    """Images finales : fenetre 0 entiere, puis chaque fenetre sans ses images reportees (deja dans la precedente)."""
    frames = []
    for current, w in zip(chunks, plan):
        current = current[:w["count"]]
        if len(current) != w["count"]:
            raise ValueError(f"chunk has {len(current)} frames, expected {w['count']}")
        frames.extend(current[w["overlap"]:])
    if len(frames) != total:
        raise ValueError(f"assembled {len(frames)} frames, expected {total}")
    return frames
