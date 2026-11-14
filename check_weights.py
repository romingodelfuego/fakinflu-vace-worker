#!/usr/bin/env python3
"""Garde-fou de BUILD : chaque poids nomme par les graphes des presets (workflow.weights) existe et n'est pas vide.

Usage : check_weights.py <dossier models>
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workflow  # noqa: E402

FOLDERS = {workflow.TEXT_ENCODER: "text_encoders", workflow.VAE: "vae",
           **{n: "diffusion_models" for n in workflow.UNETS.values()},
           **{n: "loras" for n in workflow.DISTILL_LORAS.values()}}


def main(models):
    bad = []
    for preset in workflow.PRESETS:
        for name in workflow.weights({"preset": preset}):
            path = Path(models) / FOLDERS[name] / name
            if not path.is_file() or not os.path.getsize(path):
                bad.append(f"{preset}: {path}")
    if bad:
        print("missing weights:", *sorted(set(bad)), sep="\n  ")
        return 1
    print("graph weights OK:", ", ".join(workflow.PRESETS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
