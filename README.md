# fakinflu — worker `vace` (Couche décor de la Maquette)

Génère l'image **entière** d'un plan de 5 à 15 s à partir des Passes de la Maquette (`docs/adr/0001-maquette.md`,
étape 6 de la v1) : Wan 2.2 VACE Fun A14B sur ComfyUI, nœuds **natifs** (`WanVaceToVideo`), poids dans l'image.
La couche personnage (openjutsu, mode « entrées fournies ») passe ensuite sur la zone du corps.

État : **dossier simple, rien de déployé**. Dépôt `romingodelfuego/fakinflu-vace-worker` et endpoint
`fakinflu-vace-worker` à créer avec l'accord de l'utilisateur (skill deploy-worker).

## Modèle

| | Retenu : Wan 2.2 VACE Fun A14B | Écarté : Wan 2.1 VACE 14B |
| --- | --- | --- |
| Poids | `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` @ `ee6f4a40` : experts haut et bas bruit, fp8 scaled, 17,3 Go chacun | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` : fp16 seul, 34,7 Go |
| Licence | Apache-2.0 (alibaba-pai/Wan2.2-VACE-Fun-A14B, Comfy-Org) | Apache-2.0 |
| 48 Go | un expert fp8 à la fois en VRAM (ComfyUI décharge l'autre en RAM) | fp16 + umT5 ne tient pas sans conversion fp8 au chargement |
| Template ComfyUI | aucun template officiel VACE Fun 2.2 : graphe = template `video_wan_vace_14B_v2v` (2.1) + échantillonnage à deux experts du template Wan 2.2 T2V 14B | `video_wan_vace_14B_v2v`, `video_wan_vace_inpainting` |

Encodeur umT5 fp8 scaled et VAE Wan 2.1 : mêmes fichiers (mêmes sha256) qu'openjutsu. Preset `draft` : LoRA
lightx2v Wan2.2-Lightning T2V 4 étapes v1.1 (Apache-2.0). Aucune licence non commerciale dans l'image.

## Câblage (`workflow.build_workflow`, pur, testé sans GPU)

- Chaque Passe fournie (`depth`, `clay` = pâte éclairée, `pose`) → un `WanVaceToVideo` (`control_video`), avec sa
  force (`params.strength`). Les nœuds sont **chaînés** : ComfyUI empile un contexte VACE par nœud et le modèle les
  somme (`vace_frames` / `vace_strength` en listes, `comfy/ldm/wan/model.py`).
- `mask` (blanc = zone à repeindre) → `control_masks` ; hors masque, `control_video` porte la couche amont `base`
  (pixels gardés). Masque sans `base` : refusé.
- `ref0..ref3` → une **planche** (`clips.reference_board`) → `reference_image` de chaque nœud : le nœud natif ne lit
  que la première image du lot (`reference_image[:1]`), et tous les contextes doivent avoir la même longueur.
- Deux `KSamplerAdvanced` : expert haut bruit (0 → `split`), puis bas bruit (`split` → fin).
- Fenêtres de 81 images (4n+1, entraînement 81 images à 16 i/s), **9 reportées** : les 9 dernières images générées
  d'une fenêtre ouvrent la suivante dans toutes les Passes, masque 0 (gardées), puis l'assemblage les retire.

## Contrat (`contracts/vace.input.schema.json`)

```json
{"input": {"run_id": "…", "r2": true,
  "inputs":  {"depth": {…}, "clay": {…}, "pose": {…}, "mask": {…}, "base": {…}, "ref0": {…}},
  "outputs": {"video": {…}, "preview": {…}, "probe": {…}},
  "params": {"action": "probe|generate", "prompt": "", "negative": null, "seed": 42, "max_frames": null,
             "preset": "draft|final", "steps": null, "split": null, "cfg": null, "shift": null,
             "sampler": null, "scheduler": null, "distill": null, "strength": {"depth": 1.0, "clay": 0.8}}}}
```

Vidéos normalisées par le client : 16 i/s, 480×832 / 832×480 (draft) ou 720×1280 / 1280×720 (final), même taille
pour toutes ; 241 images au plus. Sortie `result.mp4` sans audio + `preview.jpg`. Événement `done` : `seed`,
`timings`, `gpu` (le coût se calcule côté client : durée facturée × tarif du pool, jamais ici).

## Tests (0 crédit)

```bash
python -m pytest -q tests     # câblage du graphe, presets, fenêtres, report, planche, refus du handler
```
