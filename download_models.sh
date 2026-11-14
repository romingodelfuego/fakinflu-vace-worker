#!/usr/bin/env bash
# fakinflu — poids du worker vace (Couche decor : Wan 2.2 VACE Fun A14B), BAKES dans l'image.
# Tous Apache-2.0 (voir README, section Licences). sha256 = oid LFS lus via l'API HF le 2026-11-13.
#
#   MODELS_DIR=/comfyui/models ./download_models.sh core    # ~7 Go : umT5 fp8 scaled + VAE Wan 2.1 (memes fichiers qu'openjutsu)
#   MODELS_DIR=/comfyui/models ./download_models.sh vace    # ~34,7 Go : experts haut et bas bruit, fp8 scaled (17,3 Go chacun)
#   MODELS_DIR=/comfyui/models ./download_models.sh loras   # ~2,5 Go : lightx2v Wan2.2-Lightning T2V 4 etapes v1.1 (preset draft)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/comfyui/models}"

# DL <depot> <commit> <chemin dans le depot> <sous-dossier local> <sha256>
DL() {
  local repo="$1" rev="$2" src="$3" sub="$4" sha="$5" name
  name="$(basename "$src")"
  local dest="$MODELS_DIR/$sub/$name" url="https://huggingface.co/${repo}/resolve/${rev}/${src}"
  mkdir -p "$MODELS_DIR/$sub"
  echo ">> $sub/$name"
  local t0=$SECONDS
  if command -v aria2c >/dev/null; then
    aria2c -q -c -x 16 -s 16 -k 64M --max-tries=10 --retry-wait=5 --timeout=30 \
           --lowest-speed-limit=1M --file-allocation=none --console-log-level=warn \
           -d "$MODELS_DIR/$sub" -o "$name" "$url"
  else
    wget -q -c --tries=10 --timeout=30 --waitretry=5 -O "$dest" "$url"
  fi
  echo "   downloaded in $((SECONDS - t0)) s"
  echo "$sha  $dest" | sha256sum -c --quiet - || { echo "!! bad sha256: $dest"; rm -f "$dest"; exit 1; }
  test -s "$dest"
}

WA=e7181fe1896b7f2ce34120d1a3663413548240d7   # Comfy-Org/Wan-Animate-2 @ 2026-10-14 (encodeur, VAE : memes octets qu'openjutsu)
W22=ee6f4a40737a995bf5818954cfce6d59443b0f04  # Comfy-Org/Wan_2.2_ComfyUI_Repackaged @ 2026-09-23 (apache-2.0)

case "${1:-}" in
  core)
    DL Comfy-Org/Wan-Animate-2 "$WA" text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders \
       c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
    DL Comfy-Org/Wan-Animate-2 "$WA" vae/Wan2_1_VAE_bf16.safetensors vae \
       1ab9a32cc2c740f6e39d80d367ce5dcc28db8c71b79b28670546b8973e9d75f9 ;;
  vace)
    # alibaba-pai/Wan2.2-VACE-Fun-A14B (apache-2.0) reconditionne par Comfy-Org : modele complet (T2V A14B + blocs
    # VACE) par expert, charge par UNETLoader. bf16 = 34,7 Go chacun : trop pour 48 Go, d'ou le fp8 scaled.
    DL Comfy-Org/Wan_2.2_ComfyUI_Repackaged "$W22" split_files/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors diffusion_models \
       23130f30207f6c8697a5bead113ddc9904cfbb5e8fa51a550df3cee4d21fb54c
    DL Comfy-Org/Wan_2.2_ComfyUI_Repackaged "$W22" split_files/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors diffusion_models \
       ca55a5cc543e28576edbd2079f8c1de05f71cd9e85bf96fab635f55663dbcb69 ;;
  loras)
    # lightx2v/Wan2.2-Lightning (apache-2.0), T2V A14B 4 etapes v1.1, reconditionne par Comfy-Org.
    DL Comfy-Org/Wan_2.2_ComfyUI_Repackaged "$W22" split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors loras \
       698321cb86bd30c4af06c9b84e656a1048c8cb54e06d50694536fb5de37fde41
    DL Comfy-Org/Wan_2.2_ComfyUI_Repackaged "$W22" split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors loras \
       ec95216e614b3c132c11bfb387b11feedf62163150ccc9068bca8a189771e75a ;;
  *) echo "usage: $0 core|vace|loras" >&2; exit 2 ;;
esac
