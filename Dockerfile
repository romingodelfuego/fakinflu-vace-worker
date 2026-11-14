# fakinflu — worker RunPod : vace, Couche decor de la Maquette (Wan 2.2 VACE Fun A14B fp8 scaled, ComfyUI natif).
# Build par RunPod a chaque push sur main du depot du worker (a creer, accord de l'utilisateur).
#
# Base : celle d'openjutsu / wananimate (5.10.0-base = ComfyUI v0.34.0, COMFY_GRAPH_REF du Makefile : a bouger ensemble).
# Les garde-fous ci-dessous echouent le BUILD si un noeud des graphes manque : on le sait sans payer un job GPU.
ARG WORKER_TAG=5.10.0-base
FROM runpod/worker-comfyui:${WORKER_TAG}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN grep -q 'control_masks' /comfyui/comfy_extras/nodes_wan.py \
 && for n in WanVaceToVideo TrimVideoLatent KSamplerAdvanced ImageToMask LoadVideo GetVideoComponents CreateVideo \
             SaveVideo ModelSamplingSD3 LoraLoaderModelOnly UNETLoader CLIPLoader VAELoader; do \
      grep -rqw "$n" /comfyui/nodes.py /comfyui/comfy_extras || { echo "!! missing ComfyUI node: $n"; exit 1; }; \
    done

# ComfyUI dimensionne sa memoire epinglee sur la RAM de l'HOTE, pas du conteneur (cf. scail, openjutsu) :
# ${COMFY_EXTRA_ARGS} ajoute au start.sh (defaut ci-dessous, modifiable dans l'env de l'endpoint sans rebuild).
RUN sed -i 's#--log-stdout &#--log-stdout ${COMFY_EXTRA_ARGS} \&#' /start.sh \
 && test "$(grep -c 'COMFY_EXTRA_ARGS} &' /start.sh)" -ge 1
ENV COMFY_EXTRA_ARGS="--disable-pinned-memory"

RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
 && rm -rf /var/lib/apt/lists/*

# Versions de l'image de base FIGEES : un torch remplace casserait CUDA, un huggingface-hub 1.x casserait ComfyUI.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|huggingface-hub|huggingface_hub)==' > /tmp/pins.txt \
 && pip install --no-cache-dir -c /tmp/pins.txt av imageio-ffmpeg pillow websocket-client \
 && python3 -c "import av, imageio_ffmpeg, PIL, websocket" \
 && pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy|transformers|huggingface-hub|huggingface_hub)==' | diff - /tmp/pins.txt

# --- Poids BAKES (~44 Go) ---------------------------------------------------------------------------
COPY download_models.sh /tmp/download_models.sh
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh core
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh vace
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh loras && rm -f /tmp/download_models.sh

# --- Handler fakinflu (remplace /handler.py de worker-comfyui) ---------------------------------------
COPY transfer.py clips.py workflow.py VERSION* /fakinflu/
COPY handler.py /handler.py
COPY test_input.json /test_input.json
# Chaque graphe des presets nomme des poids presents dans l'image (echoue le build sinon).
COPY check_weights.py /fakinflu/check_weights.py
RUN python3 /fakinflu/check_weights.py /comfyui/models
ENV PYTHONPATH=/fakinflu \
    COMFY_LOG_LEVEL=INFO

CMD ["/start.sh"]
