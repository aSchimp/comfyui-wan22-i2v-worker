FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
ARG IMAGE_VERSION=0.1.0
LABEL org.opencontainers.image.source="https://github.com/aschimp/comfyui-wan22-i2v-worker" \
      org.opencontainers.image.version="${IMAGE_VERSION}"
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_NO_INPUT=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI && git -C /opt/ComfyUI checkout --detach 72865f4f27eaf5396f8f36370e0a2be3a9a090ee
COPY requirements.txt /tmp/worker-requirements.txt
RUN python3 -m pip install --break-system-packages --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
 && grep -Ev '^(torch|torchvision|torchaudio)([<>=!~].*)?$' /opt/ComfyUI/requirements.txt > /tmp/comfy-requirements.txt \
 && python3 -m pip install --break-system-packages -r /tmp/comfy-requirements.txt -r /tmp/worker-requirements.txt
COPY extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
COPY handler.py workflow.py entrypoint.py /opt/worker/
RUN python3 -m py_compile /opt/worker/*.py && python3 /opt/ComfyUI/main.py --quick-test-for-ci --cpu \
 && ! find /opt/ComfyUI -type f \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' \) -print -quit | grep -q .
WORKDIR /opt/worker
CMD ["python3", "entrypoint.py"]
