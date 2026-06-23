# CUDA 12.8 runtime with cuDNN 9 — needed for NVIDIA Blackwell GPUs (RTX 50xx,
# sm_120) and providing the libcudnn that ctranslate2 (faster-whisper) uses.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    DATA_DIR=/data

# Python 3.10, ffmpeg (for audio extraction + whisperx load_audio), git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install PyTorch from the CUDA 12.8 (cu128) wheels first so Blackwell GPUs
# (sm_120) are supported; the bare torch in requirements.txt is then satisfied.
RUN python -m pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt \
    # whisperx 3.3.4 pins ctranslate2<4.5.0, but those builds need cuDNN 8 (the
    # image ships cuDNN 9) and lack Blackwell sm_120 kernels — transcription would
    # abort with "libcudnn_ops_infer.so.8 not found". Force the cuDNN-9 / sm_120
    # build over the pin (--no-deps so whisperx itself is left intact).
    && pip install --no-deps --upgrade "ctranslate2==4.8.0"

COPY src ./src
COPY web ./web

EXPOSE 8000

# Run the FastAPI web interface (container port 8000; published as 8800 in compose).
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
