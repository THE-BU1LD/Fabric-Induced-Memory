# syntax=docker/dockerfile:1.7
# Digest is the Docker Hub image identity for this exact PyTorch/CUDA tag.
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace/itfm_v2
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir numpy==2.4.4 scipy==1.17.1 \
    && python -m pip install --no-deps . \
    && python -c "import importlib.metadata as m; assert m.version('torch') == '2.11.0'"

ENTRYPOINT ["fim-sota"]
CMD ["--help"]
