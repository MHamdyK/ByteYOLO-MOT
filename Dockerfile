FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt && \
    pip install -e .

# Apply the NumPy‐2.0 patch at build time
RUN python -c "import src.utils.numpy_patch"

ENTRYPOINT ["byteyolo"]
CMD ["--help"]
