# PLIP Docker Image — Protein-Ligand Interaction Profiler
# Build:  docker build -t zhuangzhou-plip .
# Usage:  docker run --rm -v /path/to/input:/input:ro -v /path/to/output:/output zhuangzhou-plip /input/complex.pdb /output
FROM debian:trixie-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-setuptools \
        python3-numpy python3-lxml \
        python3-openbabel openbabel && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
    --no-build-isolation --no-deps plip

COPY run_plip.py /opt/run_plip.py

ENTRYPOINT ["python3", "/opt/run_plip.py"]
