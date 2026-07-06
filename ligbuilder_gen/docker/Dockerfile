# LigBuilder V3 Docker Image — De Novo Ligand Design
# Build:  docker build -t zhuangzhou-ligbuilder .
# Usage:  docker run --rm -v /path/to/LigBuilderV3:/ligbuilder:ro \
#           -v /path/to/input:/input:ro -v /path/to/output:/output \
#           zhuangzhou-ligbuilder full /input/protein.pdb /output \
#           --center 37.3 31.2 44.1 --size 15 15 15
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LIGBUILDER_ROOT=/ligbuilder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        openbabel libopenbabel-dev \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# LigBuilder binaries are mounted at /ligbuilder via volume
# or can be copied during build for self-contained image:
# COPY LigBuilderV3/bin/ /opt/ligbuilder/bin/
# COPY LigBuilderV3/parameter/ /opt/ligbuilder/parameter/
# COPY LigBuilderV3/default/ /opt/ligbuilder/default/

COPY run_ligbuilder.py /opt/run_ligbuilder.py

WORKDIR /output

ENTRYPOINT ["python3", "/opt/run_ligbuilder.py"]
