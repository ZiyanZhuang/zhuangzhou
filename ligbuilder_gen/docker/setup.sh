#!/bin/bash
# Setup script for LigBuilder V3 Docker integration
# Run from the docker/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_IMAGE="zhuangzhou-ligbuilder"

echo "=== LigBuilder V3 Docker Setup ==="
echo "Script directory: $SCRIPT_DIR"

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not found. Install Docker Desktop first."
    exit 1
fi

# Check LigBuilder source
LB_ROOT="${LIGBUILDER_ROOT:-}"
if [ -z "$LB_ROOT" ]; then
    echo "ERROR: LIGBUILDER_ROOT is not set."
    exit 1
fi
if [ ! -f "$LB_ROOT/bin/cavity" ]; then
    echo "ERROR: LigBuilder not found at $LB_ROOT"
    echo "Set LIGBUILDER_ROOT environment variable"
    exit 1
fi

echo "LigBuilder root: $LB_ROOT"

# Build Docker image
echo ""
echo "Building Docker image: $DOCKER_IMAGE ..."
docker build -t "$DOCKER_IMAGE" "$SCRIPT_DIR"

echo ""
echo "=== Docker image built successfully ==="
echo ""
echo "Test with:"
echo "  docker run --rm -v $LB_ROOT:/ligbuilder:ro \\"
echo "    -v /path/to/input:/input:ro -v /path/to/output:/output \\"
echo "    $DOCKER_IMAGE cavity /input/protein.pdb /output \\"
echo "    --center 37.3 31.2 44.1 --size 15 15 15"
echo ""
echo "Full pipeline:"
echo "  docker run --rm -v $LB_ROOT:/ligbuilder:ro \\"
echo "    -v /path/to/input:/input:ro -v /path/to/output:/output \\"
echo "    $DOCKER_IMAGE full /input/protein.pdb /output \\"
echo "    --center 37.3 31.2 44.1 --size 15 15 15 \\"
echo "    --sessions 5 --molecules 500"
