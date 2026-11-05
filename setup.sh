#!/bin/bash
set -euo pipefail

GDRIVE_ID="1hqBU7RqradOPMq88byANlqIOVakE7vxJ"
DATA_ZIP="tisum_data.zip"

echo "=== TiSum Setup ==="

# 1. Create conda environment
if conda env list | grep -q "^tisum "; then
    echo "[SKIP] Conda environment 'tisum' already exists."
else
    echo "[1/3] Creating conda environment 'tisum'..."
    conda env create -f environment.yaml
fi

# 2. Download data from Google Drive
if [ -d "data/nets" ] && [ -d "data/vnncomp_benchmarks" ]; then
    echo "[SKIP] Data directory already populated."
else
    echo "[2/3] Downloading data from Google Drive..."
    conda run -n tisum pip install -q gdown
    conda run -n tisum gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "${DATA_ZIP}"

    echo "[3/3] Extracting data..."
    unzip -qo "${DATA_ZIP}"
    rm "${DATA_ZIP}"
fi

echo ""
echo "Setup complete. Run: conda activate tisum"
