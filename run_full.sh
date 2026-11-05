#!/bin/bash
# Full experiment suite.
set -e
cd experiments

echo "=== Full Experiment Run: TiSum ==="
echo ""

echo "--- Complete Verification (VNNCOMP 2024) ---"
bash exp_vnncomp2024.sh

echo ""
echo "--- Sound Local Robustness ---"
bash exp_local_robust.sh

echo ""
echo "--- Output Bounds Comparison ---"
bash exp_output_bounds.sh

echo ""
echo "=== Full experiment run complete ==="
