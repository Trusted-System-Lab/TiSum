#!/bin/bash
# Smoke test: run a minimal subset of each experiment type.
# Expects network models in ../data/nets/ and VNNCOMP benchmarks in ../data/vnncomp_benchmarks/.
set -e
cd experiments

echo "=== Smoke Test: TiSum ==="
echo ""

echo "--- Local Robustness (1 sample per network) ---"
python exp_smoke_local_robust.py ../data/nets/relu/mnist_relu_3_200.onnx mnist 0.03
python exp_smoke_local_robust.py ../data/nets/relu/cifar_relu_3_200.onnx cifar10 0.005

echo ""
echo "--- VNNCOMP (1st property per benchmark) ---"
python exp_smoke_vnncomp2024.py acasxu_2023 quad
python exp_smoke_vnncomp2024.py safenlp quad
python exp_smoke_vnncomp2024.py tllverifybench_2023 quad
python exp_smoke_vnncomp2024.py linearizenn quad

echo ""
echo "--- Output Bounds (3 radii) ---"
RADII_SMALL="0.01"
NETS=(
    "relu/mnist_relu_3_200"
    "relu/mnist_relu_convbase"
    "sigmoid/mnist_sigmoid_3_200"
    "tanh/mnist_tanh_3_200"
)
for net in "${NETS[@]}"; do
    echo "  $net"
    python exp_smoke_output_bounds.py "../data/nets/${net}.onnx" mnist $RADII_SMALL
done

echo ""
echo "=== Smoke test complete ==="
