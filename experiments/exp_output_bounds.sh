#!/bin/bash
set -euo pipefail

python3 run/run_output_bounds.py mnist  # 仅 mnist，与论文一致；cifar10 数据在 data/ 中保留