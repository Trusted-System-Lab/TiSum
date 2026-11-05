#!/bin/bash
set -euo pipefail

python3 run/run_local_robust.py mnist
python3 run/run_local_robust.py cifar10