#!/bin/bash
set -euo pipefail

python3 run/run_vnncomp2024.py linearizenn linear
python3 run/run_vnncomp2024.py tllverifybench_2023 linear
python3 run/run_vnncomp2024.py acasxu_2023 linear
python3 run/run_vnncomp2024.py safenlp linear

python3 run/run_vnncomp2024.py linearizenn quad
python3 run/run_vnncomp2024.py tllverifybench_2023 quad
python3 run/run_vnncomp2024.py acasxu_2023 quad
python3 run/run_vnncomp2024.py safenlp quad

python3 run/run_vnncomp2024.py linearizenn ada
python3 run/run_vnncomp2024.py tllverifybench_2023 ada
python3 run/run_vnncomp2024.py acasxu_2023 ada
python3 run/run_vnncomp2024.py safenlp ada
