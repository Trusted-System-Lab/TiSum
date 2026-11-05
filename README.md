# TiSum

> **Artifact for:** *Local Nonlinear Substitution for Bounding Activation Functions*
> Under submission, ICSE 2027

TiSum verifies neural network safety properties through quadratic symbolic bound propagation.
Supports linear (LinProp), quadratic (QuadProp), and adaptive (AdaProp) relaxation modes
with branch-and-bound search.

## Requirements

| Category | Detail |
|----------|--------|
| OS | Linux (tested on Ubuntu 22.04) |
| Python | 3.11 |
| Conda | Miniconda / Anaconda |
| GPU | CUDA-capable, >= 8 GB VRAM (tested on NVIDIA A6000 48GB) |
| Gurobi | Valid license (academic free) |
| Disk | ~8 GB for data download |

## Quick Start (~5 min)

```bash
# 1. Setup environment + download data (~8 GB)
bash setup.sh
conda activate tisum

# 2. Smoke test — verifies 1 sample per configuration
bash run_smoke.sh
```

## Reproducing Paper Results

All experiments expect `conda activate tisum` and `cd experiments/`.
All times below measured on an **NVIDIA A6000 (48 GB)** with single-GPU sequential execution.

### Experiment 1: Sound Local Robustness
Certifies local robustness of 28 network configurations (14 MNIST + 14 CIFAR10),
100 random samples each.

```bash
bash exp_local_robust.sh   # ~1.5–2 hours
```

Output: `logs/local_robust/log_{net}_{radius}_{ts}.log`

### Experiment 2: Output Bounds Comparison
7 MNIST networks, radius sweeps (50 radii each), comparing linear vs quadratic relaxation.

```bash
bash exp_output_bounds.sh  # ~2–4 hours
```

Output: `logs/output_bounds/{net}/quad/log_{net}_{radius}_{ts}.log`

### Experiment 3: Complete Verification (VNNCOMP 2024)
Branch-and-bound verification on 4 VNNCOMP-2024 benchmarks × 3 propagation methods.

```bash
bash exp_vnncomp2024.sh    # ~6 hours
```

Output: `logs/vnncomp2024/{benchmark}/results_{ts}_{method}.csv`

## Generating Tables & Figures

After experiments complete, run post-processing scripts to produce paper-ready tables and plots:

```bash
cd experiments

# Local robustness: comparison table (vs DeepPoly/CROWN/α-CROWN/PRIMA)
python make/make_local_robust.py

# Output bounds: linear-vs-quadratic errorbar plots per network
python make/make_output_bounds.py

# VNNCOMP: baseline comparison table + ablation table + cactus plots
python make/make_vnncomp2024.py
```

| Output | Location |
|--------|----------|
| CSV tables | `experiments/tabs/vnncomp24_baselines_and_tisum_results.csv`, `ablation_study.csv`, `incomplete_verification.csv` |
| PDF figures | `experiments/imgs/vnncomp24/*.pdf`, `experiments/imgs/output_bounds/*.pdf` |

## Expected Outputs

### Smoke test
- Local robustness: 2 networks verified, each prints `Results Summary` with SAFE/UNSAFE counts (~1 min)
- VNNCOMP: 4 benchmarks, 1st property each, prints verification result to stdout (~2–3 min)
- Output bounds: 4 networks × 1 radius, prints per-sample bounds to stdout (~3–5 min)

### Archived reference results
Pre-generated logs, tables, and figures are available in `TiSum_archived_ICSE2027_logs.zip`.
Extract to `data/` to obtain:

- `data/imgs/` — Paper PDF figures (output bounds plots, cactus plots)
- `data/tabs/` — Paper CSV tables (ablation study, baseline comparison)
- `data/pre_stored/` — Baseline method results (DeepPoly, CROWN, α-CROWN, PRIMA)
- `data/log_archived/` — Complete experiment logs for verification

## Project Structure

```
├── setup.sh                # Environment + data setup
├── run_smoke.sh            # Smoke test (~5 min)
├── run_full.sh             # Full experiment suite
├── environment.yaml        # Conda environment (tisum)
├── requirements.txt        # Python deps (pip reference)
├── src/                    # Bound propagation engine
├── experiments/            # Run scripts, configs, post-processing
│   ├── exp_*.sh / exp_*.py # Experiment entry points
│   ├── configs/            # 16 YAML configs
│   ├── run/                # Multi-process launchers
│   ├── make/               # Table & figure generation
│   └── logs/               # Runtime output (gitignored)
└── data/                   # External data (downloaded via setup.sh, ~8 GB)
```

## License

MIT License — see [LICENSE](LICENSE).
