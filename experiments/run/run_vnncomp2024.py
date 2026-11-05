import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(description="Experiment about VNNCOMP2024 benchmarks.")
parser.add_argument("benchmark", type=str, help="The benchmark name.")
parser.add_argument("method", type=str, help="which relaxation.")
args = parser.parse_args()
benchmark = args.benchmark
method = args.method

if benchmark not in [
    "acasxu_2023",
    "safenlp",
    "tllverifybench_2023",
    "linearizenn",
]:
    raise ValueError(f"Unsupported dataset: {benchmark}.")

net_dir_path = "../data/nets/"

log_dir = Path("./logs/vnncomp2024")
log_dir.mkdir(exist_ok=True)


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f"log_{benchmark}_{method}_{timestamp}.log"

print(f"Running {benchmark}")

start_time = time.perf_counter()
with open(log_file, "w") as f:
    f.write(f"# benchmark: {benchmark}\n")
    f.write(f"# method: {method}\n")
    f.write(f"# timestamp: {timestamp}\n")
    f.write("-" * 80 + "\n")
    subprocess.run(
        ["python3", "exp_vnncomp2024.py", benchmark, method, log_dir, timestamp],
        stdout=f,
        stderr=subprocess.STDOUT,
    )
    print(f"Log file saved to: {log_file}")
end_time = time.perf_counter()

print(f"Experiment completed in {end_time - start_time:.2f} seconds.")
