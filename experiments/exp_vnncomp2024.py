import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import VNNLIBVerifier
from vnncomp_args import get_arguments_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a script to verify neural networks using the TiSum library."
    )
    parser.add_argument("benchmark", type=str, help="Benchmark name.")
    parser.add_argument("method", type=str, help="which relaxation.")
    parser.add_argument(
        "results_instances_csv_dir", type=str, help="Directory to save results CSV."
    )
    parser.add_argument("timestamp", type=str, help="Timestamp for results files.")

    args = parser.parse_args()
    target_dir_path = args.benchmark
    method = args.method
    results_instances_csv_dir = args.results_instances_csv_dir
    timestamp = args.timestamp

    vnncomp_dir = "../data/vnncomp_benchmarks/"
    if not os.path.exists(vnncomp_dir):
        raise FileNotFoundError(
            f"The specified VNNComp benchmarks directory does not exist: {vnncomp_dir}"
        )

    target_dir_path = vnncomp_dir + target_dir_path
    category = os.path.basename(target_dir_path)
    shared_kwargs = {"use_vnnlib": True}
    if method == "linear":
        suffix = "_linear"
    elif method == "quad":
        suffix = "_quad"
    elif method == "ada":
        suffix = "_ada"
    else:
        raise ValueError(f"Unsupported method: {method}")
    arguments_list = get_arguments_list(category, shared_kwargs, suffix=suffix)
    for argument in arguments_list:
        print(argument)
    print("-" * 80)
    verifier = VNNLIBVerifier(
        target_dir_path,
        arguments_list,
        apply_slimonnx=True,
        apply_torchonnx=True,
        apply_torchvnnlib=True,
        record_result=True,
        results_instances_csv_dir=results_instances_csv_dir,
        results_timestamp=f"{timestamp}{suffix}",
        verbose=True,
    )

    verifier.verify()
