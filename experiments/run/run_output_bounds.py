"""
Script to run experiments for output bounds with different perturbation radii.
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

arg = argparse.ArgumentParser(
    description="Experiment about output bounds for different perturbation radius."
)
arg.add_argument(
    "dataset", type=str, help="The dataset to use, only for 'mnist' and 'cifar10'."
)

args = arg.parse_args()
dataset = args.dataset

if dataset not in ["mnist", "cifar10"]:
    raise ValueError(
        f"Unsupported dataset: {dataset}. Supported datasets are 'mnist' and 'cifar10'."
    )

net_dir_path = "../data/nets/"
mnist_net_path_list = [
    "relu/mnist_relu_3_200.onnx",
    "relu/mnist_relu_convbase.onnx",
    "relu/mnist_relu_convdeep.onnx",
    "sigmoid/mnist_sigmoid_3_200.onnx",
    "sigmoid/mnist_sigmoid_convbase.onnx",
    "sigmoid/mnist_sigmoid_convdeep.onnx",
    "tanh/mnist_tanh_3_200.onnx",
    "tanh/mnist_tanh_convbase.onnx",
    "tanh/mnist_tanh_convdeep.onnx",
    "relu/mnist_relu_convmaxpoolbase.onnx",
    "relu/mnist_relu_convmaxpooldeep.onnx",
]
cifar10_net_path_list = [
    "relu/cifar_relu_3_200.onnx",
    "relu/cifar_relu_convbase.onnx",
    "relu/cifar_relu_convdeep.onnx",
    "sigmoid/cifar_sigmoid_3_200.onnx",
    "sigmoid/cifar_sigmoid_convbase.onnx",
    "sigmoid/cifar_sigmoid_convdeep.onnx",
    "tanh/cifar_tanh_3_200.onnx",
    "tanh/cifar_tanh_convbase.onnx",
    "tanh/cifar_tanh_convdeep.onnx",
    "relu/cifar_relu_convmaxpoolbase.onnx",
    "relu/cifar_relu_convmaxpooldeep.onnx",
]

if dataset == "mnist":
    net_path_list = mnist_net_path_list
elif dataset == "cifar10":
    net_path_list = cifar10_net_path_list
else:
    raise ValueError(f"Unsupported dataset: {dataset}.")


log_root = Path("logs/output_bounds")
log_root.mkdir(exist_ok=True)

relu_radius_list = [round(0.01 * i, 3) for i in range(1, 51)]
sigmoid_radius_list = [round(0.001 * i, 3) for i in range(1, 51)]
tanh_radius_list = [round(0.001, 3) * i for i in range(1, 51)]

for net_path_str in net_path_list:
    net_path = Path(net_dir_path + net_path_str)
    net_name = net_path.stem  # e.g., "cifar_relu_3_200"

    if "relu" in net_name:
        radius_list = relu_radius_list
    elif "sigmoid" in net_name:
        radius_list = sigmoid_radius_list
    elif "tanh" in net_name:
        radius_list = tanh_radius_list
    else:
        raise ValueError(f"Unsupported network type in path: {net_path_str}.")

    dir_quad = log_root / net_name / "quad"
    dir_quad.mkdir(parents=True, exist_ok=True)

    for radius in radius_list:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"log_{net_name}_{radius}_{timestamp}.log"

        log_quad = dir_quad / log_filename

        print("=" * 30)
        print("Running experiments for:")
        print(f"  Network: {net_path}")
        print(f"  Radius : {radius}")
        print("=" * 30)

        # Quadratic
        print(f"[Quadratic] Log file: {log_quad}")
        start = time.time()
        with open(log_quad, "w") as f:
            f.write(f"# network: {net_path}\n")
            f.write(f"# dataset: {dataset}\n")
            f.write(f"# radius: {radius}\n")
            f.write(f"# timestamp: {timestamp}\n")
            f.write(f"# method: quad\n")
            f.write("-" * 80 + "\n")
            subprocess.run(
                [
                    "python3",
                    "exp_output_bounds_quad.py",
                    str(net_path),
                    dataset,
                    str(radius),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        print(f"[Quadratic] Done in {time.time() - start:.4f} seconds")
