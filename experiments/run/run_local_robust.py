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
    "relu/mnist_relu_9_200.onnx",
    "relu/mnist_relu_convbase.onnx",
    "relu/mnist_relu_convdeep.onnx",
    "sigmoid/mnist_sigmoid_3_200.onnx",
    "sigmoid/mnist_sigmoid_9_200.onnx",
    "sigmoid/mnist_sigmoid_convbase.onnx",
    "sigmoid/mnist_sigmoid_convdeep.onnx",
    "tanh/mnist_tanh_3_200.onnx",
    "tanh/mnist_tanh_9_200.onnx",
    "tanh/mnist_tanh_convbase.onnx",
    "tanh/mnist_tanh_convdeep.onnx",
    "relu/mnist_relu_convmaxpoolbase.onnx",
    "relu/mnist_relu_convmaxpooldeep.onnx",
]
mnist_perturb_radius_list = [
    0.03,  # mnist_relu_3_200.onnx
    0.03,  # mnist_relu_9_200.onnx
    0.08,  # mnist_relu_convbase.onnx
    0.015,  # mnist_relu_convdeep.onnx
    0.02,  # mnist_sigmoid_3_200.onnx
    0.02,  # mnist_sigmoid_9_200.onnx
    0.04,  # mnist_sigmoid_convbase.onnx
    0.02,  # mnist_sigmoid_convdeep.onnx
    0.01,  # mnist_tanh_3_200.onnx
    0.005,  # mnist_tanh_9_200.onnx
    0.03,  # mnist_tanh_convbase.onnx
    0.015,  # mnist_tanh_convdeep.onnx
    0.02,  # mnist_relu_convmaxpoolbase.onnx
    0.015,  # mnist_relu_convmaxpooldeep.onnx
]

cifar_net_path_list = [
    "relu/cifar_relu_3_200.onnx",
    "relu/cifar_relu_9_200.onnx",
    "relu/cifar_relu_convbase.onnx",
    "relu/cifar_relu_convdeep.onnx",
    "sigmoid/cifar_sigmoid_3_200.onnx",
    "sigmoid/cifar_sigmoid_9_200.onnx",
    "sigmoid/cifar_sigmoid_convbase.onnx",
    "sigmoid/cifar_sigmoid_convdeep.onnx",
    "tanh/cifar_tanh_3_200.onnx",
    "tanh/cifar_tanh_9_200.onnx",
    "tanh/cifar_tanh_convbase.onnx",
    "tanh/cifar_tanh_convdeep.onnx",
    "relu/cifar_relu_convmaxpoolbase.onnx",
    "relu/cifar_relu_convmaxpooldeep.onnx",
]
cifar_perturb_radius_list = [
    0.005,  # for cifar_relu_3_200.onnx
    0.003,  # for cifar_relu_9_200.onnx
    0.01,  # for cifar_relu_convbase.onnx
    0.0015,  # for cifar_relu_convdeep.onnx
    0.001,  # for cifar_sigmoid_3_200.onnx
    0.001,  # for cifar_sigmoid_9_200.onnx
    0.006,  # for cifar_sigmoid_convbase.onnx
    0.0015,  # for cifar_sigmoid_convdeep.onnx
    0.002,  # for cifar_tanh_3_200.onnx
    0.001,  # for cifar_tanh_9_200.onnx
    0.006,  # for cifar_tanh_convbase.onnx
    0.0015,  # for cifar_tanh_convdeep.onnx
    0.003,  # for cifar_convmaxpoolbase.onnx
    0.001,  # for cifar_convmaxpooldeep.onnx
]

if dataset == "mnist":
    net_path_list = mnist_net_path_list
    perturb_radius_list = mnist_perturb_radius_list
elif dataset == "cifar10":
    net_path_list = cifar_net_path_list
    perturb_radius_list = cifar_perturb_radius_list
else:
    raise ValueError(f"Unsupported dataset: {dataset}")


log_dir = Path("./logs/local_robust")
log_dir.mkdir(exist_ok=True)

for net_path, radius in zip(net_path_list, perturb_radius_list):
    net_path = Path(net_dir_path + net_path)
    net_name = net_path.stem

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"log_{net_name}_{radius}_{timestamp}.log"

    print(f"Running network: {net_path} with perturbation radius: {radius}")

    start_time = time.perf_counter()
    with open(log_file, "w") as f:
        f.write(f"# network: {net_path}\n")
        f.write(f"# dataset: {dataset}\n")
        f.write(f"# radius: {radius}\n")
        f.write(f"# timestamp: {timestamp}\n")
        f.write("-" * 80 + "\n")
        subprocess.run(
            ["python3", "exp_local_robust.py", net_path, dataset, str(radius)],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        print(f"Log file saved to: {log_file}")
    end_time = time.perf_counter()

    print(f"Experiment completed in {end_time - start_time:.2f} seconds.")
