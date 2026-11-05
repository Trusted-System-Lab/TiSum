import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src
from src.substorch import RelaxMode  # noqa
from src.args import *  # noqa
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a script to verify neural networks using the TiSum library."
    )

    parser.add_argument("net_path", type=str, help="Path to the ONNX model file.")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to use for " "verification (e.g., 'mnist', 'cifar10').",
    )
    parser.add_argument(
        "perturb_radius", type=float, help="Perturbation radius for the input data."
    )

    args = parser.parse_args()

    net_path = args.net_path
    dataset = args.dataset
    perturb_radius = args.perturb_radius

    shared_kwargs = {
        "use_dataset": True,
        "perturb_radius": perturb_radius,
        "dataset": dataset,
        "device": "cuda:0",
        "dtype": "float32",
    }

    template_argument = [
        OriModelArgs(**shared_kwargs),
        QuadPropArgs(
            **shared_kwargs,
            store_all_bnds=True,
            refine_by_scalarize_each_layer=True,
            refine_by_interval_arithmetic=True,
            act_relax_mode=RelaxMode.SINGLE_QUAD,
        ),
        LinPropArgs(**shared_kwargs),
    ]

    verifier = src.DatasetVerifier(
        net_path,
        template_argument,
        num_samples=100,
        apply_slimonnx=True,
        apply_torchonnx=True,
        ignore_incorrectly_classified=True,
        verbose=True,
    )

    verifier.verify()
