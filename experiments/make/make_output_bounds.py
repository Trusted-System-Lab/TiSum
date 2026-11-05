import os
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "pgf.rcfonts": False,
    }
)

_BASE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.join(_BASE, "..")
_REPO = os.path.join(_EXP, "..")

logs_root = os.path.join(_EXP, "logs", "output_bounds")
archived_root = os.path.join(_REPO, "data", "log_archived", "output_bounds")


def get_logs(logs_dir):
    if not os.path.isdir(logs_dir):
        return []
    files = os.listdir(logs_dir)
    return [f for f in files if f.startswith("log_") and f.endswith(".log")]


def filter_logs(logs, logs_dir, remove_old=False):
    log_dict = defaultdict(list)
    for log in logs:
        parts = log.split("_")
        if len(parts) < 5:
            continue
        net_parts = parts[1:-3]
        net_name = "_".join(net_parts)
        perturb_radius = parts[-3]
        log_dict[(net_name, perturb_radius)].append(log)

    for key, log_list in log_dict.items():
        log_list.sort(
            key=lambda x: datetime.strptime(
                ("_".join(x.split("_")[-2:])).split(".")[0], "%Y-%m-%d_%H-%M-%S"
            ),
            reverse=True,
        )
        log_dict[key] = log_list[:1]
        if remove_old:
            for log in log_list[1:]:
                os.remove(os.path.join(logs_dir, log))
    filtered_logs = [log for log_list in log_dict.values() for log in log_list]
    return filtered_logs


def read_and_get_results(logs, logs_dir):
    results = {}
    for idx, log in enumerate(logs):
        eps = float(log.split("_")[-3])
        lb_list = []
        ub_list = []
        with open(os.path.join(logs_dir, log), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("[COLLECT] Final output bounds:"):
                    lb_text = f.readline().strip()
                    ub_text = f.readline().strip()
                    lb_list.append(eval(lb_text))
                    ub_list.append(ub_text)

        lbs = np.asarray(lb_list)
        if len(lbs) == 0:
            results[eps] = (0, 0, 0, 0, 0, 0, 0, 0)
            continue
        ubs_list = []
        for ub_text_line in ub_list:
            try:
                ubs_list.append(eval(ub_text_line))
            except Exception:
                pass
        ubs = np.asarray(ubs_list) if ubs_list else lbs

        lbs_mean = np.mean(lbs).tolist()
        ubs_mean = np.mean(ubs).tolist()
        lbs_std = np.std(lbs).tolist()
        ubs_std = np.std(ubs).tolist()
        lbs_max = np.max(lbs).tolist()
        ubs_max = np.max(ubs).tolist()
        lbs_min = np.min(lbs).tolist()
        ubs_min = np.min(ubs).tolist()
        results[eps] = (
            lbs_mean, lbs_std, lbs_min, lbs_max,
            ubs_mean, ubs_std, ubs_min, ubs_max,
        )

    results = dict(sorted(results.items(), key=lambda item: item[0]))

    eps_list = list(results.keys())
    lb_mean_list, lb_std_list, lb_min_list, lb_max_list = [], [], [], []
    ub_mean_list, ub_std_list, ub_min_list, ub_max_list = [], [], [], []
    for k, v in results.items():
        lb_mean, lb_std, lb_min, lb_max, ub_mean, ub_std, ub_min, ub_max = v
        lb_mean_list.append(lb_mean)
        lb_std_list.append(lb_std)
        lb_min_list.append(lb_min)
        lb_max_list.append(lb_max)
        ub_mean_list.append(ub_mean)
        ub_std_list.append(ub_std)
        ub_min_list.append(ub_min)
        ub_max_list.append(ub_max)

    return {
        "eps": eps_list,
        "lb_mean": lb_mean_list, "lb_std": lb_std_list,
        "lb_min": lb_min_list, "lb_max": lb_max_list,
        "ub_mean": ub_mean_list, "ub_std": ub_std_list,
        "ub_min": ub_min_list, "ub_max": ub_max_list,
    }


def plot_errorbars(data, save_path=None):
    assert len(data) == 2
    data1, data2 = data[0], data[1]

    x = data1["eps"]
    l1_mean, l1_std, l1_min, l1_max = data1["lb_mean"], data1["lb_std"], data1["lb_min"], data1["lb_max"]
    u1_mean, u1_std, u1_min, u1_max = data1["ub_mean"], data1["ub_std"], data1["ub_min"], data1["ub_max"]
    l2_mean, l2_std, l2_min, l2_max = data2["lb_mean"], data2["lb_std"], data2["lb_min"], data2["lb_max"]
    u2_mean, u2_std, u2_min, u2_max = data2["ub_mean"], data2["ub_std"], data2["ub_min"], data2["ub_max"]

    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.98, wspace=0, hspace=0)
    plt.tight_layout()

    linear_color = "blue"
    quad_color = "red"
    linear_linestyle = "dotted"
    quad_linestyle = "-"
    lower_fmt = "s"
    upper_fmt = "^"
    alpha = 0.2

    kwargs = {"fmt": lower_fmt, "color": linear_color, "linestyle": linear_linestyle}
    plt.errorbar(x, l1_mean, yerr=l1_std, label="Lower Bound (Linear)", **kwargs)
    kwargs = {"fmt": upper_fmt, "color": linear_color, "linestyle": linear_linestyle}
    plt.errorbar(x, u1_mean, yerr=u1_std, label="Upper Bound (Linear)", **kwargs)

    kwargs = {"color": linear_color, "alpha": alpha}
    plt.fill_between(x, l1_min, l1_max, label="Bound Range (Linear)", **kwargs)
    plt.fill_between(x, u1_min, u1_max, **kwargs)

    kwargs = {"fmt": lower_fmt, "color": quad_color, "linestyle": quad_linestyle}
    plt.errorbar(x, l2_mean, yerr=l2_std, label="Lower Bound (Quadratic)", **kwargs)
    kwargs = {"fmt": upper_fmt, "color": quad_color, "linestyle": quad_linestyle}
    plt.errorbar(x, u2_mean, yerr=u2_std, label="Upper Bound (Quadratic)", **kwargs)

    kwargs = {"color": quad_color, "alpha": alpha}
    plt.fill_between(x, l2_min, l2_max, label="Bound Range (Quadratic)", **kwargs)
    plt.fill_between(x, u2_min, u2_max, **kwargs)

    plt.xlabel("Perturbation Radii", fontsize=38)
    plt.ylabel("Bounds", fontsize=38)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.gca().yaxis.get_offset_text().set_fontsize(28)

    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_and_save_data(net_name):
    data_list = []
    for subdir in ["linear", "quad"]:
        subdir_path = os.path.join(logs_root, net_name, subdir)
        if not os.path.isdir(subdir_path):
            subdir_path = os.path.join(archived_root, net_name, subdir)
        logs = get_logs(subdir_path)
        logs = filter_logs(logs, subdir_path, remove_old=False)
        data = read_and_get_results(logs, subdir_path)
        data_list.append(data)
    data_tuple = tuple(data_list)
    imgs_dir = os.path.join(_EXP, "imgs", "output_bounds")
    save_path = os.path.join(imgs_dir, f"output_bounds_{net_name}.pdf")
    os.makedirs(imgs_dir, exist_ok=True)
    plot_errorbars(data_tuple, save_path=save_path)
    print(f"  plot saved to {save_path}")
    return save_path


if __name__ == "__main__":
    subdir_list = [
        "mnist_relu_3_200", "mnist_relu_convbase",
        "mnist_sigmoid_3_200", "mnist_sigmoid_convbase",
        "mnist_tanh_3_200", "mnist_tanh_convbase",
        "mnist_relu_convmaxpoolbase",
    ]

    generated = []
    for i, net_name in enumerate(subdir_list):
        print(f"[{i+1}/{len(subdir_list)}] {net_name}...", end=" ")
        p = plot_and_save_data(net_name)
        generated.append(p)

    print(f"Generated {len(generated)} figures in {os.path.join(_EXP, 'imgs', 'output_bounds')}/")
