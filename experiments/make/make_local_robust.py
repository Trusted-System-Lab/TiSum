import csv
import os
from collections import defaultdict

_BASE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.join(_BASE, "..")
_REPO = os.path.join(_EXP, "..")

logs_root = os.path.join(_EXP, "logs", "local_robust")
pre_stored_file = os.path.join(_REPO, "data", "pre_stored", "local_robust", "incomplete_verification_baselines.csv")

# Network metadata: (dataset, activation, network_short_name) keyed by net_name
NETWORK_MAP = {
    # MNIST
    "mnist_relu_3_200":        ("mnist", "relu",    "fcn_base"),
    "mnist_relu_9_200":        ("mnist", "relu",    "fcn_deep"),
    "mnist_relu_convbase":     ("mnist", "relu",    "cnn_base"),
    "mnist_relu_convdeep":     ("mnist", "relu",    "cnn_deep"),
    "mnist_sigmoid_3_200":     ("mnist", "sigmoid", "fcn_base"),
    "mnist_sigmoid_9_200":     ("mnist", "sigmoid", "fcn_deep"),
    "mnist_sigmoid_convbase":  ("mnist", "sigmoid", "cnn_base"),
    "mnist_sigmoid_convdeep":  ("mnist", "sigmoid", "cnn_deep"),
    "mnist_tanh_3_200":        ("mnist", "tanh",    "fcn_base"),
    "mnist_tanh_9_200":        ("mnist", "tanh",    "fcn_deep"),
    "mnist_tanh_convbase":     ("mnist", "tanh",    "cnn_base"),
    "mnist_tanh_convdeep":     ("mnist", "tanh",    "cnn_deep"),
    "mnist_relu_convmaxpoolbase": ("mnist", "maxpool", "pool_base"),
    "mnist_relu_convmaxpooldeep": ("mnist", "maxpool", "pool_deep"),
    # CIFAR10
    "cifar_relu_3_200":        ("cifar10", "relu",    "fcn_base"),
    "cifar_relu_9_200":        ("cifar10", "relu",    "fcn_deep"),
    "cifar_relu_convbase":     ("cifar10", "relu",    "cnn_base"),
    "cifar_relu_convdeep":     ("cifar10", "relu",    "cnn_deep"),
    "cifar_sigmoid_3_200":     ("cifar10", "sigmoid", "fcn_base"),
    "cifar_sigmoid_9_200":     ("cifar10", "sigmoid", "fcn_deep"),
    "cifar_sigmoid_convbase":  ("cifar10", "sigmoid", "cnn_base"),
    "cifar_sigmoid_convdeep":  ("cifar10", "sigmoid", "cnn_deep"),
    "cifar_tanh_3_200":        ("cifar10", "tanh",    "fcn_base"),
    "cifar_tanh_9_200":        ("cifar10", "tanh",    "fcn_deep"),
    "cifar_tanh_convbase":     ("cifar10", "tanh",    "cnn_base"),
    "cifar_tanh_convdeep":     ("cifar10", "tanh",    "cnn_deep"),
    "cifar_relu_convmaxpoolbase": ("cifar10", "maxpool", "pool_base"),
    "cifar_relu_convmaxpooldeep": ("cifar10", "maxpool", "pool_deep"),
}


def read_baseline_csv(file_path):
    baselines = {}
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dataset"], row["activation"], row["network"], row["method"])
            verified = int(row["verified"])
            t = float(row["time"])
            baselines[key] = (verified, t)
    return baselines


def parse_log_filename(filename):
    if not filename.startswith("log_") or not filename.endswith(".log"):
        return None
    inner = filename[4:-4]
    parts = inner.split("_")
    if len(parts) < 5:
        return None
    timestamp = "_".join(parts[-2:])
    radius = parts[-3]
    net_name = "_".join(parts[:-3])
    return net_name, radius, timestamp


def extract_our_results(logs_dir):
    results = {}
    if not os.path.isdir(logs_dir):
        print(f"Warning: {logs_dir} does not exist")
        return results

    for fname in os.listdir(logs_dir):
        if not fname.startswith("log_") or not fname.endswith(".log"):
            continue
        parsed = parse_log_filename(fname)
        if parsed is None:
            continue
        net_name, radius, timestamp = parsed
        if net_name not in NETWORK_MAP:
            continue

        verified = 0
        total_time = 0.0
        avg_time = 0.0
        with open(os.path.join(logs_dir, fname), "r") as f:
            summary = {}
            for line in f:
                line = line.strip()
                if "Results Summary:" in line:
                    for _ in range(3):
                        line = f.readline().strip()
                        if ":" in line:
                            key, value = line.split(":", 1)
                            summary[key.strip()] = int(value.strip())
                elif "Total used time" in line:
                    total_time = float(line.split(":")[1].strip().replace("s", ""))
                elif "Average used time" in line:
                    avg_time = float(line.split(":")[1].strip().replace("s", ""))

        verified_safe = summary.get("SAFE", 0)
        if verified_safe > 0:
            verified = verified_safe

        key = (net_name, radius)
        if key not in results:
            results[key] = []
        results[key].append({
            "verified": verified,
            "total_time": total_time,
            "avg_time": avg_time,
        })

    return results


def generate_incomplete_verification_csv(
    our_results, baselines, save_path, logs_dir
):
    net_name_to_radius = {}
    for fname in os.listdir(logs_dir):
        if not fname.startswith("log_") or not fname.endswith(".log"):
            continue
        parsed = parse_log_filename(fname)
        if parsed is None:
            continue
        net_name = parsed[0]
        radius = parsed[1]
        net_name_to_radius[net_name] = radius

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "activation", "network", "method", "verified", "time"])

        for dataset, activation, networks in [
            ("mnist", "relu",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
            ("mnist", "sigmoid", ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
            ("mnist", "tanh",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
            ("mnist", "maxpool", ["pool_base", "pool_deep"]),
            ("cifar10", "relu",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
            ("cifar10", "sigmoid", ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
            ("cifar10", "tanh",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
            ("cifar10", "maxpool", ["pool_base", "pool_deep"]),
        ]:
            for net_short in networks:
                net_label = {"fcn_base": "FCN-base", "fcn_deep": "FCN-deep",
                             "cnn_base": "CNN-base", "cnn_deep": "CNN-deep",
                             "pool_base": "POOL-base", "pool_deep": "POOL-deep"}[net_short]

                for method in ["deeppoly", "crown", "alpha_crown", "prima"]:
                    base_key = (dataset, activation, net_short, method)
                    if base_key in baselines:
                        b_verified, b_time = baselines[base_key]
                    else:
                        b_verified, b_time = -1, -1.0
                    writer.writerow([dataset, activation, net_label, method, b_verified, b_time])

                net_name = None
                for nn in NETWORK_MAP:
                    if NETWORK_MAP[nn] == (dataset, activation, net_short):
                        net_name = nn
                        break
                our_verified = 0
                our_time = 0.0
                if net_name and net_name in net_name_to_radius:
                    radius = net_name_to_radius[net_name]
                    key = (net_name, radius)
                    if key in our_results and our_results[key]:
                        best = max(our_results[key], key=lambda x: x["verified"])
                        our_verified = best["verified"]
                        our_time = best["avg_time"]
                writer.writerow([dataset, activation, net_label, "tisum", our_verified, our_time])


def print_pretty_table(baselines, our_results, logs_dir):
    methods = ["deeppoly", "crown", "alpha_crown", "prima"]
    method_labels = ["DeepPoly", "CROWN", "\u03b1-CROWN", "PRIMA"]

    net_name_to_radius = {}
    for fname in os.listdir(logs_dir):
        if not fname.startswith("log_") or not fname.endswith(".log"):
            continue
        parsed = parse_log_filename(fname)
        if parsed is None:
            continue
        net_name_to_radius[parsed[0]] = parsed[1]

    def _our(net_name):
        if net_name and net_name in net_name_to_radius:
            r = net_name_to_radius[net_name]
            k = (net_name, r)
            if k in our_results and our_results[k]:
                b = max(our_results[k], key=lambda x: x["verified"])
                return b["verified"], b["total_time"]
        return 0, 0.0

    def _bl(ds, act, net, m):
        k = (ds, act, net, m)
        if k in baselines:
            return baselines[k]
        return -1, -1.0

    def _cell(v, t):
        if v < 0 and t < 0:
            return "  -      -"
        return f"{int(v):>3d} {t:>7.2f}"

    network_order = [
        ("mnist", "ReLU",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
        ("mnist", "Sigmoid", ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
        ("mnist", "Tanh",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
        ("mnist", "MaxPool", ["pool_base", "pool_deep"]),
        ("cifar10", "ReLU",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
        ("cifar10", "Sigmoid", ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
        ("cifar10", "Tanh",    ["fcn_base", "fcn_deep", "cnn_base", "cnn_deep"]),
        ("cifar10", "MaxPool", ["pool_base", "pool_deep"]),
    ]
    net_labels = {"fcn_base": "FCN-base", "fcn_deep": "FCN-deep",
                  "cnn_base": "CNN-base", "cnn_deep": "CNN-deep",
                  "pool_base": "POOL-base", "pool_deep": "POOL-deep"}

    sep = "=" * 79
    print(sep)
    print("  Local Robustness Verification")
    print(sep)
    hdr = "{:<7s} {:<9s}  {:>3s} {:>7s}  {:>3s} {:>7s}  {:>3s} {:>7s}  {:>3s} {:>7s}  {:>3s} {:>7s}".format(
        "Activ", "Network", "DeepPoly", "", "CROWN", "", "\u03b1-CROWN", "", "PRIMA", "", "TiSum", "(Ours)")
    print(hdr)
    hdr2 = "{:<7s} {:<9s}  {:>3s} {:>7s}  {:>3s} {:>7s}  {:>3s} {:>7s}  {:>3s} {:>7s}  {:>3s} {:>7s}".format(
        "", "", "#V", "Time(s)", "#V", "Time(s)", "#V", "Time(s)", "#V", "Time(s)", "#V", "Time(s)")
    print(hdr2)
    print(sep)

    cur_ds = None
    row_fmt = "{:<7s} {:<9s}  {}  {}  {}  {}  {}"
    for ds, act_label, nets in network_order:
        ds_tag = ds.upper()
        if ds_tag != cur_ds:
            print(f"  {ds_tag}")
            print(sep)
            cur_ds = ds_tag

        for i, ns in enumerate(nets):
            a = act_label if i == 0 else ""
            nl = net_labels[ns]
            cells = [_cell(*_bl(ds, act_label.lower(), ns, m)) for m in methods]
            nn = None
            for k, v in NETWORK_MAP.items():
                if v == (ds, act_label.lower(), ns):
                    nn = k
                    break
            cells.append(_cell(*_our(nn)))
            print(row_fmt.format(a, nl, *cells))

        print(sep)


if __name__ == "__main__":
    baselines = read_baseline_csv(pre_stored_file)
    print(f"Loaded baselines from {pre_stored_file}")

    our_results = extract_our_results(logs_root)
    print(f"Found {len(our_results)} network/radius combos in {logs_root}")

    print()
    print_pretty_table(baselines, our_results, logs_root)

    tabs_dir = os.path.join(_EXP, "tabs")
    os.makedirs(tabs_dir, exist_ok=True)
    generate_incomplete_verification_csv(
        our_results, baselines, os.path.join(tabs_dir, "incomplete_verification.csv"),
        logs_root,
    )
    print(f"Table written to {tabs_dir}/incomplete_verification.csv")
