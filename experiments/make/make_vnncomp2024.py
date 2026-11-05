import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

logs_root = os.path.join(_EXP, "logs", "vnncomp2024")
pre_stored_dir = os.path.join(_REPO, "data", "pre_stored", "vnncomp2024")
baseline_results_file = os.path.join(pre_stored_dir, "regular_track_results.txt")
categories = ["acasxu_2023", "safenlp", "tllverifybench_2023", "linearizenn"]
suffixes = ["_linear", "_quad", "_ada"]
our_results_files = []
for category in categories:
    cat_dir = os.path.join(logs_root, category)
    if not os.path.isdir(cat_dir):
        continue
    for sfx in suffixes:
        matched = [f for f in os.listdir(cat_dir) if f.endswith(f"{sfx}.csv")]
        if matched:
            our_results_files.append(os.path.join(cat_dir, sorted(matched)[-1]))
tools = [
    "$\\alpha\\beta$-CROWN",
    "PyRAT",
    "Marabou",
    "nnenum",
    "NNV",
    "NeVer2",
    "CORA",
    "NeuralSAT",
    "Ours (TiSum)",
]
our_tools = ["Ours (Linear)", "Ours (Optimizable)", "Ours (TiSum)"]
linestyles = [
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5)),
    (0, (5, 10)),
    (0, (1, 1)),
    (0, (5, 1)),
    (0, (7, 3)),
    (0, (2, 2, 8, 2)),
    "-",
]
colors = [
    "#228B22",
    "#FF8C00",
    "#C71585",
    "#20B2AA",
    "#BDB76B",
    "#FF69B4",
    "#1E90FF",
    "#6A5ACD",
    "#FF0000",
]
our_linestyles = [":", "-.", "-"]
our_colors = ["#1E90FF", "#32CD32", "#FF0000"]


def read_baseline_results(file_path):
    ori_categories = ["Acasxu 2023", "Safenlp", "Tllverifybench 2023", "Linearizenn"]
    results = {tool: {category: {} for category in categories} for tool in tools[:8]}

    with open(file_path, "r") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(" & ")
            category, idx = parts[0], parts[1]
            if category not in ori_categories:
                continue
            category = categories[ori_categories.index(category)]
            for i, tool in enumerate(tools[:8]):
                tool_result = parts[3 + i]
                results[tool][category][idx] = tool_result

    return results


def summarize_baseline_results(results):
    time_lists = {tool: {} for tool in results}
    unknown_counts = {tool: {} for tool in results}
    error_counts = {tool: {} for tool in results}

    for tool, category_result in results.items():
        for category, idx_result in category_result.items():
            time_lists[tool][category] = []
            unknown_counts[tool][category] = 0
            error_counts[tool][category] = 0

            for result in idx_result.values():
                try:
                    time_lists[tool][category].append(float(result))
                except ValueError:
                    if result == "-":
                        unknown_counts[tool][category] += 1
                    elif result == "error":
                        error_counts[tool][category] += 1
                    else:
                        raise ValueError(f"Unknown result {result}")

    return time_lists, unknown_counts, error_counts


def read_our_results(csv_paths):
    time_lists = {}
    time_lists["Ours (Linear)"] = {}
    time_lists["Ours (TiSum)"] = {}
    time_lists["Ours (Optimizable)"] = {}
    unknown_counts = {}
    unknown_counts["Ours (Linear)"] = {}
    unknown_counts["Ours (TiSum)"] = {}
    unknown_counts["Ours (Optimizable)"] = {}

    for csv_path in csv_paths:
        parts = csv_path.split("/")
        category = parts[-2]
        if "linear" in parts[-1]:
            method = "Ours (Linear)"
        elif "quad" in parts[-1]:
            method = "Ours (TiSum)"
        elif "ada" in parts[-1]:
            method = "Ours (Optimizable)"
        else:
            raise ValueError(f"Unknown method in csv path {csv_path}")

        time_list, unknown_count = summarize_our_results(csv_path)
        time_lists[method][category] = time_list
        unknown_counts[method][category] = unknown_count

    return time_lists, unknown_counts


def summarize_our_results(file_path):
    times_list = []
    unknown_count = 0

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            timelimit, result, cost_time = float(parts[-3]), parts[-2], float(parts[-1])
            if result == "unknown" or cost_time > timelimit:
                unknown_count += 1
            elif result == "error":
                raise ValueError(f"Error result in {file_path}")
            else:
                times_list.append(cost_time)

    return times_list, unknown_count


def count_csv_results(csv_path, timelimit):
    solved = 0
    unknown = 0
    error = 0
    total_time = 0.0
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            tl, result, cost_time = float(parts[-3]), parts[-2], float(parts[-1])
            if result == "unknown" or cost_time > tl:
                unknown += 1
            elif result == "error":
                error += 1
            else:
                solved += 1
                total_time += cost_time
    avg_time = total_time / solved if solved > 0 else 0.0
    return solved, unknown, error, avg_time


def generate_baseline_csv(csv_paths, save_path):
    tools_names = [
        ("alpha-beta-CROWN", 0), ("PyRAT", 1), ("Marabou", 2),
        ("nnenum", 3), ("NNV", 4), ("NeVer2", 5),
        ("CORA", 6), ("NeuralSAT", 7), ("TiSum", 8),
    ]
    baseline_results = read_baseline_results(baseline_results_file)
    time_lists, unknown_counts, error_counts = summarize_baseline_results(baseline_results)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "tool", "solved", "unknown", "error", "avg_time"])
        for category in categories:
            for tool_name, tool_idx in tools_names:
                if tool_idx < 8:
                    tl = time_lists[tools[tool_idx]].get(category, [])
                    unk = unknown_counts[tools[tool_idx]].get(category, 0)
                    err = error_counts[tools[tool_idx]].get(category, 0)
                    solved = len(tl)
                    avg_t = sum(tl) / solved if solved > 0 else 0.0
                else:
                    csv_path = [p for p in csv_paths if "quad" in p and category in p]
                    if csv_path:
                        solved, unk, err, avg_t = count_csv_results(csv_path[0], 600)
                    else:
                        solved, unk, err, avg_t = 0, 0, 0, 0.0
                writer.writerow([category, tool_name, solved, unk, err, round(avg_t, 2)])


def generate_ablation_csv(csv_paths, save_path):
    cat_files = {
        "acasxu_2023": [p for p in csv_paths if "acasxu_2023" in p],
        "safenlp": [p for p in csv_paths if "safenlp" in p],
        "tllverifybench_2023": [p for p in csv_paths if "tllverifybench" in p],
        "linearizenn": [p for p in csv_paths if "linearizenn" in p],
    }
    method_order = ["linear", "ada", "quad"]
    method_labels = ["linear", "optimizable", "quad"]

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "method", "solved", "unknown", "branches", "avg_time"])
        for ci, category in enumerate(categories):
            for fi, ms in enumerate(method_order):
                method_file = [p for p in cat_files[category] if p.endswith(f"_{ms}.csv")]
                if method_file:
                    solved, unk, err, avg_t = count_csv_results(method_file[0], 600)
                    branch_count = avg_branches_per_solved(method_file[0], category, ms)
                else:
                    solved, unk, err, avg_t = 0, 0, 0, 0.0
                    branch_count = 0
                writer.writerow([category, method_labels[fi], solved, unk, branch_count, round(avg_t, 2)])


def avg_branches_per_solved(csv_path, category, method):
    results = []
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            result = parts[-2]
            results.append(result)

    log_files = []
    for fname in os.listdir(logs_root):
        if not (fname.startswith("log_") and fname.endswith(".log")):
            continue
        if category not in fname:
            continue
        if f"_{method}_" not in fname:
            continue
        log_files.append(os.path.join(logs_root, fname))

    instance_branches = {}
    for path in log_files:
        current_instance = -1
        instance_branch_total = 0
        with open(path, "r") as f:
            for line in f:
                if line.startswith("Instance-"):
                    if current_instance >= 0 and instance_branch_total > 0:
                        instance_branches[current_instance] = instance_branch_total
                    current_instance += 1
                    instance_branch_total = 0
                elif line.startswith("Total"):
                    parts = line.split()
                    if len(parts) > 3 and parts[2:4] == ["branches", "processed,"]:
                        instance_branch_total = int(parts[1])
            if current_instance >= 0 and instance_branch_total > 0:
                instance_branches[current_instance] = instance_branch_total

    solved_branches = []
    for i, result in enumerate(results):
        if result in ("sat", "unsat"):
            solved_branches.append(instance_branches.get(i, 0))

    if not solved_branches:
        return 0
    return round(sum(solved_branches) / len(solved_branches))


def _v(x):
    if x < 0:
        return "-"
    return str(x)

def _tv(x):
    if x < 0:
        return "-"
    return f"{x:>7.2f}"

def _tool_cell(solved, unknown, error, avg_time):
    f = "{:>5s} {:>5s} {:>5s} {:>7s}"
    return f.format(_v(solved), _v(unknown), _v(error), _tv(avg_time))

# Known N/A (tool, category) pairs where tool does not support the benchmark
NA_MAP = {
    ("NNV", "linearizenn"), ("NeVer2", "linearizenn"),
    ("CORA", "linearizenn"), ("NeuralSAT", "linearizenn"),
}

def _row_data(t, cat, tl, uc, ec, our_results_files):
    if (t, cat) in NA_MAP:
        return (-1, -1, -1, -1.0)
    if t == tools[8]:
        csv_path = [p for p in our_results_files if "quad" in p and cat in p]
        if csv_path:
            s, u, e, avg = count_csv_results(csv_path[0], 600)
        else:
            s, u, e, avg = 0, 0, 0, 0.0
    else:
        times = tl[t].get(cat, [])
        u = uc[t].get(cat, 0)
        e = ec[t].get(cat, 0)
        s = len(times)
        avg = sum(times) / s if s > 0 else 0.0
    return (s, u, e, avg)

def print_baseline_and_tisum_results_table():
    baseline_results = read_baseline_results(baseline_results_file)
    tl, uc, ec = summarize_baseline_results(baseline_results)
    tl2, uc2 = read_our_results(our_results_files)
    for tool, data in tl2.items():
        tl[tool] = data
        ec[tool] = {c: 0 for c in categories}
    for tool, data in uc2.items():
        uc[tool] = data
        ec[tool] = {c: 0 for c in categories}

    blocks = [(tools[0], tools[1], tools[2]),
              (tools[3], tools[4], tools[5]),
              (tools[6], tools[7], tools[8])]
    cat_lbl = {"acasxu_2023": "ACAS-Xu", "safenlp": "SafeNLP",
                "tllverifybench_2023": "TLLVerify", "linearizenn": "LinearizeNN"}

    sep = "=" * 92
    for t1, t2, t3 in blocks:
        print(sep)
        hdr = "{:<12s}  {:>26s}  {:>26s}  {:>26s}".format("Benchmark", t1, t2, t3)
        print(hdr)
        sub = "{:<12s}  {:>5s} {:>5s} {:>5s} {:>7s}  {:>5s} {:>5s} {:>5s} {:>7s}  {:>5s} {:>5s} {:>5s} {:>7s}".format(
            "", "Solved", "Timeou", "Error", "Time",
            "Solved", "Timeou", "Error", "Time",
            "Solved", "Timeou", "Error", "Time")
        print(sub)
        print(sep)
        for cat in categories:
            d1 = _row_data(t1, cat, tl, uc, ec, our_results_files)
            d2 = _row_data(t2, cat, tl, uc, ec, our_results_files)
            d3 = _row_data(t3, cat, tl, uc, ec, our_results_files)
            cells = [_tool_cell(*d) for d in [d1, d2, d3]]
            print("{:<12s}  {}  {}  {}".format(cat_lbl[cat], *cells))
        print()

def print_ablation_table():
    time_list, unknown_count = read_our_results(our_results_files)
    cat_lbl = {"acasxu_2023": "ACAS-Xu", "safenlp": "SafeNLP",
                "tllverifybench_2023": "TLLVerify", "linearizenn": "LinearizeNN"}
    mt = ["Ours (Linear)", "Ours (Optimizable)", "Ours (TiSum)"]

    sep = "=" * 92
    print(sep)
    hdr = "{:<12s}  {:>26s}  {:>26s}  {:>26s}".format("Benchmark", "Ours Linear", "Ours Optimizable", "TiSum")
    print(hdr)
    sub = "{:<12s}  {:>5s} {:>5s} {:>6s} {:>7s}  {:>5s} {:>5s} {:>6s} {:>7s}  {:>5s} {:>5s} {:>6s} {:>7s}".format(
        "", "Solved", "Timeou", "Branches", "Time",
        "Solved", "Timeou", "Branches", "Time",
        "Solved", "Timeou", "Branches", "Time")
    print(sub)
    print(sep)
    for cat in categories:
        row_data = []
        for m in mt:
            times = time_list[m].get(cat, [])
            unk = unknown_count[m].get(cat, 0)
            s = len(times)
            avg = sum(times) / s if s > 0 else 0.0
            sfx = {"Ours (Linear)": "linear", "Ours (Optimizable)": "ada", "Ours (TiSum)": "quad"}[m]
            csv_path = [p for p in our_results_files if sfx in p and cat in p]
            branch = avg_branches_per_solved(csv_path[0], cat, sfx) if csv_path else 0
            row_data.append((s, unk, branch, avg))
        cells = [_tool_cell(*d) for d in row_data]
        print("{:<12s}  {}  {}  {}".format(cat_lbl[cat], *cells))
    print(sep)


def plot_baselines_and_tisum_time_lists(times_list, save_path):
    for i, category in enumerate(categories):
        plt.figure(figsize=(6, 2))
        axe = plt.gca()

        for j, tool in enumerate(times_list.keys()):
            if category not in times_list[tool]:
                continue
            time_list = times_list[tool][category]

            accumu_time_list = [sum(time_list[: k + 1]) for k in range(len(time_list))]
            x = accumu_time_list
            y = range(1, len(accumu_time_list) + 1)
            if len(y) == 0:
                continue

            axe.plot(x, y, label=tool, linestyle=linestyles[j], color=colors[j])
            axe.axvline(x=accumu_time_list[-1], linestyle="--", color="gray", alpha=0.3)
            axe.axhline(y=len(accumu_time_list), linestyle="--", color="gray", alpha=0.3)
            axe.plot(
                accumu_time_list[-1], len(accumu_time_list),
                marker="o", markersize=5, color=colors[j],
            )
        axe.tick_params(axis="x", labelsize=16)
        axe.tick_params(axis="y", labelsize=16)
        axe.set_xlabel("Time ($s$)", fontsize=18)
        axe.set_ylabel("Solved Instances", fontsize=18)

        cat_name = category.replace(" ", "_").lower()
        plt.savefig(
            os.path.join(save_path, f"vnncomp2024_baselines_and_tisum_results_{cat_name}.pdf"),
            bbox_inches="tight", dpi=300, pad_inches=0.1,
        )


def plot_baselines_and_tisum_legend(save_path, tools_list):
    fig = plt.figure(figsize=(8, 2))
    plt.tight_layout(rect=(0.0, 0.1, 1, 0.9))

    handles = [
        plt.Line2D([], [], color=colors[i], linestyle=linestyles[i], label=tool)
        for i, tool in enumerate(tools_list)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=12, frameon=True)

    fig.savefig(
        os.path.join(save_path, "vnncomp2024_baselines_and_tisum_legend.pdf"),
        bbox_inches="tight", dpi=300, pad_inches=0.1,
    )


def plot_baselines_and_tisum(save_path):
    baseline_results = read_baseline_results(baseline_results_file)
    time_list, unknown_count, error_count = summarize_baseline_results(baseline_results)
    time_list2, unknown_count2 = read_our_results(our_results_files)
    for tool, categories_data in time_list2.items():
        time_list[tool] = categories_data
    for tool, categories_data in unknown_count2.items():
        unknown_count[tool] = categories_data

    for k in {"Ours (Linear)", "Ours (Optimizable)"}:
        time_list.pop(k, None)
        unknown_count.pop(k, None)
        error_count.pop(k, None)

    os.makedirs(save_path, exist_ok=True)
    plot_baselines_and_tisum_time_lists(time_list, save_path)
    plot_baselines_and_tisum_legend(save_path, tools)


def plot_ablation_study_time_lists(times_list, save_path):
    for i, category in enumerate(categories):
        plt.figure(figsize=(6, 2))
        axe = plt.gca()

        for j, tool in enumerate(our_tools):
            if category not in times_list[tool]:
                continue

            time_list = times_list[tool][category]
            accumu_time_list = [sum(time_list[: k + 1]) for k in range(len(time_list))]
            x = accumu_time_list
            y = range(1, len(accumu_time_list) + 1)
            if len(y) == 0:
                continue

            axe.plot(x, y, label=tool, linestyle=our_linestyles[j], color=our_colors[j])
            axe.axvline(x=accumu_time_list[-1], linestyle="--", color="gray", alpha=0.3)
            axe.axhline(y=len(accumu_time_list), linestyle="--", color="gray", alpha=0.3)
            axe.plot(
                accumu_time_list[-1], len(accumu_time_list),
                marker="o", markersize=5, color=our_colors[j],
            )
        axe.tick_params(axis="x", labelsize=16)
        axe.tick_params(axis="y", labelsize=16)
        axe.set_xlabel("Time ($s$)", fontsize=18)
        axe.set_ylabel("Solved Instances", fontsize=18)

        cat_name = category.replace(" ", "_").lower()
        plt.savefig(
            os.path.join(save_path, f"ablation_study_{cat_name}.pdf"),
            bbox_inches="tight", dpi=300, pad_inches=0.1,
        )


def plot_ablation_study_legend(save_path, tools_list):
    fig = plt.figure(figsize=(3, 1))
    plt.tight_layout(rect=(0.0, 0.1, 1, 0.9))

    handles = [
        plt.Line2D([], [], color=our_colors[i], linestyle=our_linestyles[i], label=tool)
        for i, tool in enumerate(our_tools)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=12, frameon=True)

    fig.savefig(
        os.path.join(save_path, "ablation_study_legend.pdf"),
        bbox_inches="tight", dpi=300, pad_inches=0.1,
    )


def plot_ablation(save_path):
    time_list, unknown_count = read_our_results(our_results_files)

    os.makedirs(save_path, exist_ok=True)
    plot_ablation_study_time_lists(time_list, save_path)
    plot_ablation_study_legend(save_path, tools)


if __name__ == "__main__":
    img_dir = os.path.join(_EXP, "imgs", "vnncomp24")
    tabs_dir = os.path.join(_EXP, "tabs")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(tabs_dir, exist_ok=True)

    # Console output
    print_baseline_and_tisum_results_table()
    print_ablation_table()

    # PDF figures
    plot_baselines_and_tisum(img_dir)
    plot_ablation(img_dir)

    # CSV tables
    generate_baseline_csv(our_results_files, os.path.join(tabs_dir, "vnncomp24_baselines_and_tisum_results.csv"))
    generate_ablation_csv(our_results_files, os.path.join(tabs_dir, "ablation_study.csv"))
    print(f"Tables written to {tabs_dir}/")
    print(f"Figures written to {img_dir}/")
