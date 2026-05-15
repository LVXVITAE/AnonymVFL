"""
Plotting utilities for performance benchmarks and model evaluation.

All plots are saved as PNG files (150 DPI) to the test_results/ directory.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (10, 6),
})
# 设置中文显示（解决中文方框乱码问题）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def save_performance_table(records: list[dict], csv_path: str):
    """Save a list of dicts as a CSV file."""
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def plot_time_vs_samples(records: list[dict], title: str, png_path: str,
                         x_key: str = "样本数量", y_key: str = "总时间(s)",
                         comm_key: str = "总通信量(MB)"):
    """Line chart: time vs sample size, with optional communication volume on secondary y-axis."""
    df = pd.DataFrame(records)
    color_time = "#4C72B0"
    color_comm = "#DD8452"

    fig, ax1 = plt.subplots()
    ax1.plot(df[x_key], df[y_key], marker="o", color=color_time, linewidth=2, label=y_key)
    ax1.set_xlabel(x_key)
    ax1.set_ylabel(y_key, color=color_time)
    ax1.tick_params(axis="y", labelcolor=color_time)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    if comm_key in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df[x_key], df[comm_key], marker="s", color=color_comm,
                 linewidth=2, linestyle="--", label=comm_key)
        ax2.set_ylabel(comm_key, color=color_comm)
        ax2.tick_params(axis="y", labelcolor=color_comm)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_batch_size_impact(records: list[dict], png_path: str,
                            comm_key: str = "总通信量(MB)"):
    """Line chart: batch size vs training time, one line per sample count.
    If comm_key column is present, communication volume is plotted on a secondary y-axis."""
    df = pd.DataFrame(records)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    comm_colors = ["#DD8452", "#E39566", "#EAA67A", "#F0B78E", "#F6C8A2"]
    has_comm = comm_key in df.columns

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() if has_comm else None

    groups = df.groupby("样本数量")
    for idx, (n_samples, grp) in enumerate(sorted(groups)):
        grp = grp.sort_values("批次大小")
        color = colors[idx % len(colors)]
        ax1.plot(grp["批次大小"], grp["总训练时间(s)"],
                 marker="o", color=color, linewidth=2,
                 label=f"训练时间 样本数={int(n_samples)}")
        for x, y in zip(grp["批次大小"], grp["总训练时间(s)"]):
            ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)
        if has_comm:
            ax2.plot(grp["批次大小"], grp[comm_key],
                     marker="s", color=comm_colors[idx % len(comm_colors)],
                     linewidth=2, linestyle="--",
                     label=f"{comm_key} 样本数={int(n_samples)}")

    ax1.set_xlabel("批次大小")
    ax1.set_ylabel("总训练时间 (s)", color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax1.set_title("批次大小对SSLR训练性能的影响")
    ax1.grid(True, alpha=0.3)

    if has_comm:
        ax2.set_ylabel(comm_key, color="#DD8452")
        ax2.tick_params(axis="y", labelcolor="#DD8452")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    else:
        ax1.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_inference_latency(records: list[dict], png_path: str,
                            comm_key: str = "总通信量(MB)"):
    """Two-subplot chart: (1) total latency + communication volume, (2) per-sample latency."""
    df = pd.DataFrame(records)
    color1, color2, color3 = "#4C72B0", "#55A868", "#DD8452"
    has_comm = comm_key in df.columns

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Top subplot: 总延迟 + 总通信量 ---
    ax_top.set_ylabel("推理延迟 (s)", color=color1)
    l1, = ax_top.plot(df["样本数量"], df["推理延迟(s)"], marker="o", color=color1,
                      linewidth=2, label="总延迟")
    ax_top.tick_params(axis="y", labelcolor=color1)
    ax_top.set_title("推理延迟与通信量")
    ax_top.grid(True, alpha=0.3)

    top_lines, top_labels = [l1], ["总延迟"]
    if has_comm:
        ax_top2 = ax_top.twinx()
        ax_top2.set_ylabel(comm_key, color=color3)
        l3, = ax_top2.plot(df["样本数量"], df[comm_key], marker="^", color=color3,
                           linewidth=2, linestyle="--", label=comm_key)
        ax_top2.tick_params(axis="y", labelcolor=color3)
        top_lines.append(l3)
        top_labels.append(comm_key)
    ax_top.legend(top_lines, top_labels, loc="upper left", fontsize=9)

    # --- Bottom subplot: 单样本延迟 ---
    ax_bot.set_xlabel("样本数量")
    ax_bot.set_ylabel("单样本平均延迟 (ms)", color=color2)
    ax_bot.plot(df["样本数量"], df["单样本平均延迟(ms)"], marker="s", color=color2,
                linewidth=2, label="单样本延迟")
    ax_bot.tick_params(axis="y", labelcolor=color2)
    ax_bot.set_title("单样本平均推理延迟")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wan_network_impact(records: list[dict], title: str, png_path: str,
                            y_key: str):
    """Line chart for fixed-workload WAN tests: bandwidth vs metric, grouped by latency."""
    df = pd.DataFrame(records).sort_values(["延迟(ms)", "带宽限制(Mb/s)"])
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots()
    for idx, (latency, grp) in enumerate(df.groupby("延迟(ms)", sort=True)):
        ax.plot(
            grp["带宽限制(Mb/s)"],
            grp[y_key],
            marker="o",
            linewidth=2,
            color=colors[idx % len(colors)],
            label=f"{int(latency)} ms",
        )
    ax.set_xlabel("带宽限制 (Mb/s)")
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="延迟")
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: list[dict], dataset_name: str, png_path: str):
    """Grouped bar chart: compare methods on Accuracy/Precision/Recall/F1/AUC."""
    df = pd.DataFrame(results)
    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    available = [m for m in metrics if m in df.columns]
    methods = df["方法"].tolist()

    x = range(len(available))
    width = 0.8 / len(methods)
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(16, 9))
    for i, method in enumerate(methods):
        row = df[df["方法"] == method].iloc[0]
        vals = [float(row[m]) for m in available]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width,
                      label=method, color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(available)
    ax.set_ylabel("指标值")
    ax.set_title(f"模型效果对比 — {dataset_name}")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(steps: list, train_metric: list, test_metric: list | None,
                        ylabel: str, title: str, png_path: str):
    """Plot training (and optionally test) metric curve over epochs/trees."""
    fig, ax = plt.subplots()
    ax.plot(steps, train_metric, marker="o", linewidth=2, label="训练集")
    if test_metric:
        ax.plot(steps, test_metric, marker="s", linewidth=2,
                linestyle="--", label="测试集")
    ax.set_xlabel("轮次")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
