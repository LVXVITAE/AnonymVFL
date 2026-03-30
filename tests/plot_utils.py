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
                         x_key: str = "样本数量", y_key: str = "总时间(s)"):
    """Line chart: time vs sample size."""
    df = pd.DataFrame(records)
    fig, ax = plt.subplots()
    ax.plot(df[x_key], df[y_key], marker="o", linewidth=2)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_batch_size_impact(records: list[dict], png_path: str):
    """Line chart: batch size vs training time, one line per sample count."""
    df = pd.DataFrame(records)
    fig, ax = plt.subplots()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    groups = df.groupby("样本数量")
    for idx, (n_samples, grp) in enumerate(sorted(groups)):
        grp = grp.sort_values("批次大小")
        color = colors[idx % len(colors)]
        ax.plot(grp["批次大小"], grp["总训练时间(s)"],
                marker="o", color=color, linewidth=2,
                label=f"样本数={int(n_samples)}")
        for x, y in zip(grp["批次大小"], grp["总训练时间(s)"]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("批次大小")
    ax.set_ylabel("总训练时间 (s)")
    ax.set_title("批次大小对SSLR训练性能的影响")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_inference_latency(records: list[dict], png_path: str):
    """Line chart: inference latency vs sample count."""
    df = pd.DataFrame(records)
    fig, ax1 = plt.subplots()
    color1, color2 = "#4C72B0", "#DD8452"

    ax1.set_xlabel("样本数量")
    ax1.set_ylabel("推理延迟 (s)", color=color1)
    ax1.plot(df["样本数量"], df["推理延迟(s)"], marker="o", color=color1,
             linewidth=2, label="总延迟")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("单样本平均延迟 (ms)", color=color2)
    ax2.plot(df["样本数量"], df["单样本平均延迟(ms)"], marker="s", color=color2,
             linewidth=2, linestyle="--", label="单样本延迟")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("推理延迟测试结果")
    ax1.grid(True, alpha=0.3)
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
