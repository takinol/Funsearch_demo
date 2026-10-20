"""根据 results 下的 metrics.jsonl / samples.jsonl 绘制实验图。

用法:
    from demo_1 import visual
    paths = visual.plot_all()          # 默认读取 demo/results
    paths = visual.plot_all(show=True) # 可选：弹窗预览
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


# 默认结果目录: demo/results （visual.py 位于 demo/demo_1/ 下）
def _default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def _load_jsonl(path: Path) -> list[dict]:
    """读取 jsonl，返回 dict 列表。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件: {path}\n请先运行 demo_1.run 生成日志。")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _setup_style() -> None:
    """让中文标题尽量正常显示，同时兼容无中文字体的服务器。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_evolution(
    metrics: Sequence[dict],
    path: str | Path | None = None,
) -> Path:
    """进化曲线：平均使用箱子数越低越好。

    metrics 中分数是负的箱子数，因此绘图时取 -score 转成正数。
    """
    _setup_style()
    iterations = [m["iteration"] for m in metrics]
    bins_used = [-m["global_best_score"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, bins_used, color="#1f77b4", lw=2, marker=".",
            ms=5, label="best bins used")

    best_idx = min(range(len(bins_used)), key=lambda i: bins_used[i])
    ax.scatter(iterations[best_idx], bins_used[best_idx],
               color="red", zorder=5, label="best")
    ax.annotate(
        f"{bins_used[best_idx]:.1f} bins @ iter {iterations[best_idx]}",
        xy=(iterations[best_idx], bins_used[best_idx]),
        xytext=(8, 8), textcoords="offset points",
        color="red", fontsize=9)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average bins used (lower is better)")
    ax.set_title("FunSearch: best program evolution")
    ax.grid(alpha=0.3)
    ax.legend()

    path = Path(path) if path else _default_results_dir() / "plots" / "evolution.png"
    return _save_fig(fig, path)


def plot_population(
    metrics: Sequence[dict],
    path: str | Path | None = None,
) -> Path:
    """种群规模与簇数量随迭代的变化。"""
    _setup_style()
    iterations = [m["iteration"] for m in metrics]
    programs = [m["num_programs"] for m in metrics]
    clusters = [m["num_clusters"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, programs, color="#2ca02c", lw=2, label="programs")
    ax.plot(iterations, clusters, color="#ff7f0e", lw=2, label="clusters")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count")
    ax.set_title("Population / cluster growth")
    ax.grid(alpha=0.3)
    ax.legend()

    path = Path(path) if path else _default_results_dir() / "plots" / "population.png"
    return _save_fig(fig, path)


def plot_acceptance_and_scores(
    samples: Sequence[dict],
    path: str | Path | None = None,
) -> Path:
    """左图：每轮 accepted / rejected；右图：入库程序的分数分布。"""
    _setup_style()
    accepted = [r for r in samples if r.get("accepted")]

    # 每轮通过/拒绝数
    iterations = sorted({r["iteration"] for r in samples})
    acc_count = Counter(r["iteration"] for r in samples if r.get("accepted"))
    rej_count = Counter(r["iteration"] for r in samples if not r.get("accepted"))
    acc = [acc_count.get(i, 0) for i in iterations]
    rej = [rej_count.get(i, 0) for i in iterations]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(iterations, acc, color="#2ca02c", label="accepted")
    ax1.bar(iterations, rej, bottom=acc, color="#d62728", label="rejected")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Samples")
    ax1.set_title("Acceptance per iteration")
    ax1.legend()

    # 分数转正数后画直方图
    bins_used = [-r["score"] for r in accepted if r.get("score") is not None]
    if bins_used:
        ax2.hist(bins_used, bins=40, color="#1f77b4", alpha=0.8)
        ax2.axvline(min(bins_used), color="red", ls="--",
                    label=f"best {min(bins_used):.1f}")
        ax2.set_xlabel("Average bins used (lower is better)")
        ax2.set_ylabel("Accepted programs")
        ax2.set_title("Score distribution of accepted programs")
        ax2.legend()

    path = path if path else _default_results_dir() / "plots" / "acceptance.png"
    return _save_fig(fig, Path(path))


def plot_diversity(
    samples: Sequence[dict],
    path: str | Path | None = None,
) -> Path:
    """累计唯一代码数 vs 累计唯一行为签名数。

    code_hash 不同 = 代码实现不同；
    signature 不同 = 在数据集上的得分向量不同（行为不同）。
    """
    _setup_style()
    code_hashes: set[str] = set()
    signatures: set[tuple[float, ...]] = set()
    xs: list[int] = []
    code_curve: list[int] = []
    sig_curve: list[int] = []

    for order, r in enumerate(samples):
        if not r.get("accepted"):
            continue
        if r.get("code_hash"):
            code_hashes.add(r["code_hash"])
        sig = r.get("signature")
        if sig:
            signatures.add(tuple(sig))
        xs.append(order)
        code_curve.append(len(code_hashes))
        sig_curve.append(len(signatures))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, code_curve, color="#1f77b4", lw=2, label="unique code")
    ax.plot(xs, sig_curve, color="#ff7f0e", lw=2, label="unique behavior (signature)")
    ax.set_xlabel("Accepted sample order")
    ax.set_ylabel("Cumulative unique count")
    ax.set_title("Code diversity vs behavioral diversity")
    ax.grid(alpha=0.3)
    ax.legend()

    path = Path(path) if path else _default_results_dir() / "plots" / "diversity.png"
    return _save_fig(fig, path)


def plot_all(
    results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> list[Path]:
    """一键生成全部图表。

    Args:
        results_dir: 存放 metrics.jsonl / samples.jsonl 的目录，默认 demo/results。
        output_dir: 图片输出目录，默认 demo/results/plots。
        show: 是否调用 plt.show() 弹窗预览。

    Returns:
        生成图片的路径列表。
    """
    results = Path(results_dir) if results_dir else _default_results_dir()
    out = Path(output_dir) if output_dir else results / "plots"
    out.mkdir(parents=True, exist_ok=True)

    metrics = _load_jsonl(results / "metrics.jsonl")
    samples = _load_jsonl(results / "samples.jsonl")

    paths = [
        plot_evolution(metrics, out / "evolution.png"),
        plot_population(metrics, out / "population.png"),
        plot_acceptance_and_scores(samples, out / "acceptance.png"),
        plot_diversity(samples, out / "diversity.png"),
    ]

    if show:
        plt.show()
    return paths


if __name__ == "__main__":
    for p in plot_all():
        print(p)
