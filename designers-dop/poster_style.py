"""Единая палитра и графики исследования постеров (Matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

INK = "#19334B"
MUTED = "#60758A"
BLUE = "#436C9A"
TEAL = "#16A398"
CORAL = "#DE715D"
GRID = "#E3EAF0"
PAPER = "#FFFFFF"
HEAT = LinearSegmentedColormap.from_list("poster", ["#F2F7FA", "#94C7CF", TEAL, INK])


def _axes(ax):
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, pad=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.85)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_marginals(pmf: pd.DataFrame):
    """Сравнение P(X=x) и P(Y=y) на общей корректной шкале 0...1."""
    fig, ax = plt.subplots(figsize=(9, 3.7), dpi=120, layout="constrained")
    levels = pmf.index.to_numpy(dtype=int)
    width = 0.35
    for offset, column, label, color in [
        (-width / 2, "P(X=x)", "X: все записи", BLUE),
        (width / 2, "P(Y=y)", "Y: уникальные", TEAL),
    ]:
        ax.bar(levels + offset, pmf[column], width=width, color=color,
               edgecolor="white", linewidth=0.5, label=label, zorder=3)
    _axes(ax)
    ax.set_xticks(levels)
    ax.set_xlim(levels.min() - 0.65, levels.max() + 0.65)
    ax.set_ylim(0, max(pmf.to_numpy().max() * 1.14, 0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Число записей / уникальных цветов, шт.")
    ax.set_ylabel("Доля постеров, %")
    ax.set_title("Распределения числа цветов", loc="left", fontsize=13, weight="bold", pad=17)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK, ncol=2)
    return fig


def plot_joint(joint: pd.DataFrame):
    """Совместное распределение и произведение маргинальных, обе шкалы в %."""
    expected = pd.DataFrame(
        np.outer(joint.sum(axis=1), joint.sum(axis=0)),
        index=joint.index, columns=joint.columns,
    )
    maximum = max(joint.to_numpy().max(), expected.to_numpy().max()) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), dpi=120, layout="constrained")
    for ax, (title, values) in zip(axes, [
        ("Наблюдаемое P(X=x, Y=y)", joint),
        ("При независимости P(X=x)·P(Y=y)", expected),
    ]):
        data = values.to_numpy() * 100
        image = ax.imshow(data, cmap=HEAT, vmin=0, vmax=maximum, aspect="equal")
        ax.set_xticks(range(len(joint.columns)), joint.columns)
        ax.set_yticks(range(len(joint.index)), joint.index)
        ax.set_xlabel("Y: уникальных цветов, шт.")
        ax.set_ylabel("X: записей цветов, шт.")
        ax.set_title(title, fontsize=11, color=INK, weight="bold", pad=14)
        ax.tick_params(colors=MUTED, length=0)
        for (i, j), value in np.ndenumerate(data):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                    color=PAPER if value > maximum * 0.5 else INK)
    bar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    bar.set_label("Доля постеров, %", color=INK)
    bar.ax.tick_params(colors=MUTED)
    return fig


def plot_simulation(samples: np.ndarray, x_levels, y_levels):
    """Пузырьковая диаграмма результатов интерактивных испытаний."""
    observed = pd.DataFrame(samples, columns=["X", "Y"]).value_counts().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(7.9, 4.5), dpi=120, layout="constrained")
    _axes(ax)
    lo = min(min(x_levels), min(y_levels))
    hi = max(max(x_levels), max(y_levels))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", lw=1.4, alpha=0.75)
    for invalid, color in [(False, TEAL), (True, CORAL)]:
        part = observed[(observed["Y"] > observed["X"]) == invalid]
        ax.scatter(part["X"], part["Y"],
                   s=55 + 900 * part["n"] / len(samples), color=color,
                   edgecolors="white", linewidths=1.5, alpha=0.9, zorder=3)
        for item in part.itertuples(index=False):
            ax.annotate(str(item.n), (item.X, item.Y), ha="center", va="center",
                        fontsize=8, color=INK, xytext=(0, -14),
                        textcoords="offset points")
    ax.set_xlim(min(x_levels) - 0.55, max(x_levels) + 0.55)
    ax.set_ylim(min(y_levels) - 0.55, max(y_levels) + 0.55)
    ax.set_xticks(sorted(set(x_levels)))
    ax.set_yticks(sorted(set(y_levels)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X: записей цветов, шт.")
    ax.set_ylabel("Y: уникальных цветов, шт.")
    ax.set_title("Смоделированные пары (X, Y)", loc="left", fontsize=13, weight="bold", pad=14)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL,
               markeredgecolor="white", markersize=10, label="Y ≤ X"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL,
               markeredgecolor="white", markersize=10, label="Y > X (невозможно для постера)"),
    ], frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), labelcolor=INK)
    return fig
