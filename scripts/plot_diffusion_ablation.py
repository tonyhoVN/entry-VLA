#!/usr/bin/env python3
"""Line plot of validation error vs training step for entry-diffusion runs (train_log.jsonl).

Two small multiples (position, rotation: different units, never a dual axis), one line per run, the
task-mean baseline as a grey dashed reference, a ring on each run's best (selected) checkpoint.

  python scripts/plot_diffusion_ablation.py --out runs/entry_diffusion_ablation.png
"""

import argparse
import json
import os

RUNS = [("full (points + crops + text)", "runs/entry_diffusion_v1"),
        ("no crops (vision off)", "runs/entry_diffusion_v1_novision"),
        ("no text", "runs/entry_diffusion_v1_notext")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated, light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/entry_diffusion_ablation.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = {}
    for name, d in RUNS:
        rows = [json.loads(line) for line in open(os.path.join(d, "train_log.jsonl"))]
        best = json.load(open(os.path.join(d, "report.json")))["best_step"]
        logs[name] = (rows, best)
    base = next(iter(logs.values()))[0][0]
    panels = [("val_medoid_pos_m", 100, "cm", "Position error", "val_task_mean_pos_m"),
              ("val_medoid_rot_rad", 1, "rad", "Rotation error", "val_task_mean_rot_rad")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, (key, scale, unit, title, bkey) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ends = []
        for (name, (rows, best)), color in zip(logs.items(), COLORS):
            x = [r["step"] for r in rows]
            y = [r[key] * scale for r in rows]
            ax.plot(x, y, color=color, lw=2, label=name, solid_capstyle="round")
            yb = next(r[key] * scale for r in rows if r["step"] == best)
            ax.plot([best], [yb], "o", ms=8, mfc=color, mec=SURFACE, mew=2, zorder=3)
            ends.append([x[-1], y[-1], name, color])
        b = base[bkey] * scale
        ax.axhline(b, color=INK2, lw=1.2, ls=(0, (4, 3)))
        ax.text(0.01, b, f" task-mean baseline {b:.1f} {unit}" if unit == "cm" else f" task-mean baseline {b:.2f} {unit}",
                transform=ax.get_yaxis_transform(), color=INK2, fontsize=8, va="bottom", ha="left")
        # direct labels at line ends, nudged apart so they never overlap
        ends.sort(key=lambda e: e[1])
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.06
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
        for xe, ye, name, color in ends:
            ax.text(xe * 1.01, ye, name.split(" (")[0], color=INK, fontsize=8, va="center", ha="left")
        ax.set_xlim(0, max(e[0] for e in ends) * 1.18)
        ax.set_title(f"{title} ({unit}), validation medoid of 16 samples", fontsize=10, loc="left", color=INK)
        ax.set_xlabel("training step", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=3,
               labelcolor=INK)
    fig.suptitle("Entry-pose diffusion ablation (RLBench, 63 train / 14 val episodes) - ring = selected checkpoint",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(a.out)


if __name__ == "__main__":
    main()
