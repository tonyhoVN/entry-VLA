#!/usr/bin/env python3
"""t-SNE of the cached per-frame latent embeddings of each episode, coloured by the latent
phase split, with clusters named "phase i: frame a -> frame b".

Reads <out>/<episode>/latent/{embeddings_<cam>.npy, latent_split.json} written by
scripts/latent_phase_split.py. Writes <episode>/latent/tsne_<view>_<method>.png per episode
and <out>/latent_tsne_overview_<view>_<method>.png (all episodes, small multiples).

  python scripts/plot_latent_tsne.py --out data/entry_process/push_button_seed42/variation4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.perception.latent_split import standardise  # noqa: E402

# Categorical slots 1..7 of the dataviz reference palette (light surface), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # secondary encoding: identity never colour-only
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"


def tsne_2d(X, seed):
    from sklearn.manifold import TSNE
    perp = float(max(5, min(30, (len(X) - 1) // 3)))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(X)


def phase_bounds(split, view, method):
    sp = split["splits"][view]
    b = sp["k3"] if method == "k3" else [sp["k2" if method == "k2" else "novelty_peak"]]
    return [x for x in b if x is not None]


def draw(ax, Y, bounds, title, small=False):
    n = len(Y)
    edges = [0] + list(bounds) + [n]
    ax.set_facecolor(SURFACE)
    ax.plot(Y[:, 0], Y[:, 1], color=GRID, lw=1.0 if small else 1.5, zorder=1)  # time order
    for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        if e <= s:
            continue
        c, m = SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]
        name = f"phase {i + 1}: frame {s} → frame {e - 1}"
        ax.scatter(Y[s:e, 0], Y[s:e, 1], s=14 if small else 36, c=c, marker=m, edgecolors=SURFACE,
                   linewidths=0.8, zorder=2, label=name)
        cx, cy = Y[s:e].mean(0)
        ax.annotate(f"phase {i + 1}" if small else name, (cx, cy), xytext=(0, 10 if small else 14),
                    textcoords="offset points", ha="center", fontsize=6.5 if small else 9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85), zorder=4)
    ax.scatter(*Y[0], s=60 if not small else 24, facecolors="none", edgecolors=INK, lw=1.2, zorder=3)
    ax.annotate("start", Y[0], xytext=(-4, -12), textcoords="offset points", fontsize=6.5 if small else 8,
                color=INK2, ha="right")
    ax.set_title(title, fontsize=8 if small else 11, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="processed root with <episode>/latent/")
    ap.add_argument("--view", default="wrist", choices=["exo", "wrist", "all"])
    ap.add_argument("--method", default="k3", choices=["k2", "k3", "novelty"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent-dir", default="latent")
    ap.add_argument("--episodes", nargs="+", default=None)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = sorted((d for d in os.listdir(a.out) if d.startswith("episode")
                  and os.path.exists(os.path.join(a.out, d, a.latent_dir, "latent_split.json"))
                  and (not a.episodes or d in a.episodes)), key=lambda d: int(d[7:]))
    cols = 5
    rows = int(np.ceil(len(eps) / cols))
    fig_all, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for k, d in enumerate(eps):
        lat = os.path.join(a.out, d, a.latent_dir)
        split = json.load(open(os.path.join(lat, "latent_split.json")))
        cams = split["view_sets"][a.view]
        X = standardise(np.concatenate([np.load(os.path.join(lat, f"embeddings_{c}.npy")) for c in cams], axis=1))
        Y = tsne_2d(X, a.seed)
        np.save(os.path.join(lat, f"tsne_{a.view}.npy"), Y)
        bounds = phase_bounds(split, a.view, a.method)
        refs = split.get("references", {})
        sub = f"stop {refs.get('stop_frame')}, contact {refs.get('contact_frame')}" if refs else ""

        fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
        draw(ax, Y, bounds, f"{d}: t-SNE of DINOv2 {a.view} embeddings ({a.method} split)"
             + (f"\nstate reference: {sub}" if sub else ""))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=9,
                  labelcolor=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(lat, f"tsne_{a.view}_{a.method}.png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)
        draw(axes[k], Y, bounds, f"{d}  (stop {refs.get('stop_frame')})", small=True)
        print(d, "bounds", bounds, flush=True)
    for ax in axes[len(eps):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker=MARKERS[i], color=SERIES[i], ls="", markersize=7,
                          label=f"phase {i + 1}") for i in range(3 if a.method == "k3" else 2)]
    fig_all.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10, labelcolor=INK)
    fig_all.suptitle(f"t-SNE of per-frame DINOv2 {a.view} embeddings, coloured by latent {a.method} phases "
                     f"(grey line = time order, ring = first frame)", color=INK, fontsize=12, x=0.01, ha="left")
    fig_all.tight_layout(rect=(0, 0.04, 1, 0.97))
    tag = "" if a.latent_dir == "latent" else f"_{a.latent_dir}"
    path = os.path.join(a.out, f"latent_tsne_overview{tag}_{a.view}_{a.method}.png")
    fig_all.savefig(path, dpi=130, facecolor=SURFACE)
    print("overview:", path)


if __name__ == "__main__":
    main()
