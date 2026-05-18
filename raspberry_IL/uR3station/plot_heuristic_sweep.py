"""
Summarise a 2-D heuristic parameter sweep and plot a success-rate heatmap.

A trial is counted as successful when its events.csv contains a 'detach_detected' event.

Usage:
    python raspberry_IL/uR3station/plot_heuristic_sweep.py \
        --sweep-root trial_logs_heuristic_sweep \
        --out heuristic_sweep.png
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 14})

TRIAL_RE = re.compile(r"trial_\d+")


def trial_succeeded(trial_dir):
    event_file = os.path.join(trial_dir, "events.csv")
    if not os.path.isfile(event_file):
        return False
    with open(event_file, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event") == "detach_detected":
                return True
    return False


def count_results(log_dir):
    successes, total = 0, 0
    if not os.path.isdir(log_dir):
        return None, None
    for name in os.listdir(log_dir):
        if TRIAL_RE.match(name) and os.path.isdir(os.path.join(log_dir, name)):
            total += 1
            if trial_succeeded(os.path.join(log_dir, name)):
                successes += 1
    return successes, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="trial_logs_heuristic_sweep",
                        help="Parent directory containing one subdirectory per parameter combination")
    parser.add_argument("--out", default="heuristic_sweep.png")
    parser.add_argument("--thresholds", default="8,10,12")
    parser.add_argument("--close-steps", default="0.00005,0.0001,0.0002")
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    close_steps = [float(x) for x in args.close_steps.split(",")]

    grid = np.full((len(thresholds), len(close_steps)), np.nan)
    counts = {}

    for i, thresh in enumerate(thresholds):
        for j, step in enumerate(close_steps):
            tag = f"thresh{int(thresh)}_close{str(step).replace('.', '')}"
            log_dir = os.path.join(args.sweep_root, tag)
            s, t = count_results(log_dir)
            if t is not None and t > 0:
                grid[i, j] = s / t
                counts[(i, j)] = (s, t)
            else:
                counts[(i, j)] = (None, None)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Success rate")

    ax.set_xticks(range(len(close_steps)))
    ax.set_xticklabels([str(s) for s in close_steps])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([str(t) for t in thresholds])
    ax.set_xlabel("slip_close_step")
    ax.set_ylabel("slip_threshold")
    ax.set_title("Heuristic agent sweep — success rate")

    for i in range(len(thresholds)):
        for j in range(len(close_steps)):
            s, t = counts[(i, j)]
            if t is not None:
                ax.text(j, i, f"{s}/{t}", ha="center", va="center", fontsize=13, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=13)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    print("\nResults:")
    for i, thresh in enumerate(thresholds):
        for j, step in enumerate(close_steps):
            s, t = counts[(i, j)]
            if t:
                print(f"  thresh={thresh}  close_step={step}  →  {s}/{t} = {s/t:.0%}")
            else:
                print(f"  thresh={thresh}  close_step={step}  →  no data")


if __name__ == "__main__":
    main()
