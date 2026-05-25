"""
Compute the pressure-to-force OLS slope (Pa/g) per trial.

Instead of peak-to-peak, fits an OLS line through all (force, pressure) points
in the slope window (from when pressure first exceeds --pressure-start up to
the force peak), with an intercept.

Usage:
    python analyze_fp_ols.py --log-dir trial_logs_heuristic_sweep/thresh8_close0004
    python analyze_fp_ols.py --log-dir trial_logs_heuristic_sweep/thresh8_close0004 \
        --pressure-start 3000 --save-plot analysis_outputs/heuristic_thresh8_close0004/fpols_ \
        --save-summary-plot analysis_outputs/heuristic_thresh8_close0004/fpols_summary.png
"""

import argparse
import csv
import os
import re

import numpy as np
import matplotlib.pyplot as plt

NUM_SENSORS = 8
RASPBERRY_WINDOW = 10
RASPBERRY_BASE_SAMPLES = 100
ZERO_DEADBAND = 8.0
FORCE_START_OFFSET = 20.0


def discover_trials(log_dir):
    trials = {}
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Directory not found: {log_dir}")
    for name in sorted(os.listdir(log_dir)):
        full = os.path.join(log_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.match(r"trial_(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        entry = {}
        for key, fname in [("rasp", "raspberry.csv"), ("load", "loadcell.csv")]:
            p = os.path.join(full, fname)
            if os.path.isfile(p):
                entry[key] = p
        if "rasp" in entry and "load" in entry:
            trials[idx] = entry
    return trials


def read_raspberry_csv(path):
    t, sensors = [], [[] for _ in range(NUM_SENSORS)]
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_pc"]))
            for i in range(NUM_SENSORS):
                sensors[i].append(float(row[f"S{i}"]))
    return np.asarray(t), sensors


def read_loadcell_csv(path):
    t, force = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_pc"]))
            force.append(float(row["force"]))
    return np.asarray(t), np.asarray(force)


def process_raspberry(sensors):
    out = np.zeros((NUM_SENSORS, len(sensors[0])), dtype=float)
    for i, ch in enumerate(sensors):
        ch = np.asarray(ch)
        baseline = ch[:RASPBERRY_BASE_SAMPLES].mean() if len(ch) >= RASPBERRY_BASE_SAMPLES else ch.mean()
        buf = [baseline] * RASPBERRY_WINDOW
        wi = 0
        smoothed = []
        for x in ch:
            buf[wi] = x
            wi = (wi + 1) % RASPBERRY_WINDOW
            smoothed.append(sum(buf) / RASPBERRY_WINDOW)
        delta = np.asarray(smoothed) - baseline
        delta[np.abs(delta) < ZERO_DEADBAND] = 0.0
        out[i] = delta
    return out


def aggregate_pressure(processed, method="max"):
    if method == "max":
        return np.max(processed, axis=0)
    agg = np.zeros(processed.shape[1])
    for j in range(processed.shape[1]):
        active = processed[:, j][processed[:, j] > 0]
        agg[j] = active.mean() if active.size > 0 else 0.0
    return agg


def extract_trial(trial_idx, files, aggregation, pressure_start):
    rasp_t, raw_sensors = read_raspberry_csv(files["rasp"])
    load_t, load_force = read_loadcell_csv(files["load"])

    if len(load_t) < 2:
        return None

    force_interp = np.interp(rasp_t, load_t, load_force)
    processed = process_raspberry(raw_sensors)
    pressure = aggregate_pressure(processed, method=aggregation)

    idx = np.where(pressure >= pressure_start)[0]
    if idx.size == 0:
        print(f"  trial {trial_idx:03d}: pressure never reached {pressure_start} Pa, skipping")
        return None
    slope_start_i = idx[0]

    force_peak_i = slope_start_i + int(np.argmax(force_interp[slope_start_i:]))
    if force_peak_i <= slope_start_i:
        print(f"  trial {trial_idx:03d}: no force rise after pressure start, skipping")
        return None

    f_window = force_interp[slope_start_i:force_peak_i + 1]
    p_window = pressure[slope_start_i:force_peak_i + 1]
    f_pre = force_interp[:slope_start_i]
    p_pre = pressure[:slope_start_i]

    if len(f_window) < 2:
        print(f"  trial {trial_idx:03d}: slope window too short, skipping")
        return None

    # OLS with intercept: p = slope * f + intercept
    A = np.column_stack([f_window, np.ones(len(f_window))])
    result = np.linalg.lstsq(A, p_window, rcond=None)
    slope, intercept = float(result[0][0]), float(result[0][1])

    print(f"  trial {trial_idx:03d}: window={len(f_window)} pts  "
          f"force {f_window[0]:.1f}→{f_window[-1]:.1f}g  "
          f"slope={slope:.2f} Pa/g  intercept={intercept:.0f} Pa")

    return {
        "idx": trial_idx,
        "slope": slope,
        "intercept": intercept,
        "f_window": f_window,
        "p_window": p_window,
        "f_pre": f_pre,
        "p_pre": p_pre,
        "force_start": float(f_window[0]),
        "force_peak": float(f_window[-1]),
        "pressure_start": float(p_window[0]),
        "pressure_peak": float(p_window[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="trial_logs_heuristic_sweep/thresh8_close0004")
    parser.add_argument("--aggregation", choices=["max", "mean_active"], default="max")
    parser.add_argument("--pressure-start", type=float, default=3000.0)
    parser.add_argument("--ref-slope", type=float, default=None,
                        help="Reference slope to draw (default: mean of trial OLS slopes)")
    parser.add_argument("--save-plot", default=None, metavar="PREFIX")
    parser.add_argument("--save-summary-plot", default=None, metavar="PATH")
    args = parser.parse_args()

    trials = discover_trials(args.log_dir)
    if not trials:
        print(f"No trials found in {args.log_dir}")
        return

    print(f"Found {len(trials)} trials in '{args.log_dir}'\n")

    results = []
    for idx in sorted(trials):
        r = extract_trial(idx, trials[idx], args.aggregation, args.pressure_start)
        if r is not None:
            results.append(r)

    if not results:
        print("No usable trials.")
        return

    slopes = np.array([r["slope"] for r in results])
    mean_slope = float(slopes.mean())
    std_slope = float(slopes.std())
    ref_slope = args.ref_slope if args.ref_slope is not None else mean_slope

    print(f"\n── OLS slopes (pressure ≥ {args.pressure_start:.0f} Pa → force peak) ──")
    print(f"  {'trial':>5}  {'n_pts':>5}  {'slope(Pa/g)':>11}  {'intercept(Pa)':>13}")
    for r in results:
        print(f"  {r['idx']:>5}  {len(r['f_window']):>5}  {r['slope']:>11.3f}  {r['intercept']:>13.0f}")
    print(f"\n  mean slope = {mean_slope:.4f} Pa/g")
    print(f"  std  slope = {std_slope:.4f} Pa/g  ({100*std_slope/mean_slope:.1f}% of mean)")
    print(f"  ref  slope = {ref_slope:.4f} Pa/g")

    # Summary scatter
    if args.save_summary_plot or not args.save_plot:
        fig_s, ax_s = plt.subplots(figsize=(7, 5))
        for r in results:
            ax_s.scatter(r["f_window"], r["p_window"], s=10, alpha=0.4, color="steelblue")
        all_f = np.concatenate([r["f_window"] for r in results])
        x_line = np.linspace(0, all_f.max() * 1.1, 200)
        ax_s.plot(x_line, ref_slope * x_line + 0, "k-", linewidth=2,
                  label=f"mean OLS slope = {mean_slope:.2f} Pa/g")
        ax_s.set_xlim(left=0)
        ax_s.set_ylim(bottom=0)
        ax_s.set_xlabel("Load cell force [g]")
        ax_s.set_ylabel(f"Pressure ({args.aggregation}) [Pa]")
        ax_s.set_title("OLS slope window: all trials")
        ax_s.legend(fontsize=9)
        ax_s.grid(True)
        plt.tight_layout()
        if args.save_summary_plot:
            fig_s.savefig(args.save_summary_plot, dpi=150, bbox_inches="tight")
            print(f"Saved {args.save_summary_plot}")
            plt.close(fig_s)
        else:
            plt.show()

    if not args.save_plot:
        return

    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                          "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})

    for r in results:
        fig, ax = plt.subplots(figsize=(7, 5))

        if r["f_pre"].size > 0:
            ax.scatter(r["f_pre"], r["p_pre"], s=14, alpha=0.4, color="grey", zorder=2, label="pre-slope")

        ax.scatter(r["f_window"], r["p_window"], s=18, alpha=0.6, color="steelblue",
                   zorder=3, label="slope window")

        ax.scatter([r["force_start"]], [r["pressure_start"]], s=60, color="green", zorder=5,
                   label=f"start ({r['force_start']:.1f}g, {r['pressure_start']:.0f}Pa)")
        ax.scatter([r["force_peak"]], [r["pressure_peak"]], s=60, color="crimson", zorder=5,
                   label=f"peak ({r['force_peak']:.1f}g, {r['pressure_peak']:.0f}Pa)")

        x_line = np.linspace(r["force_start"], r["force_peak"] * 1.1, 200)
        ax.plot(x_line, r["slope"] * x_line + r["intercept"], color="steelblue",
                linewidth=1.4, linestyle="--", label=f"OLS slope = {r['slope']:.2f} Pa/g")
        ax.plot(x_line, ref_slope * (x_line - r["force_start"]) + r["pressure_start"],
                color="darkorange", linewidth=2, linestyle="-.",
                label=f"ref slope = {ref_slope:.1f} Pa/g")

        ax.set_xlabel("Load cell force [g]")
        ax.set_ylabel(f"Pressure ({args.aggregation}) [Pa]")
        ax.set_title(f"Trial {r['idx']:03d} — OLS force vs pressure")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        path = f"{args.save_plot}{r['idx']:03d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
