"""
Compute the pressure-to-force slope (Pa/g) from human demonstration trials.

    slope = sum(f * p) / sum(f^2)   (OLS through origin, no intercept)

Data is windowed from when the load cell first exceeds FORCE_START_THRESHOLD (5 g)
up to and including the force peak.

Usage:
    python analyze_force_pressure.py
    python analyze_force_pressure.py --log-dir human_trial_logs
    python analyze_force_pressure.py --log-dir human_trial_logs --plot
    python analyze_force_pressure.py --plot --save-plot plot_outputs/fp_
"""

import argparse
import csv
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# ── Processing parameters (match plotter.py) ──────────────────────────────────
NUM_SENSORS = 8
RASPBERRY_WINDOW = 10
RASPBERRY_BASE_SAMPLES = 100
LOAD_BASE_SAMPLES = 100
ZERO_DEADBAND = 8.0

DETACH_MIN_FORCE = 5.0
DETACH_DROP_THRESHOLD = 10.0
FORCE_START_OFFSET = 20.0  # g above the trial's initial load cell value — separates grasp from pull
SLOPE_FORCE_MIN = 45.0     # g above pull start — slope calculation begins here, skipping initial transient


# ══════════════════════════════════════════════════════════════════════════════
# IO helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Signal processing
# ══════════════════════════════════════════════════════════════════════════════

def process_raspberry(sensors, window=RASPBERRY_WINDOW, base_samples=RASPBERRY_BASE_SAMPLES,
                      deadband=ZERO_DEADBAND):
    out = np.zeros((NUM_SENSORS, len(sensors[0])), dtype=float)
    for i, ch in enumerate(sensors):
        ch = np.asarray(ch)
        baseline = ch[:base_samples].mean() if len(ch) >= base_samples else ch.mean()
        buf = [baseline] * window
        wi = 0
        smoothed = []
        for x in ch:
            buf[wi] = x
            wi = (wi + 1) % window
            smoothed.append(sum(buf) / window)
        delta = np.asarray(smoothed) - baseline
        delta[np.abs(delta) < deadband] = 0.0
        out[i] = delta
    return out


def aggregate_pressure(processed, method="mean_active"):
    if method == "max":
        return np.max(processed, axis=0)
    agg = np.zeros(processed.shape[1])
    for j in range(processed.shape[1]):
        active = processed[:, j][processed[:, j] > 0]
        agg[j] = active.mean() if active.size > 0 else 0.0
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Per-trial extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_pull_pairs(trial_idx, files, aggregation="mean_active", pressure_start=None):
    """Return (grasp_forces, grasp_pressures, pull_forces, pull_pressures).

    Grasp phase: all samples before the load cell crosses the threshold (force ≈ 0).
    Pull phase: from threshold crossing to force peak — this is where the slope is computed.
    """
    rasp_t, raw_sensors = read_raspberry_csv(files["rasp"])
    load_t, load_force = read_loadcell_csv(files["load"])

    if len(load_t) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    force_interp_raw = np.interp(rasp_t, load_t, load_force)

    processed = process_raspberry(raw_sensors)
    pressure = aggregate_pressure(processed, method=aggregation)

    # Tare using the first n_baseline samples (quiet resting state)
    n_baseline = min(50, len(force_interp_raw) // 4)
    baseline_force = float(force_interp_raw[:n_baseline].mean()) if n_baseline > 0 else float(force_interp_raw[0])
    load_tare = baseline_force
    force_interp = force_interp_raw - load_tare

    if pressure_start is not None:
        # Heuristic mode: slope window starts when pressure first exceeds the threshold
        idx = np.where(pressure >= pressure_start)[0]
        if idx.size == 0:
            print(f"  trial {trial_idx:03d}: pressure never reached {pressure_start}Pa, skipping")
            return np.array([]), np.array([]), np.array([]), np.array([])
        slope_start_i = idx[0]
    else:
        # Default: slope window starts when force crosses FORCE_START_OFFSET then SLOPE_FORCE_MIN
        start_indices = np.where(force_interp_raw >= baseline_force + FORCE_START_OFFSET)[0]
        if start_indices.size == 0:
            print(f"  trial {trial_idx:03d}: force never rose enough, skipping")
            return np.array([]), np.array([]), np.array([]), np.array([])
        start_i = start_indices[0]
        slope_indices = np.where(force_interp >= SLOPE_FORCE_MIN)[0]
        slope_indices = slope_indices[slope_indices >= start_i]
        if slope_indices.size == 0:
            print(f"  trial {trial_idx:03d}: force never reached {SLOPE_FORCE_MIN}g, skipping")
            return np.array([]), np.array([]), np.array([]), np.array([])
        slope_start_i = slope_indices[0]

    # Pull-phase endpoint: whichever comes first — force peak or pressure peak
    force_peak_i = slope_start_i + int(np.argmax(force_interp[slope_start_i:]))
    pressure_peak_i = slope_start_i + int(np.argmax(pressure[slope_start_i:]))
    peak_i = min(force_peak_i, pressure_peak_i)

    if peak_i <= slope_start_i:
        print(f"  trial {trial_idx:03d}: force never rose after slope start, skipping")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Grasp phase + early ramp: everything before slope window (shown in grey)
    grasp_forces = force_interp[:slope_start_i]
    grasp_pressures = pressure[:slope_start_i]

    # Slope window: slope start → force peak (inclusive)
    pull_forces = force_interp[slope_start_i:peak_i + 1]
    pull_pressures = pressure[slope_start_i:peak_i + 1]

    peak_force = float(pull_forces[-1])
    peak_pressure = float(pull_pressures[-1])

    print(f"  trial {trial_idx:03d}: grasp={len(grasp_forces)} samples  slope_window={len(pull_forces)} samples  "
          f"force_peak={peak_force:.2f}g  pressure@force_peak={peak_pressure:.1f}Pa")

    return grasp_forces, grasp_pressures, pull_forces, pull_pressures


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="human_trial_logs")
    parser.add_argument("--aggregation", choices=["mean_active", "max"], default="max",
                        help="How to aggregate 8 pressure channels")
    parser.add_argument("--plot", action="store_true", help="Show per-trial plots")
    parser.add_argument("--save-plot", default=None, metavar="PREFIX",
                        help="Save plots to PREFIX<trial_idx>.png instead of showing")
    parser.add_argument("--save-summary-plot", default=None, metavar="PATH",
                        help="Save summary scatter (Δforce vs Δpressure) to PATH")
    parser.add_argument("--ref-slope", type=float, default=72.0, metavar="PA_PER_G",
                        help="Reference slope to draw on every plot (default: 72.0 Pa/g)")
    parser.add_argument("--pressure-start", type=float, default=None, metavar="PA",
                        help="Use pressure threshold (Pa) to detect pull start instead of load cell")
    args = parser.parse_args()

    trials = discover_trials(args.log_dir)
    if not trials:
        print(f"No trials found in {args.log_dir}")
        return

    print(f"Found {len(trials)} trials in '{args.log_dir}'\n")

    all_grasp_forces, all_grasp_pressures = [], []
    all_forces, all_pressures, trial_indices = [], [], []
    per_trial_slopes = []  # (idx, slope)

    for idx in sorted(trials):
        gf, gp, f, p = extract_pull_pairs(idx, trials[idx], aggregation=args.aggregation,
                                           pressure_start=args.pressure_start)
        if f.size < 2:
            continue
        p_rel = p - p[0]  # pressure change relative to slope-window start
        peak_idx = len(f) - 1  # window ends at force peak
        force_start = float(f[0])
        force_peak = float(f[peak_idx])
        pressure_start = float(p[0])
        pressure_peak = float(p[peak_idx])
        delta_pressure = float(p_rel[peak_idx])
        delta_force = force_peak - force_start
        if delta_force <= 0:
            continue
        slope = delta_pressure / delta_force
        all_grasp_forces.append(gf)
        all_grasp_pressures.append(gp)
        all_forces.append(f)
        all_pressures.append(p)
        trial_indices.append(idx)
        per_trial_slopes.append((idx, slope, force_start, pressure_start, force_peak, pressure_peak))

    if not per_trial_slopes:
        print("\nNo pull-phase data extracted.")
        return

    slopes = np.array([s for _, s, *_ in per_trial_slopes])
    mean_slope = float(slopes.mean())
    std_slope = float(slopes.std())

    print(f"\n── Per-trial ΔPa/Δg (start+{FORCE_START_OFFSET}g → peak) ──")
    print(f"  {'trial':>5}  {'peak_force(g)':>13}  {'slope(Pa/g)':>11}")
    for idx, s, fs, ps, fp, pp in per_trial_slopes:
        print(f"  {idx:>5}  {fp:>13.2f}  {s:>11.3f}")

    print(f"\n── Global summary ──")
    print(f"  mean slope = {mean_slope:.4f} Pa/g")
    print(f"  std  slope = {std_slope:.4f} Pa/g  ({100*std_slope/mean_slope:.1f}% of mean)")

    print(f"\nPaste into PIDRaspberryAgent:")
    print(f"  PIDRaspberryAgent(")
    print(f"      pressure_slope={mean_slope:.2f},")
    print(f"      pressure_aggregation='{args.aggregation}',")
    print(f"  )")

    # ── Summary scatter: Δforce vs Δpressure (like peak2peak but from fp window) ──
    if args.save_summary_plot or not (args.plot or args.save_plot):
        delta_forces = np.array([fp - fs for _, _, fs, _, fp, _ in per_trial_slopes])
        delta_pressures = np.array([pp - ps for _, _, _, ps, _, pp in per_trial_slopes])
        idxs = [idx for idx, *_ in per_trial_slopes]

        fit_slope = float(np.dot(delta_forces, delta_pressures) / np.dot(delta_forces, delta_forces))
        ratios = delta_pressures / delta_forces
        std_ratio = float(ratios.std())

        fig_s, ax_s = plt.subplots(figsize=(7, 5))
        ax_s.scatter(delta_forces, delta_pressures, s=50, color="steelblue", zorder=3, label="trials")
        x_line = np.linspace(0, delta_forces.max() * 1.1, 200)
        ax_s.plot(x_line, fit_slope * x_line, "k-", linewidth=2,
                  label=f"fit slope = {fit_slope:.2f} Pa/g")
        ax_s.fill_between(x_line, (fit_slope - std_ratio) * x_line, (fit_slope + std_ratio) * x_line,
                          alpha=0.2, color="black", label=f"±1 std ratio ({std_ratio:.2f} Pa/g)")
        ax_s.set_xlim(left=0)
        ax_s.set_ylim(bottom=0)
        ax_s.set_xlabel("Δ load cell force [g]  (slope window start → force peak)")
        ax_s.set_ylabel(f"Δ pressure ({args.aggregation}) [Pa]  (slope window start → force peak)")
        ax_s.set_title("FP slope window: Δforce vs Δpressure across all trials")
        ax_s.legend(fontsize=9)
        ax_s.grid(True)
        plt.tight_layout()
        if args.save_summary_plot:
            fig_s.savefig(args.save_summary_plot, dpi=150, bbox_inches="tight")
            print(f"Saved {args.save_summary_plot}")
            plt.close(fig_s)
        else:
            plt.show()

    if not (args.plot or args.save_plot):
        return

    # One plot per trial
    for i, (idx, trial_slope, force_start, pressure_start, force_peak, pressure_peak) in enumerate(per_trial_slopes):
        gf = all_grasp_forces[i]
        gp = all_grasp_pressures[i]
        f = all_forces[i]
        p = all_pressures[i]

        plt.rcParams.update({
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        })

        fig, ax = plt.subplots(figsize=(7, 5))

        # Pre-slope points: grey (grasp + ramp up to slope window start)
        if gf.size > 0:
            ax.scatter(gf, gp, s=14, alpha=0.4, color="grey", zorder=2, label="pre-slope")

        # Pull phase: colored points used for slope
        ax.scatter(f, p, s=18, alpha=0.6, color="steelblue", zorder=3, label="pull (slope window)")

        # Mark start and peak points
        ax.scatter([force_start], [pressure_start], s=60, color="green",
                   zorder=5, marker="o", label=f"pull start ({force_start:.1f}g, {pressure_start:.0f}Pa)")
        ax.scatter([force_peak], [pressure_peak], s=60, color="crimson",
                   zorder=5, marker="o", label=f"force peak ({force_peak:.1f}g, {pressure_peak:.0f}Pa)")

        # X range: start → 10% beyond peak
        x_line = np.array([force_start, force_peak * 1.1])

        # Per-trial slope line through (force_start, pressure_start)
        ax.plot(x_line, pressure_start + trial_slope * (x_line - force_start),
                color="steelblue", linewidth=1.4, linestyle="--",
                label=f"trial slope = {trial_slope:.2f} Pa/g")

        # Reference slope
        ax.plot(x_line, pressure_start + args.ref_slope * (x_line - force_start),
                color="darkorange", linewidth=2, linestyle="-.",
                label=f"ref slope = {args.ref_slope:.1f} Pa/g")

        ax.set_xlabel("Load cell force [g]")
        ax.set_ylabel(f"Pressure above baseline ({args.aggregation}) [Pa]")
        ax.set_title(f"Trial {idx:03d} — force vs pressure")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        if args.save_plot:
            path = f"{args.save_plot}{idx:03d}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()