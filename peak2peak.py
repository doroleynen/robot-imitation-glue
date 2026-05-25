"""
Peak-to-peak Pa/g relation across human demonstration trials.

For each trial: find the peak force and the pressure at that moment.
Fit a line through the origin to the scatter of (peak_force, peak_pressure) points.

Usage:
    python analyze_peak_to_peak.py
    python analyze_peak_to_peak.py --log-dir human_trial_logs --plot
    python analyze_peak_to_peak.py --save-plot plot_outputs/p2p.png
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
LOAD_BASE_SAMPLES = 100
ZERO_DEADBAND = 8.0
FORCE_START_OFFSET = 20.0  # g above baseline — same threshold as analyze_force_pressure.py


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


def extract_peaks(trial_idx, files, aggregation="max", pressure_start=None):
    rasp_t, raw_sensors = read_raspberry_csv(files["rasp"])
    load_t, load_force = read_loadcell_csv(files["load"])

    if len(load_t) < 2:
        return None

    force_interp_raw = np.interp(rasp_t, load_t, load_force)

    processed = process_raspberry(raw_sensors)
    if aggregation == "max":
        pressure = np.max(processed, axis=0)
    else:
        pressure = np.zeros(processed.shape[1])
        for j in range(processed.shape[1]):
            active = processed[:, j][processed[:, j] > 0]
            pressure[j] = active.mean() if active.size > 0 else 0.0

    force_interp = force_interp_raw  # CSVs are pre-tared; no internal offset needed

    if pressure_start is not None:
        idx = np.where(pressure >= pressure_start)[0]
        if idx.size == 0:
            print(f"  trial {trial_idx:03d}: pressure never reached {pressure_start}Pa, skipping")
            return None
        start_i = idx[0]
    else:
        # Find pull start: first sample where force exceeds FORCE_START_OFFSET above baseline
        start_indices = np.where(force_interp >= FORCE_START_OFFSET)[0]
        if start_indices.size == 0:
            print(f"  trial {trial_idx:03d}: force never rose enough, skipping")
            return None
        start_i = start_indices[0]

    force_start = float(force_interp[start_i])
    pressure_start = float(pressure[start_i])

    peak_i = start_i + int(np.argmax(force_interp[start_i:]))
    peak_force = float(force_interp[peak_i])
    peak_pressure = float(pressure[peak_i])

    delta_force = peak_force - force_start
    delta_pressure = peak_pressure - pressure_start

    if delta_force <= 0:
        print(f"  trial {trial_idx:03d}: delta force <= 0, skipping")
        return None

    print(f"  trial {trial_idx:03d}: Δforce={delta_force:.2f}g  Δpressure={delta_pressure:.1f}Pa  ratio={delta_pressure/delta_force:.2f} Pa/g")
    return delta_force, delta_pressure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="human_trial_logs")
    parser.add_argument("--aggregation", choices=["mean_active", "max"], default="max")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", default=None, metavar="PATH")
    parser.add_argument("--pressure-start", type=float, default=None, metavar="PA",
                        help="Use pressure threshold (Pa) to detect pull start instead of load cell")
    args = parser.parse_args()

    trials = discover_trials(args.log_dir)
    if not trials:
        print(f"No trials found in {args.log_dir}")
        return

    print(f"Found {len(trials)} trials in '{args.log_dir}'\n")

    peak_forces, peak_pressures, trial_indices = [], [], []

    for idx in sorted(trials):
        result = extract_peaks(idx, trials[idx], aggregation=args.aggregation,
                               pressure_start=args.pressure_start)
        if result is None:
            continue
        pf, pp = result
        peak_forces.append(pf)
        peak_pressures.append(pp)
        trial_indices.append(idx)

    if not peak_forces:
        print("\nNo data extracted.")
        return

    peak_forces = np.array(peak_forces)
    peak_pressures = np.array(peak_pressures)

    # Per-trial ratio
    ratios = peak_pressures / peak_forces

    # Global fit: line through origin, slope = sum(f*p) / sum(f^2)
    slope = float(np.dot(peak_forces, peak_pressures) / np.dot(peak_forces, peak_forces))
    mean_ratio = float(ratios.mean())
    std_ratio = float(ratios.std())

    print(f"\n── Per-trial peak-to-peak ratios (Δforce, Δpressure from trial start) ──")
    print(f"  {'trial':>5}  {'Δforce(g)':>10}  {'Δpressure(Pa)':>14}  {'ratio(Pa/g)':>11}")
    for idx, pf, pp, r in zip(trial_indices, peak_forces, peak_pressures, ratios):
        print(f"  {idx:>5}  {pf:>10.2f}  {pp:>14.1f}  {r:>11.3f}")

    print(f"\n── Global summary ──")
    print(f"  fit slope (origin) = {slope:.4f} Pa/g")
    print(f"  mean ratio         = {mean_ratio:.4f} Pa/g")
    print(f"  std ratio          = {std_ratio:.4f} Pa/g  ({100*std_ratio/mean_ratio:.1f}% of mean)")

    if not (args.plot or args.save_plot):
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(peak_forces, peak_pressures, s=50, color="steelblue", zorder=3, label="trials")
    for idx, pf, pp in zip(trial_indices, peak_forces, peak_pressures):
        ax.annotate(str(idx), (pf, pp), textcoords="offset points", xytext=(4, 3), fontsize=7, color="steelblue")

    x_line = np.linspace(0, peak_forces.max() * 1.1, 200)
    ax.plot(x_line, slope * x_line, "k-", linewidth=2, label=f"fit slope = {slope:.2f} Pa/g")
    ax.fill_between(x_line,
                    (slope - std_ratio) * x_line,
                    (slope + std_ratio) * x_line,
                    alpha=0.2, color="black", label=f"±1 std ratio ({std_ratio:.2f} Pa/g)")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Δ load cell force [g]  (peak − start)")
    ax.set_ylabel(f"Δ pressure ({args.aggregation}) [Pa]  (peak − start)")
    ax.set_title("Peak-to-peak Pa/g relation across all trials (delta from trial start)")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        print(f"\nSaved {args.save_plot}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()