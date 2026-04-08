import argparse
import csv
import math
import os
import re

import matplotlib.pyplot as plt

from robot_imitation_glue.uR3station.raspberry_trial_utils import detect_detach, process_raspberry_signals

NUM_SENSORS = 8
WINDOW = 10
BASE_SAMPLES = 100
ZERO_DEADBAND = 8.0
DETACH_DROP_THRESHOLD = 0.01
DETACH_MIN_FORCE = 0.05
ANYSKIN_SMOOTH_WINDOW = 10
ANYSKIN_SLIP_SMOOTH_WINDOW = 10


def moving_average(signal, window):
    if not signal:
        return []
    history = [signal[0]] * window
    write_index = 0
    out = []
    for x in signal:
        history[write_index] = x
        write_index = (write_index + 1) % window
        out.append(sum(history) / window)
    return out


def discover_trials(log_dir):
    trials = {}
    for name in os.listdir(log_dir):
        full_path = os.path.join(log_dir, name)
        if not os.path.isdir(full_path):
            continue
        match = re.match(r"trial_(\d+)", name)
        if not match:
            continue
        trial_idx = int(match.group(1))
        trials[trial_idx] = {}
        for key, fname in [("rasp", "raspberry.csv"), ("load", "loadcell.csv"), ("event", "events.csv"), ("anyskin", "anyskin.csv")]:
            path = os.path.join(full_path, fname)
            if os.path.isfile(path):
                trials[trial_idx][key] = path
    return trials


def read_raspberry_csv(filename):
    t = []
    sensors = [[] for _ in range(NUM_SENSORS)]
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t_pc"]))
            for i in range(NUM_SENSORS):
                sensors[i].append(float(row[f"S{i}"]))
    return t, sensors


def read_loadcell_csv(filename):
    t, force = [], []
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t_pc"]))
            force.append(float(row["force"]))
    return t, force


def read_event_csv(filename):
    rows = []
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["t_pc"] = float(row["t_pc"])
            rows.append(row)
    return rows


def read_anyskin_csv(filename):
    rows = []
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames[:] if reader.fieldnames else []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in ("t_pc", "t_sensor"):
                    parsed[k] = float(v)
                elif k == "sample_idx":
                    parsed[k] = int(v)
                else:
                    parsed[k] = float(v)
            rows.append(parsed)
    return fieldnames, rows


def process_anyskin_rows(rows, fieldnames):
    mags = []
    i = 0
    while f"m{i}_x" in fieldnames and f"m{i}_y" in fieldnames and f"m{i}_z" in fieldnames:
        mags.append(i)
        i += 1
    t = [row["t_pc"] for row in rows]
    mag_signals = {}
    slip_signals = {}
    for i in mags:
        raw_mag = []
        for row in rows:
            x = row[f"m{i}_x"]
            y = row[f"m{i}_y"]
            z = row[f"m{i}_z"]
            raw_mag.append(math.sqrt(x*x + y*y + z*z))
        mag = moving_average(raw_mag, ANYSKIN_SMOOTH_WINDOW)
        slip = [0.0]
        for j in range(1, len(mag)):
            dt = t[j] - t[j-1]
            slip.append(0.0 if dt <= 0 else abs(mag[j] - mag[j-1]) / dt)
        slip = moving_average(slip, ANYSKIN_SLIP_SMOOTH_WINDOW)
        mag_signals[i] = mag
        slip_signals[i] = slip
    return t, mags, mag_signals, slip_signals


def draw_event_lines(ax, event_rows):
    y0, y1 = ax.get_ylim()
    y_text = y1 - 0.05 * (y1 - y0)
    for row in event_rows:
        t_evt = row["t_pc"]
        ax.axvline(t_evt, alpha=0.35)
        ax.text(t_evt, y_text, row["event"], rotation=90, verticalalignment="top", fontsize=8)


def plot_one_trial(trial_idx, files, output_dir):
    rasp_t, raw_sensors = read_raspberry_csv(files["rasp"])
    load_t, load_force = read_loadcell_csv(files["load"])
    event_rows = read_event_csv(files["event"])
    _, processed_sensors = process_raspberry_signals(raw_sensors, WINDOW, BASE_SAMPLES, ZERO_DEADBAND)
    detach_t, _ = detect_detach(load_t, load_force, DETACH_DROP_THRESHOLD, DETACH_MIN_FORCE)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    for i in range(NUM_SENSORS):
        ax1.plot(rasp_t, processed_sensors[i], label=f"S{i}")
    ax2 = ax1.twinx()
    ax2.plot(load_t, load_force, linestyle="--", label="Load cell force")
    if detach_t is not None:
        ax1.axvline(detach_t, linestyle=":", linewidth=2)
        ax2.axvline(detach_t, linestyle=":", linewidth=2)
    draw_event_lines(ax1, event_rows)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"Trial {trial_idx:03d}: processed raspberry + load")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"trial_{trial_idx:03d}_overview.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    if "anyskin" in files:
        anyskin_fieldnames, anyskin_rows = read_anyskin_csv(files["anyskin"])
        t, mags, mag_signals, slip_signals = process_anyskin_rows(anyskin_rows, anyskin_fieldnames)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for i in mags:
            ax1.plot(t, mag_signals[i], label=f"m{i}_mag")
            ax2.plot(t, slip_signals[i], label=f"m{i}_slip")
        if detach_t is not None:
            ax1.axvline(detach_t, linestyle=":", linewidth=2)
            ax2.axvline(detach_t, linestyle=":", linewidth=2)
        draw_event_lines(ax1, event_rows)
        draw_event_lines(ax2, event_rows)
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"trial_{trial_idx:03d}_anyskin.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-log-dir", default="trial_logs_policy")
    parser.add_argument("--output-dir", default="plot_outputs_policy")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    trials = discover_trials(args.trial_log_dir)
    for trial_idx in sorted(trials.keys()):
        if {"rasp", "load", "event"}.issubset(trials[trial_idx]):
            plot_one_trial(trial_idx, trials[trial_idx], args.output_dir)


if __name__ == "__main__":
    main()
