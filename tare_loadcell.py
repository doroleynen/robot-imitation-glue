"""
Tare load cell CSVs in-place.

For each trial directory that has both loadcell.csv and events.csv:
  1. Find the 'safe_pose_reached' event timestamp.
  2. Compute the mean of loadcell samples in [t_safe, t_safe + WINDOW_S] as the offset.
  3. Subtract the offset from every 'force' value in loadcell.csv and write back.

Fallback: if no safe_pose_reached event exists, use the mean of the first FALLBACK_N samples.
"""
import argparse
import csv
import io
import os
import re


WINDOW_S = 3.0
FALLBACK_N = 15


def find_safe_pose_t(events_path):
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["event"] == "safe_pose_reached":
                return float(row["t_pc"])
    return None


def tare_loadcell(loadcell_path, events_path, dry_run=False):
    with open(loadcell_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames[:]
        rows = list(reader)

    if not rows:
        return None

    t_safe = find_safe_pose_t(events_path) if events_path else None

    if t_safe is not None:
        window = [float(r["force"]) for r in rows if t_safe <= float(r["t_pc"]) <= t_safe + WINDOW_S]
        method = f"safe_pose+{WINDOW_S}s window (n={len(window)})"
    else:
        window = []
        method = None

    if not window:
        window = [float(r["force"]) for r in rows[:FALLBACK_N]]
        method = f"first {len(window)} samples (no safe_pose event)"

    offset = sum(window) / len(window)

    if not dry_run:
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            r["force"] = f"{float(r['force']) - offset:.4f}"
            writer.writerow(r)
        with open(loadcell_path, "w", newline="") as f:
            f.write(out.getvalue())

    return offset, method


def discover_trials(log_dir):
    trials = []
    for name in sorted(os.listdir(log_dir)):
        full = os.path.join(log_dir, name)
        if not os.path.isdir(full):
            continue
        if not re.match(r"trial_\d+", name):
            continue
        lc = os.path.join(full, "loadcell.csv")
        ev = os.path.join(full, "events.csv")
        if os.path.isfile(lc):
            trials.append((name, lc, ev if os.path.isfile(ev) else None))
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dirs", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print offsets without modifying files")
    args = parser.parse_args()

    total = 0
    for log_dir in args.log_dirs:
        trials = discover_trials(log_dir)
        if not trials:
            print(f"{log_dir}: no trials found")
            continue
        print(f"\n{log_dir} ({len(trials)} trials)")
        for name, lc_path, ev_path in trials:
            result = tare_loadcell(lc_path, ev_path, dry_run=args.dry_run)
            if result is None:
                print(f"  {name}: empty loadcell.csv, skipped")
                continue
            offset, method = result
            action = "[dry-run]" if args.dry_run else "tared"
            print(f"  {name}: {action} offset={offset:+.2f}g  ({method})")
            total += 1

    print(f"\n{'Would tare' if args.dry_run else 'Tared'} {total} loadcell.csv files.")


if __name__ == "__main__":
    main()
