import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


NUM_RASPBERRY_SENSORS = 8


def parse_raspberry_line(line: str):
    parts = line.strip().split(",")
    if len(parts) != NUM_RASPBERRY_SENSORS:
        return None

    vals = {}
    for part in parts:
        if ":" not in part:
            return None
        key, value = part.split(":", 1)
        vals[key] = float(value)
    return vals


def parse_loadcell_line(line: str):
    line = line.strip()
    if not line:
        return None
    return float(line)


def make_anyskin_fieldnames(num_mags: int, temp_filtered: bool = True):
    names = []
    if temp_filtered:
        for i in range(num_mags):
            names.extend([f"m{i}_x", f"m{i}_y", f"m{i}_z"])
    else:
        for i in range(num_mags):
            names.extend([f"m{i}_t", f"m{i}_x", f"m{i}_y", f"m{i}_z"])
    return names


def moving_average_with_baseline_prefill(signal, baseline, window):
    history = [baseline] * window
    write_index = 0
    out = []
    for x in signal:
        history[write_index] = x
        write_index = (write_index + 1) % window
        out.append(sum(history) / window)
    return out


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


def compute_baselines(sensors, base_samples=100):
    baselines = []
    for ch in sensors:
        n = min(base_samples, len(ch))
        if n == 0:
            baselines.append(0.0)
        else:
            baselines.append(sum(ch[:n]) / n)
    return baselines


def process_raspberry_signals(sensors, window, base_samples, zero_deadband=0.0):
    baselines = compute_baselines(sensors, base_samples)
    processed = []
    for i in range(len(sensors)):
        avg = moving_average_with_baseline_prefill(sensors[i], baselines[i], window)
        delta = []
        for a in avg:
            d = a - baselines[i]
            if abs(d) < zero_deadband:
                d = 0.0
            delta.append(d)
        processed.append(delta)
    return baselines, processed


def detect_anyskin_mags(fieldnames):
    mags = []
    i = 0
    while f"m{i}_x" in fieldnames and f"m{i}_y" in fieldnames and f"m{i}_z" in fieldnames:
        mags.append(i)
        i += 1
    return mags


def compute_slip_proxy(t, signal):
    if len(signal) == 0:
        return []
    slip = [0.0]
    for i in range(1, len(signal)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            slip.append(0.0)
        else:
            slip.append(abs(signal[i] - signal[i - 1]) / dt)
    return slip


def process_anyskin_rows(rows, fieldnames, smooth_window=5, slip_smooth_window=5):
    if not rows:
        return None

    mags = detect_anyskin_mags(fieldnames)
    t = [row["t_pc"] for row in rows]
    sample_idx = [row["sample_idx"] for row in rows]

    mag_signals = {}
    slip_signals = {}

    for i in mags:
        raw_mag = []
        for row in rows:
            x = row[f"m{i}_x"]
            y = row[f"m{i}_y"]
            z = row[f"m{i}_z"]
            raw_mag.append(math.sqrt(x * x + y * y + z * z))

        mag = moving_average(raw_mag, smooth_window)
        slip = compute_slip_proxy(t, mag)
        slip = moving_average(slip, slip_smooth_window)

        mag_signals[i] = mag
        slip_signals[i] = slip

    return {
        "t": t,
        "sample_idx": sample_idx,
        "mags": mags,
        "mag_signals": mag_signals,
        "slip_signals": slip_signals,
    }


def detect_detach(load_t, load_force, drop_threshold=0.02, min_force=0.05):
    if len(load_force) < 2:
        return None, None

    armed = False
    running_peak = None
    for i in range(len(load_force)):
        force = load_force[i]
        if not armed and force >= min_force:
            armed = True
            running_peak = force
        if not armed:
            continue
        if force > running_peak:
            running_peak = force
        if (running_peak - force) >= drop_threshold:
            return load_t[i], i
    return None, None


@dataclass
class OnlineFeatureConfig:
    num_raspberry_sensors: int = 8
    num_anyskin_mags: int = 5
    raspberry_window: int = 10
    raspberry_base_samples: int = 100
    raspberry_trend_horizon: int = 30  # steps over which to compute pressure trend
    zero_deadband: float = 8.0
    anyskin_smooth_window: int = 10
    anyskin_slip_threshold: float = 30.0  # raw step-to-step magnitude diff; noise floor max ~28, genuine slip up to ~174
    detach_drop_threshold: float = 0.01
    detach_min_force: float = 0.05


class RunningAverage:
    def __init__(self, window: int, prefill_value: float = 0.0):
        self.window = max(1, int(window))
        self.values: Deque[float] = deque([prefill_value] * self.window, maxlen=self.window)
        self.total = prefill_value * self.window

    def update(self, value: float) -> float:
        oldest = self.values[0]
        self.total -= oldest
        self.values.append(value)
        self.total += value
        return self.total / len(self.values)


class BaselineEstimator:
    def __init__(self, n_samples: int):
        self.n_samples = max(1, int(n_samples))
        self.values = []

    def update(self, value: float) -> float:
        if len(self.values) < self.n_samples:
            self.values.append(value)
        return self.baseline

    @property
    def baseline(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))

    @property
    def ready(self) -> bool:
        return len(self.values) >= self.n_samples


class OnlineFeatureProcessor:
    def __init__(self, cfg: OnlineFeatureConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.raspberry_baselines = [BaselineEstimator(self.cfg.raspberry_base_samples) for _ in range(self.cfg.num_raspberry_sensors)]
        self.raspberry_avgs = [RunningAverage(self.cfg.raspberry_window, 0.0) for _ in range(self.cfg.num_raspberry_sensors)]
        self.raspberry_trend_buf: Deque[np.ndarray] = deque(maxlen=self.cfg.raspberry_trend_horizon)
        self.anyskin_mag_avgs = [RunningAverage(self.cfg.anyskin_smooth_window, 0.0) for _ in range(self.cfg.num_anyskin_mags)]
        self.prev_anyskin_mag_raw = np.zeros(self.cfg.num_anyskin_mags, dtype=np.float32)
        self.prev_load_force = 0.0
        self.running_peak_force = None
        self.detach_armed = False

    def _process_raspberry(self, raw_pressures: List[float]):
        processed = []
        for i, value in enumerate(raw_pressures):
            baseline = self.raspberry_baselines[i].update(value)
            avg = self.raspberry_avgs[i].update(value if self.raspberry_baselines[i].ready else baseline)
            delta = avg - baseline
            if abs(delta) < self.cfg.zero_deadband:
                delta = 0.0
            processed.append(delta)
        processed_arr = np.asarray(processed, dtype=np.float32)
        # Trend: diff against value from trend_horizon steps ago (or oldest available)
        oldest = self.raspberry_trend_buf[0] if self.raspberry_trend_buf else np.zeros_like(processed_arr)
        diff = processed_arr - oldest
        self.raspberry_trend_buf.append(processed_arr.copy())
        return processed_arr, diff.astype(np.float32)

    def _process_anyskin(self, raw_anyskin: List[float]):
        mags = []
        raw_mags = []
        slips = []
        if len(raw_anyskin) < 3 * self.cfg.num_anyskin_mags:
            raw_anyskin = list(raw_anyskin) + [0.0] * (3 * self.cfg.num_anyskin_mags - len(raw_anyskin))
        for i in range(self.cfg.num_anyskin_mags):
            x, y, z = raw_anyskin[3 * i : 3 * i + 3]
            mag_raw = math.sqrt(x * x + y * y + z * z)
            mag = self.anyskin_mag_avgs[i].update(mag_raw)
            slip_raw = abs(mag_raw - float(self.prev_anyskin_mag_raw[i]))
            slip = slip_raw if slip_raw >= self.cfg.anyskin_slip_threshold else 0.0
            mags.append(mag)
            raw_mags.append(mag_raw)
            slips.append(slip)
        mags_arr = np.asarray(mags, dtype=np.float32)
        slips_arr = np.asarray(slips, dtype=np.float32)
        self.prev_anyskin_mag_raw = np.asarray(raw_mags, dtype=np.float32)
        return mags_arr, slips_arr

    def _process_load(self, raw_force: float):
        load_diff = float(raw_force - self.prev_load_force)
        self.prev_load_force = float(raw_force)

        detach = False
        if not self.detach_armed and raw_force >= self.cfg.detach_min_force:
            self.detach_armed = True
            self.running_peak_force = raw_force
        if self.detach_armed:
            self.running_peak_force = max(float(self.running_peak_force), float(raw_force))
            if (self.running_peak_force - raw_force) >= self.cfg.detach_drop_threshold:
                detach = True
        return np.asarray([raw_force, load_diff], dtype=np.float32), detach

    def process(self, raw_pressures: List[float], raw_anyskin: List[float], raw_force: float):
        raspberry_proc, raspberry_diff = self._process_raspberry(raw_pressures)
        anyskin_mag, anyskin_slip = self._process_anyskin(raw_anyskin)
        load_vec, detach = self._process_load(raw_force)
        state = np.concatenate([
            raspberry_proc,
            raspberry_diff,
            anyskin_mag,
            anyskin_slip,
            load_vec,
        ]).astype(np.float32)
        return {
            "state": state,
            "raspberry_state": raspberry_proc.astype(np.float32),
            "raspberry_diff": raspberry_diff.astype(np.float32),
            "anyskin_mag": anyskin_mag.astype(np.float32),
            "anyskin_slip": anyskin_slip.astype(np.float32),
            "loadcell_state": load_vec.astype(np.float32),
            "detach_detected": detach,
        }
