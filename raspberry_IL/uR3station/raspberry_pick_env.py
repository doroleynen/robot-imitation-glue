import csv
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import serial
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from anyskin import AnySkinBase

from raspberry_IL.base import BaseEnv
from raspberry_IL.uR3station.raspberry_trial_utils import (
    OnlineFeatureConfig,
    OnlineFeatureProcessor,
    make_anyskin_fieldnames,
    parse_loadcell_line,
    parse_raspberry_line,
)


class RaspberryPickEnv(BaseEnv):
    ACTION_SPEC = "GRIPPER_DELTA"
    PROPRIO_OBS_SPEC = "PROCESSED_TACTILE_FORCE"

    def __init__(
        self,
        robot_ip: str = "10.42.0.163",
        raspberry_port: str = "/dev/ttyACM3",
        loadcell_port: str = "/dev/ttyACM1",
        anyskin_port: str = "/dev/ttyACM2",
        baud_rate: int = 115200,
        anyskin_num_mags: int = 5,
        anyskin_temp_filtered: bool = True,
        anyskin_burst_mode: bool = True,
        anyskin_baudrate: int = 115200,
        arm_joint_speed: float = 0.15,
        gripper_speed: float = 0.03,
        gripper_force: float = 40.0,
        initial_open_width: float = 0.085,
        close_action_scale: float = 1.0,
        auto_start_pull_on_contact: bool = True,
        trial_log_root: str = "trial_logs_policy",
        feature_cfg: Optional[OnlineFeatureConfig] = None,
    ):
        self.robot_ip = robot_ip
        self.raspberry_port = raspberry_port
        self.loadcell_port = loadcell_port
        self.anyskin_port = anyskin_port
        self.baud_rate = baud_rate
        self.anyskin_num_mags = anyskin_num_mags
        self.anyskin_temp_filtered = anyskin_temp_filtered
        self.anyskin_burst_mode = anyskin_burst_mode
        self.anyskin_baudrate = anyskin_baudrate
        self.anyskin_fields = make_anyskin_fieldnames(anyskin_num_mags, anyskin_temp_filtered)
        self.arm_joint_speed = arm_joint_speed
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        self.initial_open_width = initial_open_width
        self.close_action_scale = close_action_scale
        self.auto_start_pull_on_contact = auto_start_pull_on_contact
        self.feature_cfg = feature_cfg or OnlineFeatureConfig(num_anyskin_mags=anyskin_num_mags)
        self.processor = OnlineFeatureProcessor(self.feature_cfg)
        self.trial_log_root = Path(trial_log_root)
        self.trial_log_root.mkdir(parents=True, exist_ok=True)

        self.SAFE_Q = np.array([-1.95987827, -3.30249323, 0.78052837, -2.17082896, -1.58573944, -1.50839597], dtype=float)
        self.APPROACH_Q = np.array([-1.11505634, -3.45552363, 0.50535185, -1.8370768, -1.58581144, -1.5083831], dtype=float)
        self.GRASP_Q = np.array([-1.07907039, -3.29095234, 0.50338775, -1.98487296, -1.54438192, -1.97415287], dtype=float)
        self.PULL_Q = np.array([-1.1152199, -3.03422322, -0.42365354, -1.29017635, -1.58889419, -1.97281915], dtype=float)

        self.robot = URrtde(self.robot_ip)
        self.gripper = Robotiq2F85(self.robot_ip)

        self._raspberry_serial = None
        self._loadcell_serial = None
        self._anyskin_sensor = None
        self._stop_threads = False
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._latest_raspberry = [0.0] * self.feature_cfg.num_raspberry_sensors
        self._latest_load_force = 0.0
        self._latest_anyskin = [0.0] * (3 * self.anyskin_num_mags)
        self._raspberry_sample_idx = 0
        self._loadcell_sample_idx = 0
        self._anyskin_sample_idx = 0

        self.episode_done = False
        self.detach_detected = False
        self.pull_started = False
        self.contact_started = False
        self.pull_start_time = None
        self.step_count = 0
        self.last_obs = None
        self.current_trial_idx = 0
        self.event_rows = []
        self.raspberry_rows = []
        self.loadcell_rows = []
        self.anyskin_rows = []

        self._start_sensor_threads()
        time.sleep(2.0)

    def _start_sensor_threads(self):
        self._stop_threads = False
        self._raspberry_serial = serial.Serial(self.raspberry_port, self.baud_rate, timeout=1)
        self._loadcell_serial = serial.Serial(self.loadcell_port, self.baud_rate, timeout=1)
        self._anyskin_sensor = AnySkinBase(
            num_mags=self.anyskin_num_mags,
            port=self.anyskin_port,
            baudrate=self.anyskin_baudrate,
            burst_mode=self.anyskin_burst_mode,
            temp_filtered=self.anyskin_temp_filtered,
        )

        self._threads = [
            threading.Thread(target=self._raspberry_reader, daemon=True),
            threading.Thread(target=self._loadcell_reader, daemon=True),
            threading.Thread(target=self._anyskin_reader, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def close(self):
        self._stop_threads = True
        for thread in self._threads:
            thread.join(timeout=1.0)
        try:
            if self._raspberry_serial is not None:
                self._raspberry_serial.close()
        except Exception:
            pass
        try:
            if self._loadcell_serial is not None:
                self._loadcell_serial.close()
        except Exception:
            pass
        try:
            if self._anyskin_sensor is not None:
                self._anyskin_sensor.close()
        except Exception:
            pass

    def _raspberry_reader(self):
        while not self._stop_threads:
            try:
                line = self._raspberry_serial.readline().decode(errors="ignore").strip()
                vals = parse_raspberry_line(line)
                if vals is None:
                    continue
                row = {"t_pc": time.perf_counter(), "sample_idx": self._raspberry_sample_idx}
                latest = []
                for i in range(self.feature_cfg.num_raspberry_sensors):
                    value = vals[f"S{i}"]
                    row[f"S{i}"] = value
                    latest.append(value)
                with self._lock:
                    self._latest_raspberry = latest
                    self._raspberry_sample_idx += 1
                    if self.current_trial_idx > 0:
                        self.raspberry_rows.append(row)
            except Exception:
                pass

    def _loadcell_reader(self):
        while not self._stop_threads:
            try:
                line = self._loadcell_serial.readline().decode(errors="ignore").strip()
                force = parse_loadcell_line(line)
                if force is None:
                    continue
                row = {"t_pc": time.perf_counter(), "sample_idx": self._loadcell_sample_idx, "force": force}
                with self._lock:
                    self._latest_load_force = force
                    self._loadcell_sample_idx += 1
                    if self.current_trial_idx > 0:
                        self.loadcell_rows.append(row)
            except Exception:
                pass

    def _anyskin_reader(self):
        while not self._stop_threads:
            try:
                t_sensor, sample = self._anyskin_sensor.get_sample()
                row = {"t_pc": time.perf_counter(), "t_sensor": float(t_sensor), "sample_idx": self._anyskin_sample_idx}
                with self._lock:
                    self._latest_anyskin = list(sample)
                    for name, value in zip(self.anyskin_fields, sample):
                        row[name] = float(value)
                    self._anyskin_sample_idx += 1
                    if self.current_trial_idx > 0:
                        self.anyskin_rows.append(row)
            except Exception:
                pass

    def log_event(self, name: str, extra: Optional[dict] = None):
        row = {"t_pc": time.perf_counter(), "event": name, "trial_idx": self.current_trial_idx}
        if extra:
            row.update(extra)
        self.event_rows.append(row)
        print(f"[EVENT] {name}")

    def move_robot_to_tcp_pose(self, pose):
        self.robot.servo_to_tcp_pose(pose, 0.1)

    def move_gripper(self, width):
        width = float(np.clip(width, self.gripper.gripper_specs.min_width, self.gripper.gripper_specs.max_width))
        self.gripper.move(width, speed=self.gripper_speed, force=self.gripper_force).wait()

    def _move_arm_q(self, q_target: np.ndarray):
        self.robot.move_to_joint_configuration(q_target, joint_speed=self.arm_joint_speed).wait()

    def reset(self, trial_idx: Optional[int] = None):
        if trial_idx is not None:
            self.current_trial_idx = int(trial_idx)
        else:
            self.current_trial_idx += 1

        self.episode_done = False
        self.detach_detected = False
        self.pull_started = False
        self.contact_started = False
        self.pull_start_time = None
        self.step_count = 0
        self.last_obs = None
        self.event_rows = []
        self.raspberry_rows = []
        self.loadcell_rows = []
        self.anyskin_rows = []
        self.processor.reset()

        self.move_gripper(self.initial_open_width)
        self._move_arm_q(self.SAFE_Q)
        self.log_event("safe_pose_reached")
        self._move_arm_q(self.APPROACH_Q)
        self.log_event("approach_reached")
        self._move_arm_q(self.GRASP_Q)
        self.log_event("grasp_pose_reached")
        return self.get_observations()

    def get_joint_configuration(self):
        return np.zeros((0,), dtype=np.float32)

    def get_robot_pose_se3(self):
        return self.robot.get_tcp_pose()

    def get_gripper_opening(self):
        return np.array([self.gripper.get_current_width()], dtype=np.float32)

    def _get_raw_snapshot(self):
        with self._lock:
            return {
                "raspberry": list(self._latest_raspberry),
                "anyskin": list(self._latest_anyskin),
                "force": float(self._latest_load_force),
            }

    def get_observations(self):
        raw = self._get_raw_snapshot()
        processed = self.processor.process(
            raw_pressures=raw["raspberry"],
            raw_anyskin=raw["anyskin"],
            raw_force=raw["force"],
        )
        gripper_state = self.get_gripper_opening().astype(np.float32)
        state = np.concatenate([processed["state"], gripper_state], axis=0).astype(np.float32)

        max_rasp = float(np.max(processed["raspberry_state"])) if processed["raspberry_state"].size else 0.0
        max_slip = float(np.max(processed["anyskin_slip"])) if processed["anyskin_slip"].size else 0.0
        if not self.contact_started and max_rasp > max(self.feature_cfg.zero_deadband, 1.0):
            self.contact_started = True
            self.log_event("contact_detected", {"max_raspberry_pressure": max_rasp})
            if self.auto_start_pull_on_contact and not self.pull_started:
                self._move_arm_q(self.PULL_Q)
                self.pull_started = True
                self.pull_start_time = time.perf_counter()
                self.log_event("pull_start")
                self.log_event("pull_reached")

        if processed["detach_detected"] and not self.detach_detected and self.pull_started:
            self.detach_detected = True
            self.episode_done = True
            self.log_event("detach_detected", {"load_force": raw["force"], "max_slip": max_slip})

        obs = {
            "state": state,
            "gripper_state": gripper_state,
            "raspberry_state": processed["raspberry_state"],
            "raspberry_diff": processed["raspberry_diff"],
            "anyskin_mag": processed["anyskin_mag"],
            "anyskin_slip": processed["anyskin_slip"],
            "loadcell_state": processed["loadcell_state"],
            "phase": np.array([
                1.0 if self.contact_started else 0.0,
                1.0 if self.pull_started else 0.0,
                1.0 if self.detach_detected else 0.0,
            ], dtype=np.float32),
        }
        self.last_obs = obs
        return obs

    def act(self, robot_pose_se3, gripper_pose, timestamp):
        if isinstance(gripper_pose, np.ndarray):
            gripper_pose = float(gripper_pose.reshape(-1)[0])
        gripper_width = float(np.clip(gripper_pose, self.gripper.gripper_specs.min_width, self.gripper.gripper_specs.max_width))
        self.gripper._set_target_width(gripper_width)
        self.step_count += 1
        return

    def save_trial(self):
        if self.current_trial_idx <= 0:
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        trial_dir = self.trial_log_root / f"trial_{self.current_trial_idx:03d}_{timestamp}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        rasp_file = trial_dir / "raspberry.csv"
        load_file = trial_dir / "loadcell.csv"
        anyskin_file = trial_dir / "anyskin.csv"
        event_file = trial_dir / "events.csv"

        with open(rasp_file, "w", newline="") as f:
            fieldnames = ["t_pc", "sample_idx"] + [f"S{i}" for i in range(self.feature_cfg.num_raspberry_sensors)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.raspberry_rows)

        with open(load_file, "w", newline="") as f:
            fieldnames = ["t_pc", "sample_idx", "force"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.loadcell_rows)

        with open(anyskin_file, "w", newline="") as f:
            fieldnames = ["t_pc", "t_sensor", "sample_idx"] + self.anyskin_fields
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.anyskin_rows)

        all_event_keys = {k for row in self.event_rows for k in row.keys()}
        preferred = ["t_pc", "event", "trial_idx"]
        remainder = sorted(k for k in all_event_keys if k not in preferred)
        with open(event_file, "w", newline="") as f:
            fieldnames = preferred + remainder
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.event_rows)
        return trial_dir
