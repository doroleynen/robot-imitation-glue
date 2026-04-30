import numpy as np

from robot_imitation_glue.base import BaseAgent


class PIDRaspberryAgent(BaseAgent):
    """
    PID gripper controller for tactile raspberry picking.

    Target is anchored to pressure at pull start, then rises with load force:

        target_pressure = pull_start_pressure + pressure_slope * load_force

    Two-speed closing prevents overshoot: fast until slow_close_threshold, then
    slow until raspberry_contact_threshold. pull_start_pressure is captured at
    the first pull step so error ≈ 0 at pull start regardless of overshoot.

    Phases:
      - Pre-contact (fast)  : close at close_delta until slow_close_threshold
      - Pre-contact (slow)  : close at close_delta_slow until raspberry_contact_threshold
      - Contact, pre-pull   : hold (arm starts pull once anyskin fires)
      - Pull PID            : track pull_start_pressure + pressure_slope * load_force
      - Detach              : stop, reset
    """

    ACTION_SPEC = "GRIPPER_DELTA"

    def __init__(
        self,
        close_delta_pre_contact: float = -0.0004,
        close_delta_slow: float = -0.0005,           # slow close to reduce overshoot near threshold
        slow_close_threshold: float = 500.0,        # Pa; switch to slow close above this
        raspberry_contact_threshold: float = 2000.0, # Pa; stop closing, anchor PID here
        pressure_slope: float = 73.0,                # Pa per gram of load force
        kp: float = 0.0000002,
        ki: float = 0.000000005,
        kd: float = 0.0000002,
        max_close_per_step: float = -0.0001,
        max_open_per_step: float = 0.00000,
        integral_clamp: float = 50.0,
        error_deadband: float = 400.0,
        pressure_aggregation: str = "max",
        min_gripper_width: float = 0.025,
    ):
        self.close_delta_pre_contact = close_delta_pre_contact
        self.close_delta_slow = close_delta_slow
        self.slow_close_threshold = slow_close_threshold
        self.raspberry_contact_threshold = raspberry_contact_threshold
        self.pressure_slope = pressure_slope
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_close_per_step = max_close_per_step
        self.max_open_per_step = max_open_per_step
        self.integral_clamp = integral_clamp
        self.error_deadband = error_deadband
        self.pressure_aggregation = pressure_aggregation
        self.min_gripper_width = min_gripper_width

        self._integral = 0.0
        self._prev_error = 0.0
        self._pull_start_pressure = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._pull_start_pressure = None

    def _aggregate_pressure(self, rasp: np.ndarray) -> float:
        if self.pressure_aggregation == "max":
            return float(np.max(rasp))
        active = rasp[rasp > 0]
        return float(np.mean(active)) if active.size > 0 else 0.0

    def get_action(self, observation: dict) -> np.ndarray:
        phase = observation.get("phase", np.zeros(3, dtype=np.float32))
        rasp = observation.get("raspberry_state", np.zeros(8, dtype=np.float32))
        load = observation.get("loadcell_state", np.zeros(2, dtype=np.float32))

        current_pressure = self._aggregate_pressure(rasp)
        contact_started = current_pressure > self.raspberry_contact_threshold
        pull_started = bool(phase[1] > 0.5)
        detach_detected = bool(phase[2] > 0.5)

        if detach_detected:
            self.reset()
            return np.array([0.0], dtype=np.float32)

        if not contact_started:
            delta = self.close_delta_slow if current_pressure > self.slow_close_threshold else self.close_delta_pre_contact
            return np.array([delta], dtype=np.float32)

        if not pull_started:
            return np.array([0.0], dtype=np.float32)

        # --- pull: PID phase ---
        load_force = float(load[0])

        if self._pull_start_pressure is None:
            self._pull_start_pressure = current_pressure

        target_pressure = self._pull_start_pressure + self.pressure_slope * load_force

        error = target_pressure - current_pressure

        if abs(error) < self.error_deadband:
            self._prev_error = error
            return np.array([0.0], dtype=np.float32)

        self._integral = float(np.clip(
            self._integral + error,
            -self.integral_clamp,
            self.integral_clamp,
        ))
        derivative = error - self._prev_error
        self._prev_error = error

        # Positive error (under-gripping) → close → negative delta
        delta = -(self.kp * error + self.ki * self._integral + self.kd * derivative)
        delta = float(np.clip(delta, self.max_close_per_step, self.max_open_per_step))
        return np.array([delta], dtype=np.float32)
