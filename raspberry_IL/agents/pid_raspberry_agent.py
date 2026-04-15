import numpy as np

from robot_imitation_glue.base import BaseAgent


class PIDRaspberryAgent(BaseAgent):
    """
    PID gripper controller for tactile raspberry picking.

    Setpoint: target raspberry pressure tracks the current pull force via a
    linear relationship fitted from human demonstrations:

        target_pressure = pressure_slope * load_force + pressure_intercept

    The PID error is (target - measured). Positive error (under-gripping) →
    close gripper (negative delta). Only raspberry_state and loadcell_state
    drive the controller; no gripper width feedback is used.

    Phases:
      - Pre-contact  : close gently until anyskin contact is detected (phase[0])
      - Contact hold : hold width while the env initiates the pull (phase[1])
      - Pull         : PID active
      - Detach       : stop, reset integrator
    """

    ACTION_SPEC = "GRIPPER_DELTA"

    def __init__(
        self,
        # Pre-contact approach speed
        close_delta_pre_contact: float = -0.001,
        # Setpoint: target_pressure = pressure_slope * load_force + pressure_intercept
        # Tune these from your human demo load-cell vs pressure analysis.
        pressure_slope: float = 200.0,       # pressure units per Newton of pull force
        pressure_intercept: float = 20.0,    # minimum target pressure at the start of pull
        # PID gains — all positive; sign is handled internally (positive error → close)
        kp: float = 0.00005,
        ki: float = 0.000002,
        kd: float = 0.00001,
        # Output limits (metres per step)
        max_close_per_step: float = -0.002,
        max_open_per_step: float = 0.0005,   # small release allowed to avoid over-gripping
        # Anti-windup: hard clamp on the accumulated error
        integral_clamp: float = 500.0,
        # Pressure aggregation: "mean_active" uses mean of sensors > 0, "max" uses the peak
        pressure_aggregation: str = "mean_active",
    ):
        self.close_delta_pre_contact = close_delta_pre_contact
        self.pressure_slope = pressure_slope
        self.pressure_intercept = pressure_intercept
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_close_per_step = max_close_per_step
        self.max_open_per_step = max_open_per_step
        self.integral_clamp = integral_clamp
        self.pressure_aggregation = pressure_aggregation

        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def _aggregate_pressure(self, rasp: np.ndarray) -> float:
        if self.pressure_aggregation == "max":
            return float(np.max(rasp))
        # default: mean of active (positive) sensors only
        active = rasp[rasp > 0]
        return float(np.mean(active)) if active.size > 0 else 0.0

    def get_action(self, observation: dict) -> np.ndarray:
        phase = observation.get("phase", np.zeros(3, dtype=np.float32))
        rasp = observation.get("raspberry_state", np.zeros(8, dtype=np.float32))
        load = observation.get("loadcell_state", np.zeros(2, dtype=np.float32))

        contact_started = bool(phase[0] > 0.5)
        pull_started = bool(phase[1] > 0.5)
        detach_detected = bool(phase[2] > 0.5)

        if detach_detected:
            self.reset()
            return np.array([0.0], dtype=np.float32)

        if not contact_started:
            return np.array([self.close_delta_pre_contact], dtype=np.float32)

        if not pull_started:
            return np.array([0.0], dtype=np.float32)

        # --- PID during pull ---
        load_force = float(load[0])
        current_pressure = self._aggregate_pressure(rasp)
        target_pressure = self.pressure_slope * load_force + self.pressure_intercept

        error = target_pressure - current_pressure

        self._integral = float(np.clip(
            self._integral + error,
            -self.integral_clamp,
            self.integral_clamp,
        ))
        derivative = error - self._prev_error
        self._prev_error = error

        # Positive error (need more grip) → close → negative delta
        delta = -(self.kp * error + self.ki * self._integral + self.kd * derivative)
        delta = float(np.clip(delta, self.max_close_per_step, self.max_open_per_step))
        return np.array([delta], dtype=np.float32)
