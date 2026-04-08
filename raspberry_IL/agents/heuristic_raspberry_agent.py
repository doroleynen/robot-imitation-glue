import numpy as np

from robot_imitation_glue.base import BaseAgent


class HeuristicRaspberryAgent(BaseAgent):
    ACTION_SPEC = "GRIPPER_DELTA"

    def __init__(self, close_delta: float = -0.0025, slip_gain: float = -0.0015, pressure_limit: float = 250.0):
        self.close_delta = close_delta
        self.slip_gain = slip_gain
        self.pressure_limit = pressure_limit

    def get_action(self, observation):
        rasp = observation.get("raspberry_state", np.zeros((8,), dtype=np.float32))
        slip = observation.get("anyskin_slip", np.zeros((5,), dtype=np.float32))
        phase = observation.get("phase", np.zeros((3,), dtype=np.float32))

        if phase[2] > 0.5:
            return np.array([0.0], dtype=np.float32)

        action = self.close_delta
        max_slip = float(np.max(slip)) if len(slip) else 0.0
        max_rasp = float(np.max(rasp)) if len(rasp) else 0.0

        action += self.slip_gain * max_slip
        if max_rasp > self.pressure_limit:
            action = max(action, 0.001)

        return np.array([action], dtype=np.float32)
