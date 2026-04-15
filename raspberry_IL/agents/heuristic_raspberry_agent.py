# import numpy as np

# from robot_imitation_glue.base import BaseAgent


# class HeuristicRaspberryAgent(BaseAgent):
#     ACTION_SPEC = "GRIPPER_DELTA"

#     # def __init__(self, close_delta: float = -0.0025, slip_gain: float = -0.001, pressure_limit: float = 250.0):
#     #     self.close_delta = close_delta
#     #     self.slip_gain = slip_gain
#     #     self.pressure_limit = pressure_limit

#     # def get_action(self, observation):
#     #     rasp = observation.get("raspberry_state", np.zeros((8,), dtype=np.float32))
#     #     slip = observation.get("anyskin_slip", np.zeros((5,), dtype=np.float32))
#     #     phase = observation.get("phase", np.zeros((3,), dtype=np.float32))

#     #     if phase[2] > 0.5:
#     #         return np.array([0.0], dtype=np.float32)

#     #     action = self.close_delta
#     #     max_slip = float(np.max(slip)) if len(slip) else 0.0
#     #     max_rasp = float(np.max(rasp)) if len(rasp) else 0.0

#     #     action += self.slip_gain * max_slip
#     #     if max_rasp > self.pressure_limit:
#     #         action = 0.0

#     #     return np.array([action], dtype=np.float32)



#     def __init__(
#         self,
#         close_delta: float = -0.0015,
#         slip_close_gain: float = -0.0008,
#         max_close_per_step: float = -0.0030,
#     ):
#         self.close_delta = close_delta
#         self.slip_close_gain = slip_close_gain
#         self.max_close_per_step = max_close_per_step

#     def get_action(self, observation):
#         slip = observation.get("anyskin_slip", np.zeros((5,), dtype=np.float32))
#         phase = observation.get("phase", np.zeros((3,), dtype=np.float32))

#         pull_started = bool(phase[1] > 0.5)
#         detach_detected = bool(phase[2] > 0.5)

#         # Once we are pulling, hold width.
#         if pull_started or detach_detected:
#             return np.array([0.0], dtype=np.float32)

#         max_slip = float(np.max(slip)) if len(slip) else 0.0

#         # Negative delta = close a bit more.
#         action = self.close_delta + self.slip_close_gain * max_slip

#         # Clamp so we never close too aggressively in one step.
#         action = max(action, self.max_close_per_step)

#         return np.array([action], dtype=np.float32)


import numpy as np

from robot_imitation_glue.base import BaseAgent


class HeuristicRaspberryAgent(BaseAgent):
    ACTION_SPEC = "GRIPPER_DELTA"

    def __init__(
        self,
        close_delta_pre_contact: float = -0.001,
        slip_threshold: float = 10,
        slip_close_step: float = -0.001,
        max_close_per_step: float = -0.0030,
        min_gripper_width: float = 0.025,  # safety: never close below this regardless of sensor state
    ):
        self.close_delta_pre_contact = close_delta_pre_contact
        self.slip_threshold = slip_threshold
        self.slip_close_step = slip_close_step
        self.max_close_per_step = max_close_per_step
        self.min_gripper_width = min_gripper_width

    def get_action(self, observation):
        slip = observation.get("anyskin_slip", np.zeros((5,), dtype=np.float32))
        phase = observation.get("phase", np.zeros((3,), dtype=np.float32))
        gripper_width = float(observation.get("gripper_state", np.array([1.0]))[0])

        contact_started = bool(phase[0] > 0.5)
        pull_started = bool(phase[1] > 0.5)
        detach_detected = bool(phase[2] > 0.5)

        if detach_detected:
            return np.array([0.0], dtype=np.float32)

        # Safety: never close below minimum width regardless of sensor state
        if gripper_width <= self.min_gripper_width:
            return np.array([0.0], dtype=np.float32)

        max_slip = float(np.max(slip)) if len(slip) else 0.0

        # Before contact: gently close
        if not contact_started:
            action = self.close_delta_pre_contact
        # Contact detected, pull not started yet: hold — pull starts next step
        elif not pull_started:
            action = 0.0
        # During pull: only tighten if slip is detected
        elif pull_started:
            if max_slip > self.slip_threshold:
                action = self.slip_close_step
            else:
                action = 0.0
        else:
            action = 0.0

        action = max(action, self.max_close_per_step)
        return np.array([action], dtype=np.float32)