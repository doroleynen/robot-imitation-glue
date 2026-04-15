import argparse
import time
from pathlib import Path

import numpy as np

from raspberry_IL.agents.bc_raspberry_agent import BCRaspberryAgent
from raspberry_IL.agents.heuristic_raspberry_agent import HeuristicRaspberryAgent
from raspberry_IL.agents.pid_raspberry_agent import PIDRaspberryAgent
from raspberry_IL.dataset_recorder import LeRobotDatasetRecorder
from raspberry_IL.uR3station.raspberry_pick_env import RaspberryPickEnv


def delta_to_abs_gripper(current_width: np.ndarray, action: np.ndarray, min_width: float, max_width: float):
    target = float(current_width[0] + action[0])
    target = float(np.clip(target, min_width, max_width))
    return np.array([target], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="raspberry_pick")
    parser.add_argument("--dataset-root", default="datasets/raspberry_pick")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--mode", choices=["heuristic", "bc", "pid"], default="heuristic")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trial-log-dir", default="trial_logs_policy")
    args = parser.parse_args()

    env = RaspberryPickEnv(trial_log_root=args.trial_log_dir)
    if args.mode == "heuristic":
        agent = HeuristicRaspberryAgent()
    elif args.mode == "pid":
        agent = PIDRaspberryAgent()
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in bc mode")
        agent = BCRaspberryAgent(args.checkpoint)

    try:
        example_obs = env.reset(trial_idx=1)
        dataset_recorder = LeRobotDatasetRecorder(
            example_obs_dict=example_obs,
            example_action=np.zeros((1,), dtype=np.float32),
            root_dataset_dir=Path(args.dataset_root),
            dataset_name=args.dataset_name,
            fps=args.fps,
            use_videos=False,
        )

        period = 1.0 / args.fps
        for episode_idx in range(args.episodes):
            print(f"\n=== Episode {episode_idx + 1}/{args.episodes} ===")
            obs = env.reset(trial_idx=episode_idx + 1)
            dataset_recorder.start_episode()

            for _ in range(args.max_steps):
                cycle_end = time.time() + period
                obs = env.get_observations()
                action = agent.get_action(obs).astype(np.float32)
                new_gripper = delta_to_abs_gripper(
                    obs["gripper_state"],
                    action,
                    env.gripper.gripper_specs.min_width,
                    env.gripper.gripper_specs.max_width,
                )
                env.act(env.get_robot_pose_se3(), new_gripper, timestamp=cycle_end)
                dataset_recorder.record_step(obs, action)
                if env.episode_done:
                    break
                wait = cycle_end - time.time()
                if wait > 0:
                    time.sleep(wait)

            dataset_recorder.save_episode()
            trial_dir = env.save_trial()
            print(f"Saved raw trial to: {trial_dir}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
