import argparse
import time
from pathlib import Path

import numpy as np

from raspberry_IL.agents.bc_raspberry_agent import BCRaspberryAgent
from raspberry_IL.agents.heuristic_raspberry_agent import HeuristicRaspberryAgent
from raspberry_IL.agents.pid_raspberry_agent import PIDRaspberryAgent
from raspberry_IL.dataset_recorder import LeRobotDatasetRecorder
from raspberry_IL.uR3station.raspberry_pick_env import RaspberryPickEnv
from raspberry_IL.uR3station.raspberry_trial_utils import OnlineFeatureConfig


def apply_delta_to_commanded(commanded_width: float, action: np.ndarray, min_width: float, max_width: float):
    target = float(np.clip(commanded_width + action[0], min_width, max_width))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--mode", choices=["heuristic", "bc", "pid"], default="heuristic")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trial-log-dir", default=None)
    parser.add_argument("--no-anyskin", action="store_true", help="Disable AnySkin sensor (run without it plugged in)")
    args = parser.parse_args()

    if args.dataset_name is None:
        args.dataset_name = f"raspberry_pick_{args.mode}"
    if args.dataset_root is None:
        args.dataset_root = f"datasets/raspberry_pick_{args.mode}"
    if args.trial_log_dir is None:
        args.trial_log_dir = f"trial_logs_{args.mode}"

    enable_anyskin = not args.no_anyskin
    if args.mode == "heuristic":
        agent = HeuristicRaspberryAgent()
        env = RaspberryPickEnv(trial_log_root=args.trial_log_dir, fps=args.fps, enable_anyskin=enable_anyskin)
    elif args.mode == "pid":
        agent = PIDRaspberryAgent()
        feature_cfg = OnlineFeatureConfig(raspberry_contact_threshold=agent.raspberry_contact_threshold)
        env = RaspberryPickEnv(trial_log_root=args.trial_log_dir, fps=args.fps, feature_cfg=feature_cfg, enable_anyskin=enable_anyskin)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in bc mode")
        agent = BCRaspberryAgent(args.checkpoint)
        env = RaspberryPickEnv(trial_log_root=args.trial_log_dir, fps=args.fps, enable_anyskin=enable_anyskin)

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
            if hasattr(agent, "reset"):
                agent.reset()
            dataset_recorder.start_episode()
            commanded_width = float(env.gripper.gripper_specs.max_width)

            for _ in range(args.max_steps):
                cycle_end = time.time() + period
                obs = env.get_observations()
                action = agent.get_action(obs).astype(np.float32)
                commanded_width = apply_delta_to_commanded(
                    commanded_width,
                    action,
                    env.gripper.gripper_specs.min_width,
                    env.gripper.gripper_specs.max_width,
                )
                env.act(env.get_robot_pose_se3(), np.array([commanded_width], dtype=np.float32), timestamp=cycle_end)
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
