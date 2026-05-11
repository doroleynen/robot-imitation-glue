import argparse
import time

import numpy as np
import torch

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from raspberry_IL.uR3station.collect_raspberry_data import apply_delta_to_commanded
from raspberry_IL.uR3station.raspberry_pick_env import RaspberryPickEnv


def make_obs_preprocessor(device):
    def preprocess(obs):
        return {
            "observation.state": torch.from_numpy(obs["observation.state_policy"]).float().unsqueeze(0).to(device),
            "observation.environment_state": torch.from_numpy(obs["observation.environment_state"]).float().unsqueeze(0).to(device),
        }
    return preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_model directory")
    parser.add_argument("--dataset-root", required=True, help="Path to the training dataset (needed for policy metadata)")
    parser.add_argument("--dataset-name", default="raspberry_pick_pid")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--trial-log-dir", default="trial_logs_diffusion_eval")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    policy = make_lerobot_policy(args.checkpoint, args.dataset_root)
    policy = policy.to(device)
    agent = LerobotAgent(policy, device, make_obs_preprocessor(device))

    env = RaspberryPickEnv(trial_log_root=args.trial_log_dir, fps=args.fps)
    period = 1.0 / args.fps

    try:
        for episode_idx in range(args.episodes):
            print(f"\n=== Episode {episode_idx + 1}/{args.episodes} ===")
            obs = env.reset(trial_idx=episode_idx + 1)
            agent.reset()
            commanded_width = float(env.gripper.gripper_specs.max_width)

            for _ in range(args.max_steps):
                cycle_end = time.time() + period
                obs = env.get_observations()
                action = agent.get_action(obs)
                commanded_width = apply_delta_to_commanded(
                    commanded_width,
                    action,
                    env.gripper.gripper_specs.min_width,
                    env.gripper.gripper_specs.max_width,
                )
                env.act(env.get_robot_pose_se3(), np.array([commanded_width], dtype=np.float32), timestamp=cycle_end)
                if env.episode_done:
                    break
                wait = cycle_end - time.time()
                if wait > 0:
                    time.sleep(wait)

            trial_dir = env.save_trial()
            print(f"Saved eval trial to: {trial_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
