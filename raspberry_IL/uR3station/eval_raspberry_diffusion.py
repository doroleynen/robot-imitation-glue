import argparse
import time

import numpy as np
import torch

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from raspberry_IL.uR3station.collect_raspberry_data import apply_delta_to_commanded
from raspberry_IL.uR3station.raspberry_pick_env import RaspberryPickEnv
from raspberry_IL.uR3station.raspberry_trial_utils import OnlineFeatureConfig


def make_obs_preprocessor(device, include_joints=False, student_obs=False):
    def preprocess(obs):
        if student_obs:
            # anyskin_mag(2) + anyskin_slip(2) + gripper_state(1) + phase(3) [+ joints(6)] = 8 or 14-dim
            parts = [obs["anyskin_mag"], obs["anyskin_slip"], obs["gripper_state"], obs["phase"]]
            if include_joints:
                parts.append(obs["joint_configuration"])
            state = np.concatenate(parts).astype(np.float32)
        else:
            state = obs["observation.state_policy"]
            if include_joints:
                state = np.concatenate([state, obs["joint_configuration"]])
        return {
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(device),
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
    parser.add_argument("--include-joints", action="store_true",
                        help="Concatenate joint_configuration into observation.state (use with joints-trained model)")
    parser.add_argument("--student-obs", action="store_true",
                        help="Use external-only student obs: anyskin_mag+slip+gripper_state+phase (no raspberry/loadcell)")
    parser.add_argument("--raspberry-contact-threshold", type=float, default=2000.0,
                        help="Max raspberry pressure to trigger pull, same as PID agent (default 2000)")
    args = parser.parse_args()

    device = args.device
    policy = make_lerobot_policy(args.checkpoint, args.dataset_root)
    policy = policy.to(device)
    agent = LerobotAgent(policy, device, make_obs_preprocessor(device, include_joints=args.include_joints, student_obs=args.student_obs))

    feature_cfg = OnlineFeatureConfig(raspberry_contact_threshold=args.raspberry_contact_threshold)
    env = RaspberryPickEnv(trial_log_root=args.trial_log_dir, fps=args.fps, feature_cfg=feature_cfg,
                           record_joints=args.include_joints)
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
                print(f"[dbg] phase={obs['phase']}  action={action[0]:.5f}  gripper={float(obs['gripper_state'][0]):.4f}m")
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
