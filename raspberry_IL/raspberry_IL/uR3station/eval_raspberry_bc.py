import argparse

from robot_imitation_glue.agents.bc_raspberry_agent import BCRaspberryAgent
from robot_imitation_glue.uR3station.collect_raspberry_data import delta_to_abs_gripper
from robot_imitation_glue.uR3station.raspberry_pick_env import RaspberryPickEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--trial-log-dir", default="trial_logs_policy_eval")
    args = parser.parse_args()

    env = RaspberryPickEnv(trial_log_root=args.trial_log_dir)
    agent = BCRaspberryAgent(args.checkpoint)
    period = 1.0 / args.fps

    try:
        for episode_idx in range(args.episodes):
            obs = env.reset(trial_idx=episode_idx + 1)
            for _ in range(args.max_steps):
                obs = env.get_observations()
                action = agent.get_action(obs)
                new_gripper = delta_to_abs_gripper(
                    obs["gripper_state"],
                    action,
                    env.gripper.gripper_specs.min_width,
                    env.gripper.gripper_specs.max_width,
                )
                env.act(env.get_robot_pose_se3(), new_gripper, 0.0)
                if env.episode_done:
                    break
            trial_dir = env.save_trial()
            print(f"Saved eval trial to: {trial_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
