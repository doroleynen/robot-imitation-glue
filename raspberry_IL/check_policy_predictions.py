"""
Offline sanity-check for trained diffusion policies.

Loops over all checkpoints found under --train-dir and saves one PNG per checkpoint.

Usage:
    python raspberry_IL/check_policy_predictions.py \
        --train-dir outputs/train \
        --dataset-root datasets/raspberry_pid_diffusion \
        --device cuda
"""

import argparse
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_imitation_glue.agents.lerobot_agent import make_lerobot_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def run_policy_on_episode(policy, episode_rows, device, n_obs_steps):
    obs_queue = {
        "observation.state": deque(maxlen=n_obs_steps),
        "observation.environment_state": deque(maxlen=n_obs_steps),
    }
    action_queue = deque()
    predicted, gt = [], []

    policy.reset()

    for row in episode_rows:
        state = torch.tensor(row["observation.state"], dtype=torch.float32)
        env_state = torch.tensor(row["observation.environment_state"], dtype=torch.float32)
        obs_queue["observation.state"].append(state)
        obs_queue["observation.environment_state"].append(env_state)

        if len(obs_queue["observation.state"]) < n_obs_steps:
            continue

        if len(action_queue) == 0:
            batch = {
                k: torch.stack(list(v)).unsqueeze(0).to(device)
                for k, v in obs_queue.items()
            }
            with torch.no_grad():
                batch = policy.normalize_inputs(batch)
                actions = policy.diffusion.generate_actions(batch)
                actions = policy.unnormalize_outputs({"action": actions})["action"]
            action_queue.extend(actions[0].cpu().numpy())

        pred_action = action_queue.popleft()
        predicted.append(float(np.array(pred_action).flat[0]))
        gt.append(float(np.array(row["action"]).flat[0]))

    return np.array(predicted), np.array(gt)


def find_checkpoints(train_dir: Path):
    """Return list of (label, pretrained_model_path) sorted by run then step."""
    checkpoints = []
    for run_dir in sorted(train_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt_root = run_dir / "checkpoints"
        if not ckpt_root.exists():
            continue
        for step_dir in sorted(ckpt_root.iterdir()):
            if step_dir.name == "last":
                continue
            model_dir = step_dir / "pretrained_model"
            if model_dir.exists():
                label = f"{run_dir.name}/{step_dir.name}"
                checkpoints.append((label, model_dir))
    return checkpoints


def run_checkpoint(checkpoint_path, dataset, hf, episode_indices, unique_eps, device):
    policy = make_lerobot_policy(str(checkpoint_path), str(dataset.root))
    policy = policy.to(device).eval()
    policy.diffusion.num_inference_steps = 10
    n_obs_steps = policy.config.n_obs_steps

    state_dim = policy.config.input_features["observation.state"].shape[0]

    all_predicted, all_gt = [], []
    fig, axes = plt.subplots(len(unique_eps), 1,
                             figsize=(12, 3 * len(unique_eps)), squeeze=False)

    for plot_idx, ep_idx in enumerate(unique_eps):
        mask = episode_indices == ep_idx
        rows = hf.select(np.where(mask)[0].tolist())
        states = np.array(rows["observation.state"], dtype=np.float32).reshape(-1, state_dim)
        env_states = np.array(rows["observation.environment_state"], dtype=np.float32).reshape(-1, 1)
        actions = np.array(rows["action"], dtype=np.float32).reshape(-1, 1)
        episode_rows = [
            {
                "observation.state": states[i],
                "observation.environment_state": env_states[i],
                "action": actions[i],
            }
            for i in range(len(rows))
        ]

        predicted, gt = run_policy_on_episode(policy, episode_rows, device, n_obs_steps)
        all_predicted.append(predicted)
        all_gt.append(gt)

        ax = axes[plot_idx][0]
        ax.plot(gt, label="ground truth", alpha=0.7)
        ax.plot(predicted, label="predicted", alpha=0.7)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title(f"Episode {ep_idx}")
        ax.set_ylabel("gripper delta")
        ax.legend(fontsize=8)

    axes[-1][0].set_xlabel("step")
    plt.tight_layout()

    all_pred = np.concatenate(all_predicted)
    all_gt_flat = np.concatenate(all_gt)
    mae = np.abs(all_pred - all_gt_flat).mean()

    return fig, all_pred, all_gt_flat, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="outputs/train",
                        help="Root directory containing all training run folders")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", default="raspberry_pick_pid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--out-dir", default="prediction_checks",
                        help="Directory to save PNGs into")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.dataset_name, root=args.dataset_root)
    hf = dataset.hf_dataset
    episode_indices = np.array(hf["episode_index"]).flatten()
    unique_eps = np.unique(episode_indices)[: args.max_episodes]

    checkpoints = find_checkpoints(Path(args.train_dir))
    if not checkpoints:
        print(f"No checkpoints found under {args.train_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints. Saving PNGs to {out_dir}/\n")

    for label, ckpt_path in checkpoints:
        print(f"Running {label} ...")
        try:
            fig, all_pred, all_gt, mae = run_checkpoint(
                ckpt_path, dataset, hf, episode_indices, unique_eps, args.device
            )
            safe_label = label.replace("/", "_")
            png_path = out_dir / f"{safe_label}.png"
            fig.suptitle(f"{label}  |  MAE: {mae:.4f}", fontsize=10)
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"  pred  min={all_pred.min():.4f} max={all_pred.max():.4f} mean={all_pred.mean():.4f}")
            print(f"  gt    min={all_gt.min():.4f}   max={all_gt.max():.4f}   mean={all_gt.mean():.4f}")
            print(f"  MAE={mae:.4f}  → saved {png_path}\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")


if __name__ == "__main__":
    main()
