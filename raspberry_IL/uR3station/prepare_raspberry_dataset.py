import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from raspberry_IL.lerobot_dataset.transform_dataset import transform_dataset


def drop_episodes(dataset_root: str, dataset_name: str, episodes_to_drop: list[int], output_root: str):
    """Write a new dataset to output_root with the specified episodes removed.

    The original dataset is not modified. Episodes in the output are renumbered from 0.

    Args:
        dataset_root: path to the source LeRobot dataset directory
        dataset_name: repo_id of the source dataset
        episodes_to_drop: list of episode indices to exclude
        output_root: path to write the cleaned dataset
    """
    transform_dataset(
        repo_id=dataset_name,
        root_dir=dataset_root,
        new_root_dir=output_root,
        new_repo_id=dataset_name,
        transform_fn=lambda frame: frame,
        episodes_to_drop=episodes_to_drop,
        use_videos=False,
    )
    print(f"Saved cleaned dataset to {output_root} (dropped episodes: {episodes_to_drop})")


def to_npz(dataset_root: str, dataset_name: str, output: str):
    dataset = LeRobotDataset(repo_id=dataset_name, root=Path(dataset_root))
    states = []
    actions = []
    for i in range(len(dataset)):
        row = dataset[i]
        state = row["observation.state"]
        action = row["action"]
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        states.append(state.astype(np.float32))
        actions.append(action.astype(np.float32))

    X = np.stack(states, axis=0)
    y = np.stack(actions, axis=0)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y)
    print(f"Saved {len(X)} samples to {out}")


def to_diffusion_dataset(dataset_root: str, dataset_name: str, output_root: str):
    """Write a training dataset for the diffusion policy.

    Replaces observation.state with the 18-dim observation.state_policy (rasp + loadcell)
    and drops observation.state_policy so the policy only sees the 18-dim state.
    """
    def transform_fn(frame):
        frame["observation.state"] = frame.pop("observation.state_policy")
        return frame

    def transform_features_fn(features):
        features["observation.state"] = features.pop("observation.state_policy")
        return features

    transform_dataset(
        repo_id=dataset_name,
        root_dir=dataset_root,
        new_root_dir=output_root,
        new_repo_id=dataset_name,
        transform_fn=transform_fn,
        transform_features_fn=transform_features_fn,
        features_to_drop=["gripper_state", "raspberry_state", "raspberry_diff",
                          "anyskin_mag", "anyskin_slip", "loadcell_state", "phase"],
        use_videos=False,
    )
    print(f"Saved diffusion training dataset to {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets/raspberry_pick_pid")
    parser.add_argument("--dataset-name", default="raspberry_pick_pid")
    parser.add_argument("--output", default="training_data/raspberry_bc_dataset.npz")
    parser.add_argument("--drop-episodes", default=None, help="Episodes to drop before converting, e.g. '3,7,10-15'")
    parser.add_argument("--output-root", default=None, help="Where to save the cleaned dataset (required with --drop-episodes)")
    parser.add_argument("--to-diffusion-dataset", action="store_true", help="Prepare dataset for diffusion policy training")
    args = parser.parse_args()

    if args.drop_episodes is not None:
        if args.output_root is None:
            parser.error("--output-root is required when using --drop-episodes")
        episodes = []
        for part in args.drop_episodes.split(","):
            if "-" in part:
                start, end = part.split("-")
                episodes.extend(range(int(start), int(end) + 1))
            else:
                episodes.append(int(part))
        drop_episodes(args.dataset_root, args.dataset_name, episodes, args.output_root)
    elif args.to_diffusion_dataset:
        if args.output_root is None:
            parser.error("--output-root is required when using --to-diffusion-dataset")
        to_diffusion_dataset(args.dataset_root, args.dataset_name, args.output_root)
    else:
        to_npz(args.dataset_root, args.dataset_name, args.output)


if __name__ == "__main__":
    main()
