import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets/raspberry_pick")
    parser.add_argument("--dataset-name", default="raspberry_pick")
    parser.add_argument("--output", default="training_data/raspberry_bc_dataset.npz")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.dataset_name, root=Path(args.dataset_root))
    states = []
    actions = []
    for i in range(len(dataset)):
        row = dataset[i]
        state = row["state"]
        action = row["action"]
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        states.append(state.astype(np.float32))
        actions.append(action.astype(np.float32))

    X = np.stack(states, axis=0)
    y = np.stack(actions, axis=0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y)
    print(f"Saved {len(X)} samples to {out}")


if __name__ == "__main__":
    main()
