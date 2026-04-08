import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training_data/raspberry_bc_dataset.npz")
    parser.add_argument("--output", default="outputs/raspberry_bc.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data = np.load(args.dataset)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    state_mean = X.mean(axis=0)
    state_std = X.std(axis=0) + 1e-6
    Xn = (X - state_mean) / state_std

    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y))
    val_size = max(1, int(0.2 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MLP(X.shape[1], args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_ds)
        print(f"epoch {epoch+1:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": X.shape[1],
            "hidden_dim": args.hidden_dim,
            "model_state_dict": best_state,
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
        },
        output,
    )
    print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
