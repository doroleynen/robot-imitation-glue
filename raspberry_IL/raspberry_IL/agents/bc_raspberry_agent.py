from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robot_imitation_glue.base import BaseAgent


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


class BCRaspberryAgent(BaseAgent):
    ACTION_SPEC = "GRIPPER_DELTA"

    def __init__(self, checkpoint_path: str):
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.model = MLP(checkpoint["input_dim"], checkpoint.get("hidden_dim", 128))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.state_mean = checkpoint["state_mean"]
        self.state_std = checkpoint["state_std"]

    def get_action(self, observation):
        state = observation["state"].astype(np.float32)
        state_norm = (state - self.state_mean) / np.maximum(self.state_std, 1e-6)
        with torch.no_grad():
            action = self.model(torch.from_numpy(state_norm).unsqueeze(0)).squeeze(0).numpy()
        return action.astype(np.float32)
