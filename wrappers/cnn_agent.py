"""EPyMARL agent for the grid observation (``obs.mode = "grid"``): CNN over the image planes, then GRU.

Select it with ``agent=wx_cnn``. It expects ``GRID_PLANES * G * G`` image inputs followed by scalars.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import load_config
from core.observation import GRID_PLANES as OBS_PLANES


class CNNAgent(nn.Module):
    """Same interface as EPyMARL's ``RNNAgent``; ``inputs`` is [batch, obs_dim + extras].

    The first ``OBS_PLANES * G * G`` inputs are the image; the rest (own scalars and whatever
    EPyMARL appends, e.g. the agent id) join the image features before the recurrent layer.
    """

    def __init__(self, input_shape: int, args) -> None:
        super().__init__()
        self.args = args
        self.G = load_config(args.env_args["map_name"]).obs.grid_n
        n_image = OBS_PLANES * self.G**2
        self.conv = nn.Sequential(
            nn.Conv2d(OBS_PLANES, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )  # fmt: skip
        n_conv = 32 * (self.G // 8) ** 2
        self.fc1 = nn.Linear(n_conv + input_shape - n_image, args.hidden_dim)
        if args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self) -> torch.Tensor:
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_image = OBS_PLANES * self.G**2
        image = inputs[:, :n_image].view(-1, OBS_PLANES, self.G, self.G)
        x = F.relu(self.fc1(torch.cat((self.conv(image), inputs[:, n_image:]), dim=1)))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in) if self.args.use_rnn else F.relu(self.rnn(x))
        return self.fc2(h), h
