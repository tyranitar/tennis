from nn_utils import init_default, get_fc_layers, assert_valid_conv_layer_dims, get_conv_layers
from contextlib import contextmanager
from torch import nn
import torch

# States: 3 x 6 = 18
# Actions: 2

class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, seed=1337):
        super(ActorNet, self).__init__()
        torch.manual_seed(seed)

        # (out_channels, kernel_size, stride)
        conv_layer_dims = [
            (8, 1, 1),
        ]

        conv_output_dim = assert_valid_conv_layer_dims((1, 6), conv_layer_dims)

        self.conv = nn.Sequential(
            # Treat stack length as channels, as in DQN.
            # This is technically 1D conv but 2D conv
            # generalizes to both, so just use 2D conv.
            *get_conv_layers(state_size[0], conv_layer_dims)
        )

        fc_layer_dims = [
            conv_output_dim,
            400,
            300,
            # action_size
        ]

        self.fc = nn.Sequential(
            *get_fc_layers(fc_layer_dims),
            init_default(nn.Linear(fc_layer_dims[-1], action_size), w_scale=1e-3),
            nn.Tanh(),
        )

    def forward(self, states):
        x = states

        # NOTE: Turn stack of 1D signals
        # into a stack of 2D "images".
        x = x.unsqueeze(-2)
        x = self.conv(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

    @contextmanager
    def eval_no_grad(self):
        with torch.no_grad():
            try:
                self.eval()
                yield
            finally:
                self.train()

class CriticNet(nn.Module):
    def __init__(self, state_size, action_size, n_atoms, seed=1337):
        super(CriticNet, self).__init__()
        torch.manual_seed(seed)

        # (out_channels, kernel_size, stride)
        conv_layer_dims = [
            (8, 1, 1),
        ]

        conv_output_dim = assert_valid_conv_layer_dims((1, 6), conv_layer_dims)

        self.conv = nn.Sequential(
            # Treat stack length as channels, as in DQN.
            # This is technically 1D conv but 2D conv
            # generalizes to both, so just use 2D conv.
            *get_conv_layers(state_size[0], conv_layer_dims)
        )

        fc_layer_dims = [
            conv_output_dim + action_size,
            400,
            300,
            # 1
        ]

        self.fc = nn.Sequential(
            *get_fc_layers(fc_layer_dims),
            init_default(nn.Linear(fc_layer_dims[-1], n_atoms)),
            nn.Softmax(dim=-1),
        )

    def forward(self, states, actions):
        a = actions
        x = states

        # NOTE: Turn stack of 1D signals
        # into a stack of 2D "images".
        x = x.unsqueeze(-2)
        x = self.conv(x)

        x = x.view(x.shape[0], -1)
        x = torch.cat([x, a], dim=-1)
        x = self.fc(x)

        return x
