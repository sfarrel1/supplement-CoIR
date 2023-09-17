'''
This program creates MLP based networks to learn
an implicit representation of a scene. ReLU and Sine
(SIREN) nonlinearity based MLPs can be created.
'''

import numpy as np
import torch
from torch import nn

class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU nonlinearity
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()


    def forward(self, input):
        return self.relu(self.linear(input))


class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.

        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.

        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class FourierFeaturesVector(torch.nn.Module):
    def __init__(self, num_input_channels, mapping_size=128, scale=10.0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.out_dim = mapping_size*2

    def forward(self, x):
        assert x.dim() == 3

        b, l, c = x.shape

        assert c == self._num_input_channels, "number channels wrong"

        # From [B, C, W, H] to [(B*W*H), C]
        x = x @ self._B.to(x.device)

        x = 2 * np.pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, nonlinearity='relu', outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30, scale=10.0,
                 ff_encode=False, mapping_size=128):
        super().__init__()
        self.ff_encode = ff_encode
        if nonlinearity == 'sine':
            self.nonlin = SineLayer
        elif nonlinearity == 'relu':
            self.nonlin = ReLULayer
        else:
            raise ValueError('Nonlinearity not known')

        if ff_encode:
            self.fourier_encoding = FourierFeaturesVector(num_input_channels=in_features,
                                                          mapping_size=mapping_size, scale=scale)
            in_features = self.fourier_encoding.out_dim

        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features,
                                    is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features,
                                        is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features, bias=True, dtype=torch.float)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features,
                                        is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.ff_encode:
            coords = self.fourier_encoding(coords)
        output = self.net(coords)

        return output

if __name__ == "__main__":
    device = 'cuda'
    nonlin = 'sine' #'relu' or 'sine'
    omega0 = 30
    ffenc = True
    map_feat = 128
    alpha = 10

    # instantiate the NN - specify hyperparameters
    model = INR(in_features=2, hidden_features=256, hidden_layers=3,
                       out_features=2, nonlinearity=nonlin, outermost_linear=True,
                       first_omega_0=omega0, hidden_omega_0=omega0, scale=alpha,
                       ff_encode=ffenc, mapping_size=map_feat).to(device)
    print('Number of parameters: ', sum(param.numel() for param in model.parameters()))

    # Build the input query coordinates
    W = 256
    H = 256
    x = torch.linspace(-1, 1, W).cuda()
    y = torch.linspace(-1, 1, H).cuda()

    X, Y = torch.meshgrid(x, y, indexing='xy')

    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

    y = model(coords)
    print(y.shape)
    print(model)