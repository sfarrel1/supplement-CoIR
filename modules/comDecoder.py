'''
This program creates an instantiation of ComDecoder
with an NN architecture I am proposing for sparse radar
imaging in the paper CoIR: Compressive Implicit Radar.

created by: Sean Farrell
date: 3/21/23
'''

import torch.nn as nn
import numpy as np


# Residual block original
class ResidualBlockOrig(nn.Module):
    def __init__(self, in_f, out_f, kernel_size):
        super(ResidualBlockOrig, self).__init__()
        self.conv1 = nn.Conv2d(in_f, out_f, kernel_size, (1, 1), padding=(kernel_size - 1) // 2, bias=True,
                               padding_mode="zeros")
        self.conv2 = nn.Conv2d(out_f, out_f, kernel_size, (1, 1), padding=(kernel_size - 1) // 2, bias=True,
                               padding_mode="zeros")
        self.bn = nn.BatchNorm2d(out_f, affine=True)
        self.silu = nn.SiLU()

    def forward(self, x):
        # original resnet paper
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn(out)

        out += residual
        out = self.silu(out)

        return out


class com_decoder(nn.Module):
    def __init__(self, num_layers, num_input_channels, num_channels, num_output_channels,
                 out_size, in_size, kernel_size, upsample_mode):
        super(com_decoder, self).__init__()

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]
        # define hidden layers
        self.net = nn.Sequential()
        for i in range(num_layers - 1):
            self.net.append(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            if i == 0:  # first layer
                self.net.append(ResidualBlockOrig(num_input_channels, num_channels, kernel_size))
            else:
                self.net.append(ResidualBlockOrig(num_channels, num_channels, kernel_size))

        # define final layer
        self.net.append(ResidualBlockOrig(num_channels,  num_channels, kernel_size))

        self.net.append(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0,
                                  bias=True, padding_mode="zeros"))

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    import torch

    # Test out the above architecture and make sure
    # design is correct

    # instantiate the NN - specify hyperparameters
    in_size = [8, 8]  # network input size
    kernel_size = 3  # size of convolutional filters
    num_layers = 6  # number of hidden layers
    input_channels = 128  # number of input channels
    num_channels = 128  # number of channels per hidden layer
    output_channels = 2  # number of output channels
    out_size = [256, 256]  # network output size
    upsample_mode = "nearest"  # upsampling mode (bilinear, nearest)

    shape = [1, input_channels, in_size[0], in_size[1]]
    ni = torch.autograd.Variable(torch.zeros(shape))
    ni.data.uniform_()

    # create random initialized network
    model = com_decoder(num_layers, input_channels, num_channels, output_channels,
                        out_size, in_size, kernel_size, upsample_mode)

    y = model(ni)
    print(y.shape)
    print(model)
    print('Number of parameters: ', sum(param.numel() for param in model.parameters()))
