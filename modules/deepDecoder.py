'''
This program creates an instantiation of the DeepDecoder
NN created by Reinhard Heckel for solving inverse problems
with untrained NN.
'''

import torch.nn as nn
import numpy as np

class deep_decoder(nn.Module):
    def __init__(self,num_layers,num_input_channels,num_channels,num_output_channels,
                 out_size,in_size,upsample_mode):
        super(deep_decoder, self).__init__()

        # parameter setup
        kernel_size = 1
        strides = [1] * (num_layers - 1)

        ### compute up-sampling factor from one layer to another
        scale_x, scale_y = (out_size[0] / in_size[0]) ** (1. / (num_layers - 1)), (out_size[1] / in_size[1]) ** (
                1. / (num_layers - 1))
        hidden_size = [(int(np.ceil(scale_x ** n * in_size[0])),
                        int(np.ceil(scale_y ** n * in_size[1]))) for n in range(1, (num_layers - 1))] + [out_size]

        # define hidden layers
        self.net = []
        for i in range(num_layers - 1):
            if(i == 0):
                conv = nn.Conv2d(num_input_channels, num_channels, kernel_size, strides[i],
                                 padding=(kernel_size - 1) // 2, bias=False, padding_mode="reflect")
            else:
                conv = nn.Conv2d(num_channels, num_channels, kernel_size, strides[i],
                                 padding=(kernel_size - 1) // 2, bias=False, padding_mode="reflect")

            self.net.append(conv)
            self.net.append(nn.Upsample(size=hidden_size[i], mode=upsample_mode))
            self.net.append(nn.ReLU())
            self.net.append(nn.BatchNorm2d(num_channels,affine=True))

        # define final layer
        self.net.append(nn.Conv2d(num_channels, num_output_channels, 1, 1, padding=0,
                                  bias=False, padding_mode="reflect"))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    import torch
    # Test out the above architecture and make sure
    # design is correct

    # instantiate the NN - specify hyperparameters
    in_size = [16, 16]  # network input size
    num_layers = 5  # number of hidden layers
    input_channels = 256  # number of input channels
    num_channels = 256  # number of channels per hidden layer
    output_channels = 2  # number of output channels
    out_size = [256, 256]  # network output size
    upsample_mode = "bilinear"  # upsample mode in NN

    shape = [1, input_channels, in_size[0], in_size[1]]
    ni = torch.autograd.Variable(torch.zeros(shape))
    ni.data.uniform_()

    # create random initialized network
    model = deep_decoder(num_layers, input_channels, num_channels, output_channels,
                       out_size, in_size, upsample_mode=upsample_mode)

    y = model(ni)
    print(y.shape)
    print(model)