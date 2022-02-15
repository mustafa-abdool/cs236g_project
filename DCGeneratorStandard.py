import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F

"""
DCGAN generator with some custom modifications. Assumes 64x64 image output size.
"""

class DCGeneratorStandard(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
        output_dim: dimension of the output to produce (assumes width and height are the same)
    '''
    def __init__(self, z_dim=16, im_chan=3, hidden_dim=64, output_dim = 64):
        super(DCGeneratorStandard, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        # todo (Moose): How do we know the final shape of the output image? And how do we control for it ? 
        # I think it just depends on the output size itself unfortunately...
        self.gen = nn.Sequential(
            self.make_conv_block(z_dim, hidden_dim * 8, kernel_size = 4, stride = 1, padding = 0),
            self.make_conv_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding = 1),
            self.make_conv_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride = 2, padding = 1),
            self.make_conv_block(hidden_dim * 2, hidden_dim, kernel_size = 4, stride = 2, padding = 1),
            self.make_conv_block(hidden_dim, im_chan, kernel_size=4, stride = 2, padding = 1, final_layer=True),
        )
        self.output_dim = output_dim

    def make_conv_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding = 0, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
            padding: Padding to use in the convolution layers
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                # interestingly, I wonder if this makes visualization kind of weird because the input pixel values
                # are in the range 0-1
                # this squishes the output in the range (-1, 1) to get back real pixel values
                nn.Tanh(), 
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)