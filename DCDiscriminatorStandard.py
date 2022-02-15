import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F

"""
DCGAN discriminator with some custom modifications. Assumes 64x64 image input.
"""

class DCDiscriminatorStandard(nn.Module):
    '''
    Discriminator Class - using deep conv layers
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
        input dim: dimension of the input image (assumes width and height are the same)
    '''
    def __init__(self, im_chan=3, hidden_dim=64):
        super(DCDiscriminatorStandard, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, kernel_size = 4, stride =2, padding = 1),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size = 4, stride =2, padding = 1),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size = 4, stride =2, padding = 1),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8, kernel_size = 4, stride =2, padding = 1),
            self.make_disc_block(hidden_dim * 8, 1, kernel_size = 4, stride =1, padding = 0, final_layer = True)
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding = 0, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
            padding: padding to use in the convolution layers
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                # if we explictly take the sigmoid here 
                # then make sure to use the proper loss later on
                # todo: should we flatten and use dropout here?
                nn.Sigmoid()
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), 1)