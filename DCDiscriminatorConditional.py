import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F
from pkmn_constants import *

"""
DCGAN discriminator with some custom modifications. Assumes 64x64 image input.
"""

class DCDiscriminatorConditional(nn.Module):
    '''
    Discriminator Class - using deep conv layers
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
        input dim: dimension of the input image (assumes width and height are the same)
        class_embed_size must be a factor of 64
    '''
    def __init__(self, im_chan=3, hidden_dim=64, early_dropout=0.2, 
                        mid_dropout = 0.25, late_dropout = 0.3,
                        class_embed_size = 16):
        super(DCDiscriminatorConditional, self).__init__()
        self.input_image_dim = 64 # can't be changed easily
        self.class_embed_size = class_embed_size
        self.late_dropout = late_dropout
        self.hidden_dim = hidden_dim

        # create the class embeddig
        self.class_embedding = nn.Embedding(num_embeddings = NUM_PKMN_TYPES, embedding_dim = class_embed_size)

        self.input_channels = im_chan + 1 # add an extra channel for the label
        self.disc = nn.Sequential(
            self.make_disc_block(self.input_channels, hidden_dim, kernel_size = 4, stride =2, padding = 1),
            nn.Dropout2d(p = early_dropout),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size = 4, stride =2, padding = 1),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size = 4, stride =2, padding = 1),
            nn.Dropout2d(p = mid_dropout),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8, kernel_size = 4, stride =2, padding = 1),
            self.make_disc_block(hidden_dim * 8, hidden_dim, kernel_size = 4, stride =1, padding = 0, final_layer = True)
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
                nn.Flatten(),
                # some people said dropout layers tend to work better
                nn.Dropout(p = self.late_dropout),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, image, class_labels):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: an image tensor of shape (bs, 3, 64, 64)
            class_labels: proposed labels of the image, shape (bs)
        '''

        bs = len(image)

        class_labels = class_labels.view(bs)

        class_embed = self.class_embedding(class_labels.long())

        first_dim_tile_size = int(self.input_image_dim / self.class_embed_size)
        class_embed_tiled = class_embed.tile((first_dim_tile_size,self.input_image_dim)).view(bs, 1, self.input_image_dim, self.input_image_dim)

        image_and_class_embed = torch.cat((image, class_embed_tiled), dim = 1)

        disc_pred = self.disc(image_and_class_embed)
        return disc_pred.view(bs, 1)