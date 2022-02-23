import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F
from pkmn_constants import *

class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True, dropout_prob = 0.5):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout(p = dropout_prob)
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


class DiscriminatorPatchGAN(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels=3, hidden_channels=8, use_dropout = False, dropout_prob = 0.5):
        super(DiscriminatorPatchGAN, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, use_dropout = use_dropout, dropout_prob = dropout_prob)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout = use_dropout, dropout_prob = dropout_prob)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout = use_dropout, dropout_prob = dropout_prob)
        self.contract4 = ContractingBlock(hidden_channels * 8, use_dropout = use_dropout, dropout_prob = dropout_prob)
        #### START CODE HERE ####
        # Basically, you want to map into a 1 channel image
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    # PatchGAN input is (x,y) where x and y are both are images (cause it's used for image2image translation)
    # not that the output is NOT passed through a sigmoid, so we have to use BCEWithLogitsLoss
    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


class DiscriminatorPatchGANConditional(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels=3, hidden_channels=8, 
                class_embed_size = 16, input_image_dim = 96, use_dropout = False, dropout_prob = 0.5):
        super(DiscriminatorPatchGANConditional, self).__init__()


        # create the class embeddig
        self.class_embedding = nn.Embedding(num_embeddings = NUM_PKMN_TYPES, embedding_dim = class_embed_size)
        self.input_image_dim = input_image_dim
        self.class_embed_size = class_embed_size

        assert input_image_dim % self.class_embed_size == 0

        # since we add an additional channel for the class map, we have to add +1 here,
        self.upfeature = FeatureMapBlock(input_channels + 1, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, use_dropout = use_dropout, dropout_prob = dropout_prob)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout = use_dropout, dropout_prob = dropout_prob)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout = use_dropout, dropout_prob = dropout_prob)
        self.contract4 = ContractingBlock(hidden_channels * 8, use_dropout = use_dropout, dropout_prob = dropout_prob)
        #### START CODE HERE ####
        # Basically, you want to map into a 1 channel image
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
        #### END CODE HERE ####

    # PatchGAN input is (x,y) where x and y are both are images (cause it's used for image2image translation)
    # not that the output is NOT passed through a sigmoid, so we have to use BCEWithLogitsLoss
    # x is shape (bs, 3, 96, 96)
    # class_labels is shape: (bs, 1)
    def forward(self, x, class_labels):

        bs = x.shape[0]

        class_labels = class_labels.view(bs)

        class_embed = self.class_embedding(class_labels.long())

        first_dim_tile_size = int(self.input_image_dim / self.class_embed_size)
        class_embed_tiled = class_embed.tile((first_dim_tile_size,self.input_image_dim)).view(bs, 1, self.input_image_dim, self.input_image_dim)

        image_and_class_embed = torch.cat((x, class_embed_tiled), dim = 1)        

        # should now be of shape (bs, 4, 96, 96)
        x = image_and_class_embed

        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn        