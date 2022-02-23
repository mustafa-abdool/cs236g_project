import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F
from pkmn_constants import *
from MappingNetwork import MappingNetwork


# GRADED FUNCTION: crop
def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    # There are many ways to implement this crop function, but it's what allows
    # the skip connection to function as intended with two differently sized images!
    #### START CODE HERE ####
    
    
    #print("Size of image is: {}".format(image.shape))
    #print("New shape is: {}".format(new_shape))
    
    new_width = new_shape[2]
    new_height = new_shape[3]
    
    curr_width = image.shape[2]
    curr_height = image.shape[3]
    
    # Figure how much you need to chop off in a central type of way
    delta_w = (curr_width - new_width) // 2
    delta_h = (curr_height - new_height) // 2
    
    width_odd_delta = 1 if (curr_width - new_width) % 2 == 1 else 0
    height_odd_delta = 1 if (curr_height - new_height) % 2 == 1 else 0
    
    #print("Delta w is: {}, delta h is: {}".format(delta_w, delta_h))
    
    cropped_image = image[:, :, delta_w:curr_width - delta_w - width_odd_delta, delta_h : curr_height - delta_h - height_odd_delta]
    
    #print("Shape of final cropped image is: {}, target shape is: {}, input image has shape: {}".format(cropped_image.shape, new_shape, image.shape))
    
    #### END CODE HERE ####
    return cropped_image
class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True, dropout_prob):
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

class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

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

class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels = 1, output_channels = 3, 
        hidden_channels=32, input_dim = 96, z_dim = 32,
        use_dropout = False, dropout_prob = 0.5):
        super(UNet, self).__init__()

        assert input_dim in set([64, 96])


        # we tile the noise vector to make the input image, so it has to be divisible by it
        self.z_dim = z_dim
        self.input_dim = input_dim
        assert input_dim % z_dim == 0
        
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True, dropout_prob = 0.5)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True, dropout_prob = 0.5)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True, dropout_prob = 0.5)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.contract5 = ContractingBlock(hidden_channels * 16)
        self.contract6 = ContractingBlock(hidden_channels * 32)
        self.expand0 = ExpandingBlock(hidden_channels * 64)
        self.expand1 = ExpandingBlock(hidden_channels * 32)
        self.expand2 = ExpandingBlock(hidden_channels * 16)
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        # Basically, you can just add some other channels if you want to upsample the conversation
        self.final_transform = torch.nn.ConvTranspose2d(output_channels, output_channels, kernel_size=8 + 25, stride = 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, noise_vec):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            noise_vec: noise tensor of shape (n_samples, z_dim) 
            Tiled to create an image tensor of shape (batch size, 1, height, width)
        '''
        
        # similar to the conditional GAN, we need to tile the noise vec to produce a (bs, 1, input_dim, input_dim)
        # image
        
        bs = noise_vec.shape[0]
        first_dim_tile_size = int(self.input_dim / self.z_dim)
        tiled_input = noise_vec.tile((first_dim_tile_size,self.input_dim)).view(bs, 1, self.input_dim, self.input_dim)

        x0 = self.upfeature(tiled_input)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        x13 = self.final_transform(xn)
        # NOTE: We use a tanh here to make sure the output image is in the range [-1, 1], the original architecture
        # used a sigmoid
        return self.tanh(x13)


class UNetConditional(nn.Module):
    '''
    UNet Class that implements a conditional GAN (labels are also provided)
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels = 1, output_channels = 3, hidden_channels=32, 
                 input_dim = 96, z_dim = 32, use_class_embed = False, class_embed_size = 16,
                 use_conditional_layer_arch = False, use_mapping_network = False, 
                 map_network_hidden_size = 16, dropout_prob = 0.5):
        super(UNetConditional, self).__init__()

        assert input_dim in set([64, 96])

        # we tile the noise vector to make the input image, so it has to be divisible by it
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.use_class_embed = use_class_embed
        self.use_conditional_layer_arch = use_conditional_layer_arch
        self.use_mapping_network = use_mapping_network

        if use_class_embed:
            self.class_embed_size = class_embed_size
            self.final_dim = self.z_dim + self.class_embed_size
            self.class_embedding = nn.Embedding(num_embeddings = NUM_PKMN_TYPES, embedding_dim = class_embed_size)
        else:
            self.final_dim = self.z_dim + NUM_PKMN_TYPES # one-hot encode
            self.class_embed_size = NUM_PKMN_TYPES

        if self.use_mapping_network:
            self.mapping_network = MappingNetwork(self.final_dim, map_network_hidden_size, self.final_dim)
        if self.use_conditional_layer_arch:
            self.intermediate_embed_dim = 1024 # corresponds to hidden_channels = 16
            # 1024 is the size of the layer after all the contracting paths
            self.intermediate_mapping_network = MappingNetwork(self.intermediate_embed_dim + self.class_embed_size, map_network_hidden_size, self.intermediate_embed_dim)        

        assert input_dim % self.final_dim == 0
        
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True, dropout_prob = 0.5)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True, dropout_prob = 0.5)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True, dropout_prob = 0.5)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.contract5 = ContractingBlock(hidden_channels * 16)
        self.contract6 = ContractingBlock(hidden_channels * 32)
        self.expand0 = ExpandingBlock(hidden_channels * 64)
        self.expand1 = ExpandingBlock(hidden_channels * 32)
        self.expand2 = ExpandingBlock(hidden_channels * 16)
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        # Basically, you can just add some other channels if you want to upsample the conversation
        self.final_transform = torch.nn.ConvTranspose2d(output_channels, output_channels, kernel_size=8 + 25, stride = 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, noise_vec, class_labels):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            noise_vec: noise tensor of shape (n_samples, z_dim)
            labels: label tenosr of shape (n_samples, 1) 
            Tiled to create an image tensor of shape (batch size, 1, height, width)
        '''
        
        # we need to tile the noise vec to produce a (bs, 1, output_dim, output_dim) image
        bs = noise_vec.shape[0]
        class_labels = class_labels.long()

        # There are two ways to do this: 
        # We just lookup class embeddings and concatenate with the noise tensor or use one-hot encoding
        if self.use_class_embed:
            label_tensor = self.class_embedding(class_labels)

        else:
            # shape (bs, 18)
            label_tensor = F.one_hot(class_labels, num_classes = NUM_PKMN_TYPES)

        noise = noise_vec.view(bs, self.z_dim)
        label_tensor = label_tensor.view(bs, self.class_embed_size)

        # concat noise and labels
        noise_and_label_input = torch.cat((noise, label_tensor), dim = 1)

        # compute how much you need to tile by
        first_dim_tile_size = int(self.input_dim / self.final_dim)

        # run through the mapping network if you want
        if self.use_mapping_network:
            mapping_output = self.mapping_network(noise_and_label_input)
        else:
            mapping_output = noise_and_label_input
            
        
        tiled_input = mapping_output.tile((first_dim_tile_size,self.input_dim)).view(bs, 1, self.input_dim, self.input_dim)


        # The second way is more complicated, we need to inject the class at all the expand layers
        x0 = self.upfeature(tiled_input)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        #print("Shape after final contraction is: {}".format(x6.shape))
        if self.use_conditional_layer_arch:
            # x6 has shape (bs, self.intermediate_embed_dim, 1, 1)
            label_tensor_reshaped = label_tensor.view(bs, self.class_embed_size, 1, 1)
            x6_concat = torch.cat((x6, label_tensor_reshaped), dim = 1).view(bs, self.intermediate_embed_dim + self.class_embed_size)
            x6_map_out = self.intermediate_mapping_network(x6_concat).view(bs, self.intermediate_embed_dim, 1, 1)
            x6 = x6_map_out
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        x13 = self.final_transform(xn)
        # NOTE: We use a tanh here to make sure the output image is in the range [-1, 1], the original architecture
        # used a sigmoid
        return self.tanh(x13)        