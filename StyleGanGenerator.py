import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F
from pkmn_constants import *


class MappingLayers(nn.Module):
    '''
    Mapping Layers Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
 
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            # Please write a neural network which takes in tensors of 
            # shape (n_samples, z_dim) and outputs (n_samples, w_dim)
            # with a hidden layer with hidden_dim neurons
            #### START CODE HERE ####
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim)
            #### END CODE HERE ####
        )

    def forward(self, noise):
        '''
        Function for completing a forward pass of MappingLayers: 
        Given an initial noise tensor, returns the intermediate noise tensor.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.mapping(noise)
    
    def get_mapping(self):
        return self.mapping

class InjectNoise(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter( # You use nn.Parameter so that these weights can be optimized
            # Initiate the weights for the channels from a random normal distribution
            #### START CODE HERE ####
            
            # you have one weight per channel and it starts off by being initialized using N(0,1)
            torch.randn((1, channels, 1, 1))
            
            #### END CODE HERE ####
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        # Set the appropriate shape for the noise!
        
        #### START CODE HERE ####
        n_samples, channels, width, height = image.shape
        # basically you want to apply the noise to all channels at once
        # you only ahve one channel of truly random noise that is applied across all channels
        noise_shape = (n_samples, 1, width, height)
        #### END CODE HERE ####
        
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise
        return image + self.weight * noise # Applies to image after multiplying by the weight for each channel
    
    def get_weight(self):
        return self.weight
    
    def get_self(self):
        return self
    
# todo: use class conditional batch norm here
class AdaIN(nn.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''

    def __init__(self, channels, w_dim, use_class_style = False, class_embed_size = 32):
        super().__init__()

        # Normalize the input per-dimension
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.use_class_style = use_class_style
        self.class_embed_size = class_embed_size

        # You want to map w to a set of style weights per channel.
        # Replace the Nones with the correct dimensions - keep in mind that 
        # both linear maps transform a w vector into style weights 
        # corresponding to the number of image channels.

        if self.use_class_style:
            # todo: make this more general based on number of types
            self.class_embedding = nn.Embedding(num_embeddings = NUM_PKMN_TYPES, embedding_dim = class_embed_size)
            self.style_scale_transform = nn.Linear(w_dim + class_embed_size, channels)
            self.style_shift_transform = nn.Linear(w_dim + class_embed_size, channels)
        else:
            self.style_scale_transform = nn.Linear(w_dim, channels)
            self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w, class_labels = None):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector, shape: (n_samples, w_dim)
            class_labels: the labels for each image you are trying to generate (only used if use_class_style = True). Shape (n_samples, 1)
        '''

        normalized_image = self.instance_norm(image)

        if self.use_class_style:
            embeddings = self.class_embedding(class_labels).view(-1, self.class_embed_size)
            # shape (bs, w_dim + class_embed_size)
            w_and_embed_concat = torch.cat((w, embeddings), dim = 1)
            style_scale = self.style_scale_transform(w_and_embed_concat)[:, :, None, None]
            style_shift = self.style_shift_transform(w_and_embed_concat)[:, :, None, None]
        else:
            style_scale = self.style_scale_transform(w)[:, :, None, None]
            style_shift = self.style_shift_transform(w)[:, :, None, None]
        
        # Calculate the transformed image
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image
    
    def get_style_scale_transform(self):
        return self.style_scale_transform
    
    def get_style_shift_transform(self):
        return self.style_shift_transform
    
    def get_self(self):
        return self 


class MicroStyleGANGeneratorBlock(nn.Module):
    '''
    Micro StyleGAN Generator Block Class
    Values:
        in_chan: the number of channels in the input, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        kernel_size: the size of the convolving kernel
        starting_size: the size of the starting image
    '''

    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True, 
                 use_class_style = False, class_embed_size = 64):
        super().__init__()
        self.use_upsample = use_upsample
        # Replace the Nones in order to:
        # 1. Upsample to the starting_size, bilinearly (https://pytorch.org/docs/master/generated/torch.nn.Upsample.html)
        # 2. Create a kernel_size convolution which takes in 
        #    an image with in_chan and outputs one with out_chan (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
        # 3. Create an object to inject noise
        # 4. Create an AdaIN object
        # 5. Create a LeakyReLU activation with slope 0.2
        
        #### START CODE HERE ####
        if self.use_upsample:
            self.upsample = nn.Upsample(starting_size, mode='bilinear')
        self.use_class_style = use_class_style
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1) # Padding is used to maintain the image size
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim, use_class_style = use_class_style, class_embed_size = class_embed_size)
        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        #### END CODE HERE ####

    def forward(self, x, w, class_labels = None):
        '''
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w, 
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
            class_labels: true class labels, shape (bs, 1) [not always used]
        '''
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        if self.use_class_style:
            x = self.adain(x, w, class_labels)
        else:
            x = self.adain(x, w)
        return x
    
    #UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self;

class MicroStyleGANGenerator(nn.Module):
    '''
    Micro StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''

    def __init__(self, 
                 z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan,
                 output_dim = 64):
        super().__init__()
        # setup specifically for this
        assert output_dim in set([64,128])
        self.output_dim = output_dim
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        # Typically this constant is initiated to all ones
        self.starting_constant = nn.Parameter(torch.ones(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)
        self.block3 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 32)
        self.block4 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 64)
        if output_dim == 128:
          self.block5 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 128)
        # You need to have a way of mapping from the output noise to an image, 
        # so you learn a 1x1 convolution to transform the e.g. 512 channels into 3 channels
        # (Note that this is simplified, with clipping used in the real StyleGAN)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block3_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block4_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        if output_dim == 128:
          self.block5_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        '''
        Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
        upsamples the first to have the same dimensions as the second.
        Parameters:
            smaller_image: the smaller image to upsample
            bigger_image: the bigger image whose dimensions will be upsampled to
        '''
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):
        '''
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise, 
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
            return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
        '''
        noise = torch.squeeze(noise)
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        
        x_block1 = self.block1(x, w) # First generator run output
        x_block1_image = self.block1_to_image(x_block1)
        
        x_block2 = self.block2(x_block1, w) # Second generator run output 
        x_block2_image = self.block2_to_image(x_block2)
        
        x_block3 = self.block3(x_block2, w) # third generator block output
        x_block3_image = self.block3_to_image(x_block3)
        
        x_block4 = self.block4(x_block3, w) # fourth generator block output
        x_block4_image = self.block4_to_image(x_block4)
                
        #x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image) # Upsample first generator run output to be same size as second generator run output 
        
        # Interpolate between the upsampled image and the image from the generator using alpha
        #### START CODE HERE ####
        # interpolation = self.alpha * x_big_image + (1 - self.alpha) * x_small_upsample
        #### END CODE HERE #### 
        
        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
          
        if self.output_dim == 128:
          x_block5 = self.block5(x_block4, w) # fourth generator block output
          x_block5_image = self.block5_to_image(x_block5)
          return x_block5_image
        
        else:
          return x_block4_image
    
    #UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self;


class MicroStyleGANGeneratorConditional(nn.Module):
    '''
    Micro StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        in_chan: the dimension of the constant input, usually w_dim, a scalar
        out_chan: the number of channels wanted in the output, a scalar
        kernel_size: the size of the convolving kernel
        hidden_chan: the inner dimension, a scalar
    '''

    def __init__(self, 
                 z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan,
                 output_dim = 64,
                 use_class_embed = False,
                 class_embed_size = 16,
                 # whether or not to learn a different scale/shift parameter for each AdaIn block
                 use_class_style = False, 
                 class_style_embed_size = 64):
        super().__init__()
        # setup specifically for this
        assert output_dim in set([64,128])
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.use_class_embed = use_class_embed
        
        # Typically this constant is initiated to all ones
        self.starting_constant = nn.Parameter(torch.ones(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False, use_class_style = use_class_style, class_embed_size = class_style_embed_size)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8, use_class_style = use_class_style, class_embed_size = class_style_embed_size)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16, use_class_style = use_class_style, class_embed_size = class_style_embed_size)
        self.block3 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 32 ,use_class_style = use_class_style, class_embed_size = class_style_embed_size)
        self.block4 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 64, use_class_style = use_class_style, class_embed_size = class_style_embed_size)
        if output_dim == 128:
          self.block5 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 128, use_class_style = use_class_style, class_embed_size = class_style_embed_size)
        # You need to have a way of mapping from the output noise to an image, 
        # so you learn a 1x1 convolution to transform the e.g. 512 channels into 3 channels
        # (Note that this is simplified, with clipping used in the real StyleGAN)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block3_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block4_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        if output_dim == 128:
          self.block5_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)

        # class embedding stuff - if you want to concat W and the noise vector
        if use_class_embed:
            self.class_embed_size = class_embed_size
            self.input_dim = self.z_dim + self.class_embed_size
            self.class_embedding = nn.Embedding(num_embeddings = NUM_PKMN_TYPES, embedding_dim = class_embed_size)
        else:
            self.input_dim = self.z_dim + NUM_PKMN_TYPES # one-hot encode
            self.class_embed_size = NUM_PKMN_TYPES        

        self.map = MappingLayers(self.input_dim, map_hidden_dim, w_dim)

        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        '''
        Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
        upsamples the first to have the same dimensions as the second.
        Parameters:
            smaller_image: the smaller image to upsample
            bigger_image: the bigger image whose dimensions will be upsampled to
        '''
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, class_labels, return_intermediate=False):
        '''
        Function for completing a forward pass of MicroStyleGANGenerator: Given noise, 
        computes a StyleGAN iteration.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
            return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
        '''
        noise = torch.squeeze(noise)

        bs = len(noise)
        class_labels = class_labels.long()

        if self.use_class_embed:
            label_tensor = self.class_embedding(class_labels)

        else:
            # shape (bs, 18)
            label_tensor = F.one_hot(class_labels, num_classes = NUM_PKMN_TYPES)

        noise = noise.view(bs, self.z_dim)
        label_tensor = label_tensor.view(bs, self.class_embed_size)

        # concat noise and labels
        noise_and_label_input = torch.cat((noise, label_tensor), dim = 1)


        x = self.starting_constant
        w = self.map(noise_and_label_input)
        x = self.block0(x, w, class_labels)
        
        x_block1 = self.block1(x, w, class_labels) # First generator run output
        #x_block1_image = self.block1_to_image(x_block1)
        
        x_block2 = self.block2(x_block1, w, class_labels) # Second generator run output 
        #x_block2_image = self.block2_to_image(x_block2)
        
        x_block3 = self.block3(x_block2, w, class_labels) # third generator block output
        #x_block3_image = self.block3_to_image(x_block3)
        
        x_block4 = self.block4(x_block3, w, class_labels) # fourth generator block output
        x_block4_image = self.block4_to_image(x_block4)
                
        #x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image) # Upsample first generator run output to be same size as second generator run output 
        
        # Interpolate between the upsampled image and the image from the generator using alpha
        #### START CODE HERE ####
        # interpolation = self.alpha * x_big_image + (1 - self.alpha) * x_small_upsample
        #### END CODE HERE #### 
        
        tanh_func = nn.Tanh()
        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
          
        # need one more block
        if self.output_dim == 128:
          x_block5 = self.block5(x_block4, w, class_labels) # fourth generator block output
          x_block5_image = self.block5_to_image(x_block5)

          return tanh_func(x_block5_image)
        
        else:
          return tanh_func(x_block4_image)
    
    #UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self;        