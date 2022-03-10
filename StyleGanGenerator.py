import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F
from pkmn_constants import *

"""
Classes used for class conditioned StyleGAN generator code.
"""


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
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim)
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
            # you have one weight per channel and it starts off by being initialized using N(0,1)
            torch.randn((1, channels, 1, 1))
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''
        # Set the appropriate shape for the noise!
        
        n_samples, channels, width, height = image.shape
        # basically you want to apply the noise to all channels at once
        # you only have one channel of truly random noise that is applied across all channels
        noise_shape = (n_samples, 1, width, height)
        
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise
        return image + self.weight * noise # Applies to image after multiplying by the weight for each channel
    
    def get_weight(self):
        return self.weight
    
    def get_self(self):
        return self
    
class AdaIN(nn.Module):
    '''
    AdaIN Class
    Values:
        channels: the number of channels the image has, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        use_class_style : whether or not class embeddings will be used to produce scale/shift parameters
        class_style_weight: weight between class normalization parameters vs. general ones
        class_embed_size: embedding size if doing class conditional normalization
    '''

    def __init__(self, channels, w_dim, use_class_style = False, class_style_weight = 1.0, class_embed_size = 32):
        super().__init__()

        # Normalize the input per-dimension
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.use_class_style = use_class_style
        self.class_embed_size = class_embed_size

        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)


        if self.use_class_style:
            self.class_style_weight = class_style_weight
            # todo: make this more general based on number of types
            self.class_embedding = nn.Embedding(num_embeddings = NUM_PKMN_TYPES, embedding_dim = class_embed_size)
            self.class_conditional_style_scale_transform = nn.Linear(class_embed_size, channels)
            self.class_conditional_style_shift_transform = nn.Linear(class_embed_size, channels)
        else:


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
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]

        if self.use_class_style:
            # get (style, shift) parameters for each class and add them based on weight
            embeddings = self.class_embedding(class_labels).view(-1, self.class_embed_size)
            style_scale_per_class = self.style_scale_transform(embeddings)[:, :, None, None]
            style_shift_per_class = self.style_shift_transform(embeddings)[:, :, None, None]
            style_scale = style_scale + self.class_style_weight * style_scale_per_class
            style_shift = style_shift + self.class_style_weight + style_shift_per_class

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
        
        if self.use_upsample:
            self.upsample = nn.Upsample(starting_size, mode='bilinear')
        self.use_class_style = use_class_style
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1) # Padding is used to maintain the image size
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim, use_class_style = use_class_style, class_embed_size = class_embed_size)
        self.activation = nn.LeakyReLU(negative_slope = 0.2)

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
                
        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
          
        if self.output_dim == 128:
          x_block5 = self.block5(x_block4, w) # fourth generator block output
          x_block5_image = self.block5_to_image(x_block5)
          return x_block5_image
        
        else:
          return x_block4_image

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
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False, use_class_style = use_class_style, class_embed_size = class_style_embed_size, class_style_weight = 0.1)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8, use_class_style = use_class_style, class_embed_size = class_style_embed_size, class_style_weight = 0.25)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16, use_class_style = use_class_style, class_embed_size = class_style_embed_size, class_style_weight = 0.25)
        self.block3 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 32 ,use_class_style = use_class_style, class_embed_size = class_style_embed_size, class_style_weight = 0.5)
        self.block4 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 64, use_class_style = use_class_style, class_embed_size = class_style_embed_size, class_style_weight = 1.0)
        if output_dim == 128:
          self.block5 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 128, use_class_style = use_class_style, class_embed_size = class_style_embed_size, class_style_weight = 1.0)
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
        
        x_block2 = self.block2(x_block1, w, class_labels) # Second generator run output 

        
        x_block3 = self.block3(x_block2, w, class_labels) # third generator block output

        
        x_block4 = self.block4(x_block3, w, class_labels) # fourth generator block output
        x_block4_image = self.block4_to_image(x_block4)
        
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
    
    def get_self(self):
        return self;        