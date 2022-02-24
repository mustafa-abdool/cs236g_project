import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F

# method to get the noise (for training time) - we just use a normal distribution
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

# method to get the noise (for training time) - we just use a normal distribution
def get_image_noise(n_samples, image_dim, device='cpu', upsampling = False, noise_dim = None):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    if noise_dim is not None and upsampling is True:
        assert noise_dim <= image_dim and image_dim % noise_dim == 0
        upsample_ratio = int (image_dim / noise_dim)
        initial_random = torch.randn((n_samples, 1, noise_dim, noise_dim), device=device)
        upsampler  = nn.Upsample(scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
        return upsampler(initial_random)
    else: 
        return torch.randn((n_samples, 1, image_dim, image_dim), device=device)




# todo: add method to get conditional sample from dataset


# method that uses the truncation trick to get a noise vector
def get_truncated_noise(num_samples, z_dim, device = 'cpu', mean = 0, std = 1, thres = 1):
    x = torch.empty(num_samples, z_dim)
    return torch.nn.init.trunc_normal_(x, mean = mean, std = std, a = -1 * thres, b = thres).to(device)
