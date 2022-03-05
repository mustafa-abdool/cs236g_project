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


# method to get random labels for some real images to use as fake examples in the discriminator
# method to get random labels for some real images to use as fake examples in the discriminator
def get_incorrect_labels_for_real(true_labels, neg_sample_bs, num_pkmn_types, device = "cpu", debug = False):
 
  true_labels_neg_sample = true_labels[0:neg_sample_bs].to(device)
  random_labels = torch.randint(low = 0, high = num_pkmn_types, size = (neg_sample_bs,)).to(device)
  # if you happened to use the correct label, just increment it!
  fixed_labels = torch.where(random_labels == true_labels_neg_sample, random_labels + 1, random_labels).to(device)

  if debug:
    print("Current batch size is: {}".format(cur_bs))
    print("Random labels: {}".format(random_labels))
    print("True labels: {}".format(true_labels_neg_sample))
    print("Equality map: {}".format(random_labels == true_labels_neg_sample))
    print(fixed_labels == true_labels_neg_sample)
    print("Final random labels: {}".format(fixed_labels))

  return fixed_labels
  


# method that uses the truncation trick to get a noise vector
def get_truncated_noise(num_samples, z_dim, device = 'cpu', mean = 0, std = 1, thres = 1):
    x = torch.empty(num_samples, z_dim)
    return torch.nn.init.trunc_normal_(x, mean = mean, std = std, a = -1 * thres, b = thres).to(device)
