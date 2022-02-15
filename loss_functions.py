
import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F

"""
Basic library of loss functions for training GANs
"""

def gen_loss_least_squares(disc_fake_pred):
    '''
    Return the loss of a generator given the discriminator's scores of the generator's fake images.
    Assumes that discriminator outputs P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    
    # we want the discriminator's predictions on the *fake* images to be equal to 1
    gen_loss = 0.5 * torch.mean((disc_fake_pred - 1)**2)
    return gen_loss	


def disc_loss_least_squares(disc_fake_pred, disc_real_pred):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images - shape (B, 1)
        disc_real_pred: the discriminator's scores of the real images - shape (B, 1)
    Returns:
        disc_loss: a scalar for the discriminator's loss, accounting for the relevant factors
    '''
    
    # first, get how the disc. performs on the "fake" images (want this to be equal to 0)

    # basically, want D(real) to be 1 and D(fake) to be 0
    disc_loss = 0.5 * (torch.mean((disc_real_pred - 1) ** 2) + torch.mean((disc_fake_pred)**2))
    
  
    return disc_loss      

def gen_loss_basic(disc_fake_pred):
    '''
    Return the loss of a generator given the discriminator's scores of the generator's fake images.
    Assumes that discriminator outputs P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    
    # we want the discriminator's predictions on the *fake* images to be equal to 1
    target_labels = torch.ones_like(disc_fake_pred)
    gen_loss = F.binary_cross_entropy(disc_fake_pred, target_labels)
    
    return gen_loss


def disc_loss_basic(disc_fake_pred, disc_real_pred):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images - shape (B, 1)
        disc_real_pred: the discriminator's scores of the real images - shape (B, 1)
    Returns:
        disc_loss: a scalar for the discriminator's loss, accounting for the relevant factors
    '''
    
    # first, get how the disc. performs on the "fake" images (want this to be equal to 0)
    target_labels_fake_images = torch.zeros_like(disc_fake_pred)
    disc_loss_fake = F.binary_cross_entropy(disc_fake_pred, target_labels_fake_images)
    
    
    target_labels_real_images = torch.ones_like(disc_real_pred)
    disc_loss_real = F.binary_cross_entropy(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
  
    return disc_loss  


def disc_loss_soft(disc_fake_pred, disc_real_pred):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images - shape (B, 1)
        disc_real_pred: the discriminator's scores of the real images - shape (B, 1)
    Returns:
        disc_loss: a scalar for the discriminator's loss, accounting for the relevant factors
    '''
    
    # first, get how the disc. performs on the "fake" images (want this to be equal to 0)
    target_labels_fake_images = torch.zeros_like(disc_fake_pred) + 0.1
    disc_loss_fake = F.binary_cross_entropy(disc_fake_pred, target_labels_fake_images)
    
    
    target_labels_real_images = torch.ones_like(disc_real_pred) * 0.9
    disc_loss_real = F.binary_cross_entropy(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
  
    return disc_loss  


def disc_loss_noisy(disc_fake_pred, disc_real_pred):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images - shape (B, 1)
        disc_real_pred: the discriminator's scores of the real images - shape (B, 1)
    Returns:
        disc_loss: a scalar for the discriminator's loss, accounting for the relevant factors
    '''
    
    # first, get how the disc. performs on the "fake" images (want this to be equal to 0)
    # uniform between (0, 0.1)
    target_labels_fake_images = torch.rand(disc_fake_pred.shape)*0.1
    disc_loss_fake = F.binary_cross_entropy(disc_fake_pred, target_labels_fake_images)
    
    # uniform between (0.9, 1.0)
    target_labels_real_images = torch.rand(disc_real_pred.shape)*0.1 + 0.9
    disc_loss_real = F.binary_cross_entropy(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
  
    return disc_loss      


def get_generator_loss_func(name):
	if name == "basic_gen_loss":
		return gen_loss_basic
	if name == "mse_gen_loss":
		return gen_loss_least_squares
	print("No gen loss function found for name: {}".format(name))


def get_disc_loss_func(name):
	if name == "basic_disc_loss":
		return disc_loss_basic
	if name == "mse_disc_loss":
		return disc_loss_least_squares
	if name == "noisy_disc_loss":
		return disc_loss_noisy
	if name == "soft_disc_loss":
		return disc_loss_soft
	else:
		print("No disc loss function found for name: {}".format(name))
