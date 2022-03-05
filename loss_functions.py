
import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F

"""
Basic library of loss functions for training GANs
"""


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((1-gradient_norm)**2)
    return penalty

# Critic loss used for the WGAN
def get_disc_loss_wgan(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''

    mean_fake_score = torch.mean(crit_fake_pred)
    mean_real_score = torch.mean(crit_real_pred)
    
    # we want the real score to be much higher than the fake score
    # for this to give us a "low" loss 
    crit_loss = -1.0 * (mean_real_score - mean_fake_score) + c_lambda * gp
    return crit_loss

# Generator loss used for WGAN
def get_gen_loss_wgan(crit_fake_pred, device):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -1.0 * torch.mean(crit_fake_pred) # the prediction of the critic is just the "realness" of an image
    # so the generator wants this to be as high as possible
    return gen_loss    


# specific to the multiclass problem for you
def get_gradient(crit, real, true_labels, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images, true_labels)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images, # this gives you a gradient which is the shape of the images, it backprops through
        # the discriminator network I guess
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient    

def gen_loss_least_squares(disc_fake_pred, device):
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


def disc_loss_patchGAN(disc_fake_pred, disc_real_pred, device):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs a logit that represents P(real) over some mxm patches 
    '''

    loss_func = nn.BCEWithLogitsLoss()
    
    # first, get how the disc. performs on the "fake" images (want this to be equal to 0)
    target_labels_fake_images = torch.zeros_like(disc_fake_pred, device = device)
    disc_loss_fake = loss_func(disc_fake_pred, target_labels_fake_images)
    
    target_labels_real_images = torch.ones_like(disc_real_pred,device = device)
    disc_loss_real = loss_func(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
    return disc_loss



# loss that combines two terms - a multiclass distirbution loss over what pokemon type it is
# And the typical P(real)/P(fake) loss term 
# the discrminator output is of shape (1 + num_pkmn_types)
# the first term corresponds to P(real)/P(fake) while the num_pkmn_type terms corresponds to the distribution
def disc_loss_ACGAN(disc_fake_pred, disc_real_pred, true_labels, num_pkmn_types, device, 
                         multi_class_lambda = 0.1, debug = False):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs a logit that represents P(real) over some mxm patches 
    '''

    loss_func_source_type = nn.BCEWithLogitsLoss()
    loss_func_multiclass_pkmn_type = nn.CrossEntropyLoss()

    bs = len(disc_fake_pred)

    # split output into source and pkmn type logits
    # 3 loss terms

    disc_source_pred_fake = disc_fake_pred[:, 0].view(bs, 1)
    disc_pkmn_type_logits_fake = disc_fake_pred[:, 1:].view(bs, num_pkmn_types + 1) # shape (bs, # of pkmn types + 1)

    disc_source_pred_real = disc_real_pred[:, 0].view(bs, 1)
    disc_pkmn_type_logits_real = disc_real_pred[:, 1:].view(bs, num_pkmn_types + 1) # shape (bs, # of pkmn types + 1)


    # first, get how the disc. performs on the "fake" images (want this to be equal to about 0)
    target_labels_fake_images = torch.rand(disc_source_pred_fake.shape, device = device)*0.05
    disc_loss_fake = loss_func_source_type(disc_source_pred_fake, target_labels_fake_images)
    
    # want prediction on the real images to be equal to about 1
    target_labels_real_images = torch.rand(disc_source_pred_real.shape, device = device)*0.05 + 0.95
    disc_loss_real = loss_func_source_type(disc_source_pred_real, target_labels_real_images)

    # *NEW TERM* we also want the class predictions on the real images to be correct
    disc_loss_multiclass_real = loss_func_multiclass_pkmn_type(disc_pkmn_type_logits_real, true_labels) 

    # *NEW TERM* we also want the class predictions on the fake images to be equal to a single (dummy) class
    # which we reserve as the last class output and only appears in this loss function
    target_labels_fake_classes = torch.ones_like(true_labels, device = device) * num_pkmn_types
    disc_loss_multiclass_fake = loss_func_multiclass_pkmn_type(disc_pkmn_type_logits_fake, target_labels_fake_classes.long()) 
    
    if debug:
        print("disc source losses ==> real images: {}, fakes images: {}".format(disc_loss_fake, disc_loss_real))
        print("disc class distribution losses ==> real images: {}, fakes images: {}".format(disc_loss_multiclass_real, disc_loss_multiclass_fake))

    disc_loss = (disc_loss_fake + disc_loss_real + multi_class_lambda * disc_loss_multiclass_real + multi_class_lambda * disc_loss_multiclass_fake) / 4.0
    return disc_loss


def gen_loss_ACGAN(disc_fake_pred, true_labels, num_pkmn_types, device, 
                         multi_class_lambda = 0.1, debug = False):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs a logit that represents P(real) over some mxm patches 
    '''

    loss_func_source_type = nn.BCEWithLogitsLoss()
    loss_func_multiclass_pkmn_type = nn.CrossEntropyLoss()

    bs = len(disc_fake_pred)

    # split disc output into source and pkmn type logits for the fake images
    disc_source_pred_fake = disc_fake_pred[:, 0].view(bs, 1)
    disc_pkmn_type_logits_fake = disc_fake_pred[:, 1:].view(bs, num_pkmn_types + 1) # shape (bs, # of pkmn types + 1)

    # first, get how the disc. performs on the "fake" images (want this to be equal to 1 so we can fool it)
    target_labels_fake_images = torch.ones_like(disc_source_pred_fake, device = device)
    gen_loss_source = loss_func_source_type(disc_source_pred_fake, target_labels_fake_images)

    # *NEW TERM* we also want the class predictions on the fakes images to be correct
    target_labels_fake_classes = true_labels
    gen_loss_multiclass = loss_func_multiclass_pkmn_type(disc_pkmn_type_logits_fake, true_labels) 


    gen_loss = (gen_loss_source + disc_loss_real + multi_class_lambda * gen_loss_multiclass) / 2.0
    
    if debug:
        print("gen source losses ==>  fake images: {}".format(gen_loss_source))
        print("gen class distribution losses ==> fakes images: {}".format(gen_loss_multiclass))

    return gen_loss           

def noisy_disc_loss_patchGAN(disc_fake_pred, disc_real_pred, device):
    '''
    Return the loss of a discriminator given the discriminator's scores for fake and real images
    Assumes that discriminator outputs a logit that represents P(real) over some mxm patches

    We do label smoothing here, so basically P(real) for fakes is int he range 0 - 0.1
    and P(real) for real is in the range 0.9 - 1

    '''

    loss_func = nn.BCEWithLogitsLoss()
    
    # first, get how the disc. performs on the "fake" images (want this to be equal to 0)
    target_labels_fake_images = torch.rand(disc_fake_pred.shape, device = device)*0.1
    disc_loss_fake = loss_func(disc_fake_pred, target_labels_fake_images)
    
    target_labels_real_images = torch.rand(disc_real_pred.shape, device = device)*0.1 + 0.9
    disc_loss_real = loss_func(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
    return disc_loss  


def disc_loss_least_squares(disc_fake_pred, disc_real_pred, device):
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


def disc_loss_least_squares_noisy(disc_fake_pred, disc_real_pred, device):
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
    rand_ones_target = torch.rand(disc_real_pred.shape, device = device)*0.1 + 0.9
    rand_zeros_target = torch.rand(disc_fake_pred.shape, device = device)*0.1
    disc_loss = 0.5 * (torch.mean((disc_real_pred - rand_ones_target) ** 2) + torch.mean((disc_fake_pred - rand_zeros_target)**2))
    
  
    return disc_loss          

def gen_loss_basic(disc_fake_pred, device):
    '''
    Return the loss of a generator given the discriminator's scores of the generator's fake images.
    Assumes that discriminator outputs P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    
    # we want the discriminator's predictions on the *fake* images to be equal to 1
    target_labels = torch.ones_like(disc_fake_pred, device = device)
    gen_loss = F.binary_cross_entropy(disc_fake_pred, target_labels)
    
    return gen_loss


def gen_loss_basic_with_logits(disc_fake_pred, device):
    '''
    Return the loss of a generator given the discriminator's scores of the generator's fake images.
    Assumes that discriminator outputs LOGITS that represent P(real)
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    
    # we want the discriminator's predictions on the *fake* images to be equal to 1
    loss_func = nn.BCEWithLogitsLoss()
    target_labels = torch.ones_like(disc_fake_pred, device = device)
    gen_loss = loss_func(disc_fake_pred, target_labels)
    
    return gen_loss


def disc_loss_basic(disc_fake_pred, disc_real_pred, device):
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
    target_labels_fake_images = torch.zeros_like(disc_fake_pred, device = device)
    disc_loss_fake = F.binary_cross_entropy(disc_fake_pred, target_labels_fake_images)
    
    
    target_labels_real_images = torch.ones_like(disc_real_pred,device = device)
    disc_loss_real = F.binary_cross_entropy(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
  
    return disc_loss  


def disc_loss_soft(disc_fake_pred, disc_real_pred, device):
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
    target_labels_fake_images = torch.zeros_like(disc_fake_pred, device = device) + 0.1
    disc_loss_fake = F.binary_cross_entropy(disc_fake_pred, target_labels_fake_images)
    
    
    target_labels_real_images = torch.ones_like(disc_real_pred, device = device) * 0.9
    disc_loss_real = F.binary_cross_entropy(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
  
    return disc_loss  


def disc_loss_noisy(disc_fake_pred, disc_real_pred, device):
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
    target_labels_fake_images = torch.rand(disc_fake_pred.shape, device = device)*0.1
    disc_loss_fake = F.binary_cross_entropy(disc_fake_pred, target_labels_fake_images)
    
    # uniform between (0.9, 1.0)
    target_labels_real_images = torch.rand(disc_real_pred.shape, device = device)*0.1 + 0.9
    disc_loss_real = F.binary_cross_entropy(disc_real_pred, target_labels_real_images)
    
    disc_loss = (disc_loss_fake + disc_loss_real) / 2.0
    
  
    return disc_loss      


def get_generator_loss_func(name):
    if name == "basic_gen_loss":
        return gen_loss_basic
    if name == "mse_gen_loss":
        return gen_loss_least_squares
    if name == "wgan_gen_loss":
        return get_gen_loss_wgan
    if name == "basic_gen_loss_with_logits":
        return gen_loss_basic_with_logits
    if name == "gen_loss_ACGAN":
        return gen_loss_ACGAN
    print("No gen loss function found for name: {}".format(name))


def get_disc_loss_func(name):
    if name == "basic_disc_loss":
        return disc_loss_basic
    if name == "mse_disc_loss":
        return disc_loss_least_squares
    if name == "mse_disc_loss_noisy":
        return disc_loss_least_squares_noisy
    if name == "noisy_disc_loss":
        return disc_loss_noisy
    if name == "soft_disc_loss":
        return disc_loss_soft
    if name == "wgan_disc_loss":
        return get_disc_loss_wgan
    if name == "patchgan_disc_loss":
        return disc_loss_patchGAN
    if name == "noisy_patchgan_disc_loss":
        return noisy_disc_loss_patchGAN
    if name == "disc_loss_ACGAN":
        return disc_loss_ACGAN
    else:
        print("No disc loss function found for name: {}".format(name))
