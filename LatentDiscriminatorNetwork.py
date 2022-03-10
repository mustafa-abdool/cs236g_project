import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F
from pkmn_constants import *
from GaussianNoise import GaussianNoise


class LatentDiscriminatorNetwork(nn.Module):
    '''
    Latent Discriminator Network - takes as input (noise_vector, class) and then outputs P(real) about whether
    the latent vector was true noise or something else?
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
 
    def __init__(self, z_dim, use_class_embed = False, class_embed_size = 16, vocab_size = 13,
        initial_hidden_size = 16, use_gaussian_noise = False, gaussian_noise_std = 0.1,):
        super().__init__()
        self.z_dim = z_dim

        self.use_class_embed = use_class_embed
        self.vocab_size = vocab_size
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std

        if use_class_embed:
            self.class_embed_size = class_embed_size
            self.input_dim = self.z_dim + self.class_embed_size
            self.class_embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = class_embed_size)
        else:
            self.input_dim = self.z_dim + vocab_size # one-hot encode
            self.class_embed_size = vocab_size


        self.mapping = nn.Sequential(
            self.make_disc_block(self.input_dim, initial_hidden_size, gaussian_noise_std = gaussian_noise_std),
            self.make_disc_block(initial_hidden_size, initial_hidden_size * 2, gaussian_noise_std = gaussian_noise_std),
            self.make_disc_block(initial_hidden_size * 2, initial_hidden_size * 4, gaussian_noise_std = gaussian_noise_std),
            self.make_disc_block(initial_hidden_size * 4, initial_hidden_size * 8, gaussian_noise_std = gaussian_noise_std),
            self.make_disc_block(initial_hidden_size * 8, 1, gaussian_noise_std = gaussian_noise_std, final_layer = True)
        )


    def make_disc_block(self, input_dim, output_dim, final_layer=False, gaussian_noise_std = 0.1):
        '''
        Function to return a sequence of operators for the latent discriminator. Basically just a 
        linear layer + batchnorm + relu
        '''

        gaussian_noise_layer = nn.Identity()
        if self.use_gaussian_noise:
            print("=== Using gaussian noise with std: {}".format(self.gaussian_noise_std))
            gaussian_noise_layer = GaussianNoise(std = self.gaussian_noise_std)

        if not final_layer:
            return nn.Sequential(
                gaussian_noise_layer,
                nn.Linear(input_dim, output_dim, bias = True),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )

        else:
            assert output_dim == 1
            return nn.Sequential(
                gaussian_noise_layer,
                nn.Linear(input_dim, output_dim, bias = True),
                nn.Sigmoid(),
            )


    def forward(self, noise, class_labels):
        '''
        Function for completing a forward pass the latent discriminator network
        Noise of shape (bs, noise_dim),
        class_labels of shape (bs, 1)
        '''
        bs = len(noise)
        class_labels = class_labels.long()

        if self.use_class_embed:
            label_tensor = self.class_embedding(class_labels).view(bs, self.class_embed_size)

        else:
            # shape (bs, self.vocab_size)
            label_tensor = F.one_hot(class_labels, num_classes = self.vocab_size).view(bs, self.class_embed_size)

        noise = noise.view(bs, self.z_dim)
        label_tensor = label_tensor.view(bs, self.class_embed_size)

        #print("shape of noise is: {}".format(noise.shape))
        #print("shape of label tensor is: {}".format(label_tensor.shape))

        # concat noise and labels
        noise_and_label_input = torch.cat((noise, label_tensor), dim = 1)

        return self.mapping(noise_and_label_input)
    