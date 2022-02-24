# CS236G Project - Conditional Pokemon Generation Using GANs

## Project Description

The goal is this project is to leverage GANs (Generative Adverserial Models) to automatically create new Pokemon. While this idea has been explored before, I want to specifically investigate the conditional generation of Pokemon by type - such as being able to generate a water, fire or grass type Pokemon!

## Where is the data located?

For copyright and size reasons, I have not uploaded the data to this repo. However, I would check out https://veekun.com/dex/downloads

## How to Run it ?

I've implemented a variety of different GAN architectures in the ipython notebooks in the repo which you should be able to just run directly. Please see below for a quick list

- `Pokemon DCGAN - Conditional Generation Model.ipynb` contains a conditional DCGAN implementation
- `Pokemon GAN - StyleGAN Conditional Exploration.ipynb` contains a conditional class implementation based on StyleGAN
- `Pokemon GAN - Unet and PatchGAN Conditional.ipynb` contains a conditional class implementation based on Unet/PatchGAN architecture

The final cell in all of the notebooks provide a way to sample Pokemon from the trained model (and you can also specify the type of Pokemon you want to generate!)


## File Structure

- `dataloaders.py` contains utils for loading a file structure of Pokemon images into a Tensorflow dataloader
- `loss_functions.py` contains a variety of different loss functions to use for training GANs (for both the generator and discrimimator). These include BCE loss, MSE loss along with label smoothing.

## Samples/Demo

Coming soon...
