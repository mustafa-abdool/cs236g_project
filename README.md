# CS236G Project - Conditional Pokemon Generation Using GANs

## Project Description

The goal is this project is to leverage GANs (Generative Adverserial Models) to automatically create new Pokemon. While this idea has been explored before, I want to specifically investigate the conditional generation of Pokemon by type - such as being able to generate a water, fire or grass type Pokemon! For full details, please see the final report file: `CS236G_Final_Report_MustafaAbdool.pdf`

## Samples/Demo
So far, the best models I've produced have used an auxillary multi-class loss for the Pokemon type or the Autoencoding Gan architecture (you can read more about that [here] (https://arxiv.org/abs/2004.05472) 


Examples of **Ghost** type Pokemon:

<img width="88" alt="Screen Shot 2022-03-09 at 8 59 12 PM" src="https://user-images.githubusercontent.com/5626138/157573362-f2a77cdb-5cb8-42be-85f0-ef8fdadd5a4f.png">

Examples of **Bug** type Pokemon:

<img width="88" alt="best bug" src="https://user-images.githubusercontent.com/5626138/157573227-f6405e5f-0391-449f-a576-579cc069c828.png">
<img width="86" alt="best bug v2" src="https://user-images.githubusercontent.com/5626138/157573228-9f168caf-9bf7-4b76-a4cc-3f1cddd26e3f.png">


Examples of **Fire** type Pokemon:

<img width="90" alt="best fire v2" src="https://user-images.githubusercontent.com/5626138/157573122-72256000-bb98-423c-bf0f-f61b842cc324.png">
<img width="90" alt="best fire" src="https://user-images.githubusercontent.com/5626138/157573130-fd41c3fd-11f1-4966-8e2f-296140ac4b4f.png">


Examples of **Steel** type Pokemon:

<img width="86" alt="best steel" src="https://user-images.githubusercontent.com/5626138/157573235-c6211680-df6e-4578-9dc0-4a8fe5d7dd36.png">


Examples of **Grass** type Pokemon:

<img width="93" alt="Screen Shot 2022-03-09 at 8 59 33 PM" src="https://user-images.githubusercontent.com/5626138/157573335-add49571-51ef-4389-9f9c-4da156b3bd7b.png">
<img width="92" alt="best grass" src="https://user-images.githubusercontent.com/5626138/157573243-e2cad7cf-7ba6-4ff4-9488-01df2f47e6f7.png">


## Where is the data located?

For copyright and size reasons, I have not uploaded the data to this repo. However, I would check out [Veekun] (https://veekun.com/dex/downloads) for all image downloads 

## How to Run it ?

I've implemented a variety of different GAN architectures in the ipython notebooks in the repo which you should be able to just run directly. Please see below for a quick list

- `Pokemon DCGAN - Conditional Generation Model.ipynb` contains a conditional DCGAN implementation
- `Pokemon GAN - StyleGAN Conditional Exploration.ipynb` contains a conditional class implementation based on StyleGAN
- `Pokemon GAN - Unet and PatchGAN Conditional.ipynb` contains a conditional class implementation based on Unet/PatchGAN architecture
- `Pokemon GAN - Unet and PatchGAN Conditional with Autoregressive encoding .ipynb` contains an example of Autoencoding GAN architecture (similar to CycleGAN and uses an additional generator/discriminator for the reverse mapping from the image space to the latent space)

The final cell in all of the notebooks provide a way to sample Pokemon from the trained model (and you can also specify the type of Pokemon you want to generate!)


## Important Files

- `dataloaders.py` contains utils for loading a file structure of Pokemon images into a Tensorflow dataloader
- `loss_functions.py` contains a variety of different loss functions to use for training GANs (for both the generator and discrimimator). These include BCE loss, MSE loss along with label smoothing.
- `UNetArchitecture.py` contains Unet specific class architectures
- `StyleGanGenerator.py` contains StyleGAN specific class architectures

## References

1. Lab Notebooks from the [GAN Coursera course] (https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/)
