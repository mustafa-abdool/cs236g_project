import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
torch.manual_seed(0) # Set for testing purposes, please do not change!

"""
Utility file to easily get different types of pokemon datasets with their associated transforms
"""


IDENTITY_NORM = transforms.Normalize(mean = (0, 0, 0), std = (1,1,1))

def get_denormalization_transform(channel_means, channel_stds):
	denorm_means = [-1.0 * mean/std for mean,std in zip(channel_means, channel_stds)]
	denorm_stds = [1.0 / std for std in channel_stds]
	return transforms.Normalize(mean = denorm_means, std = denorm_stds)


def get_normalization_stats(jpg_directory, image_dim, batch_size = 4):

	pkmn_data = datasets.ImageFolder(root = jpg_directory,transform = transforms.ToTensor())
	test_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, num_workers=1)
	batches = 0

	running_mean_sum = torch.zeros(3)
	running_std_sum = torch.zeros(3)

	for images, _ in tqdm(test_dataloader):
	  
	  flattened = images.view(batch_size, 3, -1)
	  
	  sum_per_channel_in_batch = torch.sum(flattened, dim = -1)
	  sum_per_channel = torch.sum(sum_per_channel_in_batch, dim = 0)

	  std_per_channel_in_batch = torch.std(flattened, dim = -1)
	  sum_std_per_channel = torch.sum(std_per_channel_in_batch, dim = 0)
	  

	  running_mean_sum += sum_per_channel
	  running_std_sum += sum_std_per_channel
	  
	  batches += 1

	# not exactly accurate if batch size does not divide the total length of dataset
	# but probably fine as an approximation
	final_mean_per_channel = running_mean_sum / (batches * batch_size * image_dim**2)
	final_std_per_channel = running_std_sum / (batches * batch_size)

	print("[Dataloader Stats] Final mean per channel is: {}, final std per channel is: {}".format(final_mean_per_channel, final_std_per_channel))	

	return final_mean_per_channel, final_std_per_channel

# Factory method for getting a dataloader of a certain type
# returns (dataloader, denorm_transform). 
# The denorm_transform values can be used when drawing the actual images to get it back to the right scale
def get_pkmn_dataloader(dataloader_name, batch_size):

	if dataloader_name == "shiny_64_dim":
		jpg_directory = "./pokemon_images_all_size=64_shiny=True/" 

		dataset_transforms = transforms.Compose([
		  transforms.ToTensor(),
		])
		pkmn_data = datasets.ImageFolder(root = jpg_directory, transform = dataset_transforms)
		pkmn_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)
		return pkmn_dataloader, IDENTITY_NORM

	elif dataloader_name == "shiny_64_dim_normalize_and_random_transform":
		jpg_directory = "./pokemon_images_all_size=64_shiny=True/" 

		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

		# use random flip and normalization transform
		dataset_transforms = transforms.Compose([
		  transforms.RandomHorizontalFlip(p=1.0),
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		pkmn_data = datasets.ImageFolder(root = jpg_directory,transform = dataset_transforms)
		pkmn_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform


	elif dataloader_name == "shiny_64_dim_normalize_and_random_transform_from_data":
		jpg_directory = "./pokemon_images_all_size=64_shiny=True/" 

		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = get_normalization_stats(jpg_directory, 64)

		# use random flip and normalization transform
		dataset_transforms = transforms.Compose([
		  transforms.RandomHorizontalFlip(p=1.0),
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		pkmn_data = datasets.ImageFolder(root = jpg_directory,transform = dataset_transforms)
		pkmn_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform


	elif dataloader_name == "shiny_64_dim_normalize_and_random_flip_concat_from_data":
		jpg_directory = "./pokemon_images_all_size=64_shiny=True/" 

		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = get_normalization_stats(jpg_directory, 64)


		normal_transforms = transforms.Compose([
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])


		# use random flip and normalization transform
		flip_transforms = transforms.Compose([
		  transforms.RandomHorizontalFlip(p=1.0),
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)

		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform		

	elif dataloader_name == "shiny_64_dim_normalize_and_random_flip_and_jitter":
		jpg_directory = "./pokemon_images_all_size=64_shiny=True/" 

		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = (0.5, 0.5, 0.5), (0.5,0.5,0.5)

		normal_transforms = transforms.Compose([
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		# use random flip and normalization transform
		flip_transforms = transforms.Compose([
		  transforms.RandomHorizontalFlip(p=1.0),
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		# color transform
		color_transforms = transforms.Compose([
		  transforms.ColorJitter(0.5, 0.5, 0.5),
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_color_jitter = datasets.ImageFolder(root = jpg_directory,transform = color_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_color_jitter])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "shiny_64_dim_blackbg":
		jpg_directory = "./pokemon_images_all_black_bg_size=64_shiny=True/" 

		dataset_transforms = transforms.Compose([
		  transforms.ToTensor(),
		])
		pkmn_data = datasets.ImageFolder(root = jpg_directory, transform = dataset_transforms)
		pkmn_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)
		return pkmn_dataloader, IDENTITY_NORM


	elif dataloader_name == "shiny_64_dim_blackbg_normalize_and_random_transform":
		jpg_directory = "./pokemon_images_all_black_bg_size=64_shiny=True/" 

		# Convert channels from [0, 1] to [-1, 1]
		# in this type of dataloader, we just use a heuristic mean/std
		channel_means, channel_stds = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

		# use random flip and normalization transform
		dataset_transforms = transforms.Compose([
		  transforms.RandomHorizontalFlip(p=1.0),
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		pkmn_data = datasets.ImageFolder(root = jpg_directory,transform = dataset_transforms)
		pkmn_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)


		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "shiny_64_dim_blackbg_normalize_and_random_transform_from_data":
		jpg_directory = "./pokemon_images_all_black_bg_size=64_shiny=True/" 

		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = get_normalization_stats(jpg_directory, 64)

		# use random flip and normalization transform
		dataset_transforms = transforms.Compose([
		  transforms.RandomHorizontalFlip(p=1.0),
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])

		pkmn_data = datasets.ImageFolder(root = jpg_directory,transform = dataset_transforms)
		pkmn_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=1)


		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "shiny_64_dim_condition_labels":
		return None

	elif dataloader_name == "shiny_96":
		return None

	elif dataloader_name == "shiny_96_dim_normalize_and_random_transform":
		return None


	else:
		print("No dataloader found given name: {}".format(dataloader_name))
		raise Exception("test exception")