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
Utility file to easily get different types of pokemon datasets with their associated transforms.
This code could probably be made more modular...
"""


IDENTITY_NORM = transforms.Normalize(mean = (0, 0, 0), std = (1,1,1))
STANDARD_MEAN = (0.5, 0.5, 0.5)
STANDARD_STD = (0.5, 0.5, 0.5)

def get_denormalization_transform(channel_means, channel_stds):
	denorm_means = [-1.0 * mean/std for mean,std in zip(channel_means, channel_stds)]
	denorm_stds = [1.0 / std for std in channel_stds]
	return transforms.Normalize(mean = denorm_means, std = denorm_stds)


# get normalization stats from a dataloader itself
def get_normalization_stats_from_dataloader(test_dataloader, image_dim, batch_size):

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


# get normalization stats from all files in a directory
def get_normalization_stats(jpg_directory, image_dim, batch_size = 4):

	pkmn_data = datasets.ImageFolder(root = jpg_directory,transform = transforms.ToTensor())
	test_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, num_workers=1)
	return get_normalization_stats_from_dataloader(test_dataloader, image_dim, batch_size)

# Factory method for getting a dataloader of a certain type
# returns (dataloader, denorm_transform). 
# The denorm_transform values can be used when drawing the actual images to get it back to the right scale
def get_pkmn_dataloader(dataloader_name, batch_size, num_workers = 1):

	if dataloader_name == "shiny_64_dim":
		jpg_directory = "./pokemon_images_all_size=64_shiny=True/" 

		dataset_transforms = transforms.Compose([
		  transforms.ToTensor(),
		])
		pkmn_data = datasets.ImageFolder(root = jpg_directory, transform = dataset_transforms)
		pkmn_dataloader = torch.utils.data.DataLoader(dataset=pkmn_data, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)
		return pkmn_dataloader, IDENTITY_NORM

	if dataloader_name == "conditional_64_dim_no_shiny_with_flip_standard_norm":
		jpg_directory = "./pokemon_images_by_type_all_size=64_shiny=False/" 


		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = STANDARD_MEAN, STANDARD_STD

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
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform		

	if dataloader_name == "conditional_64_dim_no_shiny_with_flip_custom_norm":
		jpg_directory = "./pokemon_images_by_type_all_size=64_shiny=False/" 


		# Convert channels from [0, 1] to [-1, 1], using real data
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
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform		

	if dataloader_name == "conditional_64_dim_no_shiny_with_flip_and_rotate_and_custom_norm":
		jpg_directory = "./pokemon_images_by_type_all_size=64_shiny=False/" 

		# Convert channels from [0, 1] to [-1, 1], using real data
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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		

		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)

		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform


	if dataloader_name == "conditional_64_dim_no_shiny_with_flip_and_rotate_and_custom_norm_black_bg":
		jpg_directory = "./pokemon_images_by_type_all_black_bg_size=64_shiny=False/" 

		# Convert channels from [0, 1] to [-1, 1], using real data
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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		

		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)

		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform		




	if dataloader_name == "conditional_64_dim_mainclass_with_shiny_and_back_flip_rotate_custom_norm":
		jpg_directory = "./pokemon_images_size=128_shiny=False_incude_back=False_bg=WHITE_mainclass=True_groupclasses=True" 

		# Convert channels from [0, 1] to [-1, 1], using real data
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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		

		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)

		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform	

	if dataloader_name == "conditional_64_dim_mainclass_with_shiny_flip_rotate_custom_norm":
		jpg_directory = "./pokemon_images_size=64_shiny=True__bg=WHITE_mainclass=True_groupclasses=True" 

		# Convert channels from [0, 1] to [-1, 1], using real data
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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		

		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)

		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform						

	if dataloader_name == "conditional_64_dim_no_shiny_with_flip_and_rotate_and_standard_norm":
		jpg_directory = "./pokemon_images_by_type_all_size=64_shiny=False/" 

		# Convert channels from [0, 1] to [-1, 1], using real data
		channel_means, channel_stds = STANDARD_MEAN, STANDARD_STD

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		

		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)

		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform							

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
		                                              num_workers=num_workers)

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
		                                              num_workers=num_workers)

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
		                                              num_workers=num_workers)

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
		                                              num_workers=num_workers)

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
		                                              num_workers=num_workers)
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
		                                              num_workers=num_workers)


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
		                                              num_workers=num_workers)


		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "conditional_64_with_shiny_and_back_flip_rotate_jitter_rotate_standard_normalize":
		jpg_directory = "./pokemon_images_size=64_shiny=True_incude_back=True_bg=WHITE_mainclass=True_groupclasses=True" 

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		


		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_color_jitter = datasets.ImageFolder(root = jpg_directory,transform = color_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_color_jitter, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "condition_64_with_shiny_flip_rotate_jitter_rotate_standard_normalize":
		jpg_directory = "./pokemon_images_size=64_shiny=True__bg=WHITE_mainclass=True_groupclasses=True" 

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		


		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_color_jitter = datasets.ImageFolder(root = jpg_directory,transform = color_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_color_jitter, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "shiny_96_flip_rotate_jitter_rotate_standard_normalize":
		jpg_directory = "./pokemon_images_all_size=96_shiny=True" 

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		


		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_color_jitter = datasets.ImageFolder(root = jpg_directory,transform = color_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_color_jitter, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "shiny_96_flip_rotate_jitter_rotate_custom_normalize":
		jpg_directory = "./pokemon_images_all_size=96_shiny=True" 

		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = get_normalization_stats(jpg_directory, 96)
		
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
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform
			

	elif dataloader_name == "conditional_64_no_shiny_mainclass_flip_rotate_standard_norm":
		jpg_directory = "./pokemon_images_size=64_shiny=False__bg=WHITE_mainclass=True_groupclasses=True" 

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		


		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	elif dataloader_name == "conditional_96_no_shiny_mainclass_flip_rotate_standard_norm":
		jpg_directory = "./pokemon_images_size=96_shiny=False__bg=WHITE_mainclass=True_groupclasses=True" 

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		


		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

		


	elif dataloader_name == "conditional_64_no_shiny_mainclass_flip_rotate_custom_norm":
		jpg_directory = "./pokemon_images_size=64_shiny=False__bg=WHITE_mainclass=True_groupclasses=True" 

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		


		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform


	elif dataloader_name == "conditional_96_no_shiny_mainclass_flip_rotate_custom_norm":
		jpg_directory = "./pokemon_images_size=96_shiny=False__bg=WHITE_mainclass=True_groupclasses=True" 

		# Convert channels from [0, 1] to [-1, 1]
		channel_means, channel_stds = get_normalization_stats(jpg_directory, 96)

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

		# use random flip and normalization transform
		rotate_transforms = transforms.Compose([
		  transforms.RandomRotation(45), # rotate up to 45 degrees in each direction
		  transforms.ToTensor(),
		  transforms.Normalize(mean = channel_means, std = channel_stds)
		])		


		pkmn_data_normal = datasets.ImageFolder(root = jpg_directory,transform = normal_transforms)
		pkmn_data_flipped = datasets.ImageFolder(root = jpg_directory,transform = flip_transforms)
		pkmn_data_rotated = datasets.ImageFolder(root = jpg_directory,transform = rotate_transforms)


		final_dataset = ConcatDataset([pkmn_data_normal, pkmn_data_flipped, pkmn_data_rotated])


		pkmn_dataloader = torch.utils.data.DataLoader(dataset=final_dataset, 
		                                              batch_size=batch_size, 
		                                              shuffle=True, 
		                                              num_workers=num_workers)

		denorm_transform = get_denormalization_transform(channel_means,channel_stds)
		return pkmn_dataloader, denorm_transform

	else:
		print("No dataloader found given name: {}".format(dataloader_name))
		raise Exception("test exception")