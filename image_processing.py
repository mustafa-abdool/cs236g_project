import PIL
from PIL import Image
import glob
import os
import argparse
import json
from pkmn_constants import PKMN_TYPES
import pprint

pp = pprint.PrettyPrinter(indent=4)

cur_dir = os.getcwd()


# used for the glob thing, format is (prefix, path)
INPUT_IMAGE_DIRS = [
	('firered-leafgreen', './pokemon_data/gen_3/main-sprites/firered-leafgreen/'),
	('ruby-sapphire', './pokemon_data/gen_3/main-sprites/ruby-sapphire/'),
	('platnium', './pokemon_data/gen_4/main-sprites/platinum/'),
	('heartgold-soulsilver', './pokemon_data/gen_4/main-sprites/heartgold-soulsilver/'),
	('black-white','./pokemon_data/gen_5/main-sprites/black-white/')
]

# metadata file location 
METADATA_FILE = "pokemon_metadata.json"

# Map from alternative type --> Main type
CLASS_GROUP_MAPPING = {
	'Ice' : 'Water',
	'Flying' : 'Dragon',
	'Fairy' : 'Psychic',
	'Ground' : 'Rock',
	'Dark' : 'Ghost',
}

class ImageProcessor:

	def get_valid_types(self):

		if self.config['group_classes']:
			all_types = set(PKMN_TYPES)
			removed_types = set(CLASS_GROUP_MAPPING.keys())
			remaining_types = all_types - removed_types
			return remaining_types

		else:
			return PKMN_TYPES

	def get_alternative_type(self,current_type):
		if current_type in CLASS_GROUP_MAPPING:
			return CLASS_GROUP_MAPPING[current_type]
		else:
			return current_type

	def __init__(self):
		print("in the init method...")
		args = self.parse_args()
		args_dict = vars(args)  # convert Namespace to a dict

		# dictionary of parameters used
		self.config = args_dict

		if self.config['classification_mode']:
			self.id_to_type_dict = self.setup_id_to_type_dict()
			self.all_types = self.get_valid_types() #set([v[0] for k,v in self.id_to_type_dict.items()])
			if self.config['group_classes']:
				assert len(self.all_types) < 18
			else:
				assert len(self.all_types) == 18


			self.type_count_dict = dict((type_name, 0) for type_name in self.all_types)
			print(self.type_count_dict)

	def parse_args(self):
		parser = argparse.ArgumentParser()

		"""optional args"""
		# debug flags
		parser.add_argument(
		    '--debug',
		    type=bool,
		    default=False)
		# output image size (assume that it's nxn)
		parser.add_argument(
		    '--output_size',
		    type=int,
		    default=64)
		# if true, will force resize the image to --ouptut_size
		parser.add_argument(
		    '--force_resize',
		    type=bool,
		    default=True)		
		# don't process images, just print out work
		parser.add_argument('--dry_run',
		                    type=bool,
		                    default=False)
		# whether to include shiny folders or not
		parser.add_argument('--include_shiny',
		                    type=bool,
		                    default=False)
		# whether to include back sprites folder or not
		parser.add_argument('--include_back',
		                    type=bool,
		                    default=False)		
		# whether to create sub-folders for each pokemon type (requires meta-data file to be loaded)		
		parser.add_argument('--classification_mode',
		                    type=bool,
		                    default=False)
		# If using conditional mode, whether to duplicate images for single-type Pokemon		
		parser.add_argument('--balance_classes',
		                    type=bool,
		                    default=False)
		# When using conditional generation, whether to group Pokemon of the same general class together
		parser.add_argument('--group_classes',
		                    type=bool,
		                    default=False)
		# When using conditional generation, whether to only take the first type of dual type Pokemon
		parser.add_argument('--main_class_only',
		                    type=bool,
		                    default=False)		                    			                    		
		parser.add_argument('--output_dir_name',
		                    type=str,
		                    default="pokemon_images")	
		parser.add_argument('--inner_dir_name',
		                    type=str,
		                    default="all_pokemon")	
		parser.add_argument('--bg_color',
		                    type=str,
		                    default="WHITE")		                    	
		args = parser.parse_args()
		return args

	def setup_id_to_type_dict(self):
		id_to_type_dict = dict()
		json_data = json.load(open(METADATA_FILE))
		for pkmn in json_data:
			pkmn_id = pkmn['id']
			pkmn_types = pkmn["type"]
			id_to_type_dict[str(pkmn_id)] = pkmn_types
		return id_to_type_dict

	def process_images(self):
		# for classification mode, the inner dir name is given by the type, so we don't specify it here
		if self.config['classification_mode']:
			output_dir = "{output_dir_name}_size={output_size}_shiny={include_shiny}_incude_back={include_back}_bg={bg_color}_mainclass={main_class_only}_groupclasses={group_classes}".format(**self.config)
		else:
			output_dir = "{output_dir_name}_size={output_size}_shiny={include_shiny}_bg={bg_color}/{inner_dir_name}".format(**self.config)
		include_shiny = self.config['include_shiny']
		include_back = self.config['include_back']
		output_width = self.config['output_size']
		debug = self.config['debug']
		resize_image = self.config['force_resize']
		classification_mode = self.config['classification_mode']
		bg_color = self.config['bg_color']

		if not os.path.exists(output_dir) and not self.config['dry_run']:
			print("Creating output dir: {}".format(output_dir))
			os.makedirs(output_dir)

		# for classification mode, we have to create all the inner dirs
		if self.config['classification_mode']:
			for pkmn_type in self.all_types:
				type_dir = "{}/{}".format(output_dir,pkmn_type)
				print("Creating output dir: {}".format(type_dir))
				if not os.path.exists(type_dir) and not self.config['dry_run']:
					os.makedirs(type_dir)				

		total_processed = 0

		for folder_prefix, pkmn_dir in INPUT_IMAGE_DIRS:
			original_dir = pkmn_dir + "*.png"
			shiny_dir = pkmn_dir + "shiny/*.png"
			back_sprite_dir = pkmn_dir + "back/*.png"
			back_shiny_dir = pkmn_dir + "shiny/back/*.png"
			all_img_dirs = [(original_dir, False, False)]

			if include_shiny:
				all_img_dirs += [(shiny_dir, True, False)]
			if include_back:
				all_img_dirs += [(back_sprite_dir, False, True)]
			if include_back and include_shiny:
				all_img_dirs += [(back_shiny_dir, True, True)]

			for img_dir,is_shiny, is_back in all_img_dirs:

				shiny_prefix = ""
				back_prefix = ""
				if is_shiny:
					shiny_prefix = "shiny_"
				if is_back:
					back_prefix = "back_"

				img_files = glob.glob(img_dir)

				print("Processing dir: {}".format(img_dir))

				if len(img_files) == 0:
					print("!!!! No image files found in dir: {}".format(img_dir))
					continue

				assert len(img_files) > 0

				print("Found {} files in dir: {}".format(len(img_files), img_dir))

				total_processed += len(img_files)

				if not self.config['dry_run']:
					for file in glob.glob(img_dir):
						img = Image.open(file).convert("RGBA")

						# this is to make it so that all pokemon have a white (or black) colored background
						new_image = Image.new("RGBA", img.size, bg_color.upper())

						new_image.paste(img, mask = img)

						# convert to RGB so we have 3 channels
						jpg = new_image.convert('RGB')

						# resize the image if this was an input
						if resize_image:
							# assumes all dimensions are the same
							scale_factor = output_width / jpg.size[0]

							if debug:
								print("original size is: {}".format(jpg.size))
								print("Trying to resize image to: ({}, {}), scale factor is: {}".format(output_width, output_height,scale_factor))
							
							jpg = jpg.resize( [output_width for s in jpg.size], PIL.Image.LANCZOS)

							# thumbnail only works if lowering size
							#jpg.thumbnail((output_width, output_height), PIL.Image.LANCZOS) #Image.BICUBIC) #PIL.Image.LANCZOS)

						# get the last part of the file path and then get the number
						filename_last_portion = file.split('/')[-1]

						# split into {number}.png
						pkmn_number = filename_last_portion.split(".")[0]


						if classification_mode:
							# to solve special cases like 649-chill, ie. a pokemon has different modes
							pkmn_num_tag = pkmn_number.split("-")[0]
							
							if pkmn_num_tag not in self.id_to_type_dict:
								print("=== WARNING, skip file with name: {} and tag: {}".format(file, pkmn_num_tag))
								continue

							pkmn_types = self.id_to_type_dict[pkmn_num_tag]

							#print("Original pkmn type is: {}".format(pkmn_types))

							# only use the first class in this mode, if a pokemon as dual types
							if self.config['main_class_only'] and len(pkmn_types) == 2:
								pkmn_types = [pkmn_types[0]]

							# if we're balancing the classes, then duplicate it if has a single type
							if self.config['balance_classes'] and len(pkmn_types) == 1:
								pkmn_types = [pkmn_types[0], pkmn_types[0]]

							# if grouping classes, some smaller classes get mapped to new types
							if self.config['group_classes']:
								pkmn_types = [self.get_alternative_type(pkmn_type) for pkmn_type in pkmn_types]

							if len(pkmn_types) > 2 or len(pkmn_types) < 1:
								print("!!!! Found bad types: {}".format(pkmn_types))

							# this might save an image multiple times if a pokemon has 2 types
							assert len(pkmn_types) <= 2 and len(pkmn_types) >= 1


							for pkmn_type in pkmn_types:
								filepath = './{}/{}/{}_{}_{}_{}.jpg'.format(output_dir, pkmn_type, shiny_prefix, back_prefix, folder_prefix, pkmn_number)
								if debug:
									print("=== in classification output mode == type is: {}".format(pkmn_type))
									print("Save filepath is: {}, raw file name is: {}".format(filepath, file))
								

								self.type_count_dict[pkmn_type] += 1
								jpg.save(filepath)

						# basic mode, only save each pokemon once to the main dir
						else:
							if debug:
								print("Filename is: {}, raw file name is: {}".format(filepath, file))
								print("Tryign to process file: {}".format(file))
								print("New file path is: {}".format(filepath))
							filepath = './{}/{}{}_{}.jpg'.format(output_dir, shiny_prefix, folder_prefix, pkmn_number)
							jpg.save(filepath)


		print("======Processed {} files in total!".format(total_processed))
		
		if classification_mode:
			print("==== Using classification mode! Images by type dict is: ")
			pp.pprint(self.type_count_dict)

		

if __name__ == '__main__':
    print("Hit the main method!")
    processor = ImageProcessor()
    processor.process_images()