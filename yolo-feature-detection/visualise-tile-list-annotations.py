import glob
import os
import argparse
import shutil
import sys
import json
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# this application assumes the predictions.json is in the tile list directory

PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines
MZ_MIN = 100.0
MZ_MAX = 1700.0
SCAN_MAX = PIXELS_Y
SCAN_MIN = 1
MZ_PER_TILE = 18.0
TILES_PER_FRAME = int((MZ_MAX - MZ_MIN) / MZ_PER_TILE) + 1
MIN_TILE_IDX = 0
MAX_TILE_IDX = TILES_PER_FRAME-1

# charge states of interest
MIN_CHARGE = 2
MAX_CHARGE = 4

# number of isotopes of interest
MIN_ISOTOPES = 3
MAX_ISOTOPES = 7

# define the feature class colours
CLASS_COLOUR = [
    '#132580',  # class 0
    '#4b27ff',  # class 1
    '#9427ff',  # class 2
    '#ff27fb',  # class 3
    '#ff2781',  # class 4
    '#ff3527',  # class 5
    '#ff6727',  # class 6
    '#ff9a27',  # class 7
    '#ffc127',  # class 8
    '#ffe527',  # class 9
    '#e0ff27',  # class 10
    '#63da21',  # class 11
    '#27ff45',  # class 12
    '#21daa5',  # class 13
    '#135e80'   # class 14
]

def calculate_feature_class(isotopes, charge):
    assert ((isotopes >= MIN_ISOTOPES) and (isotopes <= MAX_ISOTOPES)), "isotopes must be between {} and {}".format(MIN_ISOTOPES, MAX_ISOTOPES)
    assert ((charge >= MIN_CHARGE) and (charge <= MAX_CHARGE)), "charge must be between {} and {}".format(MIN_CHARGE, MAX_CHARGE)
    charge_idx = charge - MIN_CHARGE
    isotope_idx = isotopes - MIN_ISOTOPES
    feature_class = charge_idx * (MAX_ISOTOPES-MIN_ISOTOPES+1) + isotope_idx
    return feature_class

def feature_names():
    names = []
    for ch in range(MIN_CHARGE,MAX_CHARGE+1):
        for iso in range(MIN_ISOTOPES,MAX_ISOTOPES+1):
            names.append('charge-{}-isotopes-{}'.format(ch, iso))
    return names

# font paths for overlay labels
UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'


###########################
parser = argparse.ArgumentParser(description='Visualise tile list annotations.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tln','--tile_list_name', type=str, help='Name of the tile list.', required=True)

args = parser.parse_args()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# the directory for this tile list
TILE_LIST_BASE_DIR = '{}/tile-lists'.format(EXPERIMENT_DIR)
TILE_LIST_DIR = '{}/{}'.format(TILE_LIST_BASE_DIR, args.tile_list_name)
if not os.path.exists(TILE_LIST_DIR):
    print("The tile list directory is required but doesn't exist: {}".format(TILE_LIST_DIR))
    sys.exit(1)

# the tile list file
TILE_LIST_FILE_NAME = '{}/tile-list-{}.txt'.format(TILE_LIST_DIR, args.tile_list_name)
if not os.path.isfile(TILE_LIST_FILE_NAME):
    print("The tile list is required but doesn't exist: {}".format(TILE_LIST_FILE_NAME))
    sys.exit(1)
else:
    with open(TILE_LIST_FILE_NAME) as f:
        tile_file_names_l = f.read().splitlines()
    tile_file_names_df = pd.DataFrame(tile_file_names_l, columns=['full_path'])
    tile_file_names_df['base_name'] = tile_file_names_df.apply(lambda row: os.path.basename(row.full_path), axis=1)

# load the font to use for labelling the overlays
if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

# check the annotations directory
ANNOTATIONS_DIR = '{}/annotations'.format(TILE_LIST_DIR)
if not os.path.exists(EXPERIMENT_DIR):
    print("The annotations directory is required but doesn't exist: {}".format(ANNOTATIONS_DIR))
    sys.exit(1)

# load the annotations file
print('loading the annotations')
ANNOTATIONS_FILE_NAME = '{}/tile-list-{}-annotations.json'.format(ANNOTATIONS_DIR, args.tile_list_name)
if os.path.isfile(ANNOTATIONS_FILE_NAME):
    with open(ANNOTATIONS_FILE_NAME) as file:
        annotations = json.load(file)
else:
    print("The annotations file is required but does not exist: {}".format(ANNOTATIONS_FILE_NAME))
    sys.exit(1)

feature_names = feature_names()

digits = '0123456789'
for tile in list(annotations.items()):
    tile_d = tile[1]
    tile_regions = tile_d['regions']
    # process this tile if there are annotations for it
    if len(tile_regions) > 0:
        # load the tile
        tile_url = tile_d['filename']  # this is the URL so we need to download it

        # determine the frame_id and tile_id
        splits = tile_url.split('/')
        run_name = splits[5]
        tile_id = int(splits[7])
        frame_id = int(splits[9])

        base_name = "run-{}-frame-{}-tile-{}.png".format(run_name, frame_id, tile_id)
        bn = tile_file_names_df[tile_file_names_df.base_name == base_name]
        if len(bn) == 1:
            full_path = bn.iloc[0].full_path
        else:
            print('encountered a tile in the annotations that is not in the tile list: {}'.format(base_name))
            sys.exit(1)

        # load the tile from the tile set
        print("processing {}".format(base_name))
        img = Image.open(full_path)

        # get a drawing context for the tile
        draw = ImageDraw.Draw(img)

        # render the annotations
        feature_coordinates = []
        for region in tile_regions:
            shape_attributes = region['shape_attributes']
            x = shape_attributes['x']
            y = shape_attributes['y']
            width = shape_attributes['width']
            height = shape_attributes['height']
            # determine the attributes of this feature
            region_attributes = region['region_attributes']
            charge = int(''.join(c for c in region_attributes['charge'] if c in digits))
            isotopes = int(region_attributes['isotopes'])
            feature_class = calculate_feature_class(isotopes, charge)
            # draw the bounding box
            draw.rectangle(xy=[(x, y), (x+width, y+height)], fill=None, outline=CLASS_COLOUR[feature_class])
            # draw the feature class name
            draw.text((x, y-12), feature_names[feature_class], font=feature_label_font, fill=CLASS_COLOUR[feature_class])

        # write the tile to the annotations directory
        tile_name = '{}/{}'.format(ANNOTATIONS_DIR, base_name)
        img.save(tile_name)

print('wrote {} tiles to {}'.format(len(annotations.items()), ANNOTATIONS_DIR))
