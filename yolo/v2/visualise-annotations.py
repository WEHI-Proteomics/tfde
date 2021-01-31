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
PIXELS_Y = 910
MZ_MIN = 100.0
MZ_MAX = 1700.0
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

# for drawing on tiles
TINT_COLOR = (0, 0, 0)  # Black
OPACITY = int(255 * 0.1)  # lower opacity means more transparent

# font paths for overlay labels
UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'

digits = '0123456789'

def calculate_feature_class(isotopes, charge):
    # for just one class
    return 0

def number_of_feature_classes():
    # for just one class
    return 1

def feature_names():
    # for just one class
    names = []
    names.append('peptide feature')
    return names


###########################
parser = argparse.ArgumentParser(description='Visualise tile list annotations.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tln','--tile_list_name', type=str, help='Name of the tile list.', required=True)
parser.add_argument('-as','--annotations_source', type=str, choices=['via','tfe','via-trained-predictions','tfe-trained-predictions'], help='Source of the annotations.', required=True)

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

# load the tile list metadata
TILE_LIST_METADATA_FILE_NAME = '{}/metadata.json'.format(TILE_LIST_DIR)
if os.path.isfile(TILE_LIST_METADATA_FILE_NAME):
    with open(TILE_LIST_METADATA_FILE_NAME) as json_file:
        tile_list_metadata = json.load(json_file)
        tile_list_df = pd.DataFrame(tile_list_metadata['tile_info'])
else:
    print("Could not find the tile list's metadata file: {}".format(TILE_LIST_METADATA_FILE_NAME))
    sys.exit(1)

# check the annotations directory
ANNOTATIONS_DIR = '{}/annotations-from-{}'.format(TILE_LIST_DIR, args.annotations_source)
if not os.path.exists(EXPERIMENT_DIR):
    print("The annotations directory is required but doesn't exist: {}".format(ANNOTATIONS_DIR))
    sys.exit(1)

# check the base overlay directory
OVERLAY_BASE_DIR = '{}/overlays'.format(ANNOTATIONS_DIR)
if os.path.exists(OVERLAY_BASE_DIR):
    shutil.rmtree(OVERLAY_BASE_DIR)
os.makedirs(OVERLAY_BASE_DIR)

# initialise the feature class names
feature_names = feature_names()

# load the font to use for labelling the overlays
if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

# load the annotations file(s)
annotations_file_list = sorted(glob.glob("{}/annotations-run-*-tile-*.json".format(ANNOTATIONS_DIR)))
for annotation_file_name in annotations_file_list:
    # load the annotations file
    print('processing {}'.format(annotation_file_name))
    with open(annotation_file_name) as file:
        annotations = json.load(file)

    for tile_key in list(annotations.keys()):
        tile_d = annotations[tile_key]
        tile_base_name = tile_d['file_attributes']['source']['tile']['base_name']
        tile_metadata = tile_list_df[tile_list_df.base_name == tile_base_name].iloc[0]
        tile_full_path = tile_metadata.full_path
        tile_id = tile_metadata.tile_id
        frame_id = tile_metadata.frame_id
        run_name = tile_metadata.run_name
        mz_lower = tile_metadata.mz_lower
        mz_upper = tile_metadata.mz_upper
        retention_time = tile_metadata.retention_time_secs
        tile_regions = tile_d['regions']

        # load the tile from the tile set
        img = Image.open(tile_full_path)

        # get a drawing context for the tile
        draw = ImageDraw.Draw(img)

        # draw the tile info
        draw.rectangle(xy=[(0, 0), (PIXELS_X, 36)], fill=TINT_COLOR+(OPACITY,), outline=None)
        draw.text((0, 0), 'annotations from {}'.format(args.annotations_source), font=feature_label_font, fill='lawngreen')
        draw.text((0, 12), '{} - {} m/z'.format(mz_lower, mz_upper), font=feature_label_font, fill='lawngreen')
        draw.text((0, 24), '{} secs'.format(retention_time), font=feature_label_font, fill='lawngreen')

        # process this tile if there are annotations for it
        if len(tile_regions) > 0:
            # render the annotations
            feature_coordinates = []
            for region in tile_regions:
                shape_attributes = region['shape_attributes']
                x = shape_attributes['x']
                y = shape_attributes['y']
                width = shape_attributes['width']
                height = shape_attributes['height']
                feature_class = 0
                # draw the bounding box
                draw.rectangle(xy=[(x, y), (x+width, y+height)], fill=None, outline='limegreen')
                # draw the feature class name
                draw.rectangle(xy=[(x, y-14), (x+width, y-2)], fill=TINT_COLOR+(OPACITY,), outline=None)
                draw.text((x, y-14), feature_names[feature_class], font=feature_label_font, fill='limegreen')

        # write the tile to the overlays directory
        TILE_DIR = '{}/run-{}/tile-{}'.format(OVERLAY_BASE_DIR, run_name, tile_id)
        if not os.path.exists(TILE_DIR):
            os.makedirs(TILE_DIR)
        tile_name = '{}/{}'.format(TILE_DIR, tile_base_name)
        img.save(tile_name)

print('wrote {} tiles to {}'.format(len(annotations.items()), OVERLAY_BASE_DIR))
