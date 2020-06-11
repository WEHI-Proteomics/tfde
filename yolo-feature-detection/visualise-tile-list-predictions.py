import glob
import os
import argparse
import shutil
import sys
import json
from PIL import Image, ImageDraw, ImageFont

# this application assumes the predictions.json is in the tile list directory

# image dimensions
PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines

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

# font paths for overlay labels
UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'

###########################
parser = argparse.ArgumentParser(description='Visualise tile list predictions.')
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

# load the font to use for labelling the overlays
if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

# check the predictions directory
PREDICTIONS_DIR = '{}/predictions'.format(TILE_LIST_DIR)
if not os.path.exists(EXPERIMENT_DIR):
    print("The predictions directory is required but doesn't exist: {}".format(PREDICTIONS_DIR))
    sys.exit(1)

# load the predictions file
print('loading the predictions')
PREDICTIONS_FILE_NAME = '{}/tile-list-{}-predictions.json'.format(PREDICTIONS_DIR, args.tile_list_name)
if os.path.isfile(PREDICTIONS_FILE_NAME):
    with open(PREDICTIONS_FILE_NAME) as file:
        prediction_json = json.load(file)
else:
    print("The predictions file is required but does not exist: {}".format(PREDICTIONS_FILE_NAME))
    sys.exit(1)

# overlay each tile with the predictions
for prediction_idx in range(len(prediction_json)):
    tile_file_name = prediction_json[prediction_idx]['filename']
    base_name = os.path.basename(tile_file_name)
    splits = base_name.split('-')
    run_name = splits[1]
    tile_id = int(splits[5])
    print("processing {}".format(base_name))

    img = Image.open(tile_file_name)
    draw_predictions = ImageDraw.Draw(img)
    predictions = prediction_json[prediction_idx]['objects']
    for prediction in predictions:
        feature_class_name = prediction['name']
        feature_class = prediction['class_id']
        coordinates = prediction['relative_coordinates']
        x = (coordinates['center_x'] - (coordinates['width'] / 2)) * PIXELS_X
        y = (coordinates['center_y'] - (coordinates['height'] / 2)) * PIXELS_Y
        width = coordinates['width'] * PIXELS_X
        height = coordinates['height'] * PIXELS_Y
        # draw the bounding box
        draw_predictions.rectangle(xy=[(x, y), (x+width, y+height)], fill=None, outline=CLASS_COLOUR[feature_class])
        # draw the feature class name
        draw_predictions.text((x, y-12), feature_class_name, font=feature_label_font, fill=CLASS_COLOUR[feature_class])

    # write the annotated tile to the predictions directory
    tile_name = '{}/run-{}/tile-{}/{}'.format(PREDICTIONS_DIR, run_name, tile_id, base_name)
    img.save(tile_name)

print('wrote {} tiles to {}'.format(len(prediction_json), PREDICTIONS_DIR))
