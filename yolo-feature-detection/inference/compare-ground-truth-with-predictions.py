import json
from urllib.request import urlopen
from PIL import Image, ImageDraw
import pandas as pd
import sqlite3
import numpy as np
import glob
import os
import argparse
import time
import sys

PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines


#####################################
parser = argparse.ArgumentParser(description='Create composite tiles that show predictions and ground truth.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tn','--training_set_name', type=str, help='Name of the training set.', required=True)
parser.add_argument('-tid','--tile_id', type=int, help='Index of the tile for comparison of predictions.', required=True)
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# set up the training base directories
TRAINING_SET_BASE_DIR = '{}/training-sets/{}'.format(EXPERIMENT_DIR, args.training_set_name)
PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TRAINING_SET_BASE_DIR)
OVERLAY_FILES_DIR = '{}/overlays'.format(TRAINING_SET_BASE_DIR)

if not os.path.exists(TRAINING_SET_BASE_DIR):
    print("The training set directory is required but doesn't exist: {}".format(TRAINING_SET_BASE_DIR))
    sys.exit(1)

if not os.path.exists(PRE_ASSIGNED_FILES_DIR):
    print("The pre-assigned directory is required but doesn't exist: {}".format(PRE_ASSIGNED_FILES_DIR))
    sys.exit(1)

if not os.path.exists(OVERLAY_FILES_DIR):
    print("The overlay directory is required but doesn't exist: {}".format(OVERLAY_FILES_DIR))
    sys.exit(1)

PREDICTIONS_BASE_DIR = '{}/predictions'.format(EXPERIMENT_DIR)
TILE_PREDICTIONS_DIR = '{}/tile-{}'.format(PREDICTIONS_BASE_DIR, args.tile_id)
if not os.path.exists(TILE_PREDICTIONS_DIR):
    print("The tile predictions directory is required but doesn't exist: {}".format(TILE_PREDICTIONS_DIR))
    sys.exit(1)

prediction_json_file = '{}/predictions.json'.format(TILE_PREDICTIONS_DIR)
with open(prediction_json_file) as file:
    prediction_json = json.load(file)

for prediction_idx in range(len(prediction_json)):
    tile_file_name = prediction_json[prediction_idx]['filename']
    base_name = os.path.basename(tile_file_name)
    img = Image.open(tile_file_name)

    draw_predictions = ImageDraw.Draw(img)
    predictions = prediction_json[prediction_idx]['objects']
    for prediction in predictions:
        charge_state_label = prediction['name']
        coordinates = prediction['relative_coordinates']
        x = (coordinates['center_x'] - (coordinates['width'] / 2)) * PIXELS_X
        y = (coordinates['center_y'] - (coordinates['height'] / 2)) * PIXELS_Y
        width = coordinates['width'] * PIXELS_X
        height = coordinates['height'] * PIXELS_Y
        draw_predictions.rectangle(xy=[(x, y), (x+width, y+height)], fill=None, outline=(100,255,100,20))

    # write the annotated tile to the predictions directory
    draw_predictions.save('{}/{}'.format(TILE_PREDICTIONS_DIR, base_name))
