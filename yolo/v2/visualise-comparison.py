import json
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import argparse
import time
import sys
import shutil

# font paths for overlay labels
UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'

#####################################
parser = argparse.ArgumentParser(description='Create composite tiles that show predictions and ground truth.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tln','--tile_list_name', type=str, help='Name of the tile list.', required=True)
parser.add_argument('-asa','--annotations_source_a', type=str, choices=['via','tfe','predictions'], help='Source of the A annotations.', required=True)
parser.add_argument('-asb','--annotations_source_b', type=str, choices=['via','tfe','predictions'], help='Source of the B annotations.', required=True)
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

# the directory for this tile list
TILE_LIST_BASE_DIR = '{}/tile-lists'.format(EXPERIMENT_DIR)
TILE_LIST_DIR = '{}/{}'.format(TILE_LIST_BASE_DIR, args.tile_list_name)
if not os.path.exists(TILE_LIST_DIR):
    print("The tile list directory is required but doesn't exist: {}".format(TILE_LIST_DIR))
    sys.exit(1)

# load the tile list metadata
TILE_LIST_METADATA_FILE_NAME = '{}/metadata.json'.format(TILE_LIST_DIR)
if os.path.isfile(TILE_LIST_METADATA_FILE_NAME):
    with open(TILE_LIST_METADATA_FILE_NAME) as json_file:
        tile_list_metadata = json.load(json_file)
        tile_list_df = pd.DataFrame(tile_list_metadata['tile_info'])

# load the font to use for labelling the overlays
if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

# check the annotations A directory
ANNOTATIONS_A_DIR = '{}/annotations-from-{}'.format(TILE_LIST_DIR, args.annotations_source_a)
if not os.path.exists(ANNOTATIONS_A_DIR):
    print("The annotations directory is required but doesn't exist: {}".format(ANNOTATIONS_A_DIR))
    sys.exit(1)

# check the base overlay A directory
OVERLAY_A_BASE_DIR = '{}/overlays'.format(ANNOTATIONS_A_DIR)
if not os.path.exists(OVERLAY_A_BASE_DIR):
    print("The overlay A directory is required but doesn't exist: {}".format(OVERLAY_A_BASE_DIR))
    sys.exit(1)

# check the annotations B directory
ANNOTATIONS_B_DIR = '{}/annotations-from-{}'.format(TILE_LIST_DIR, args.annotations_source_b)
if not os.path.exists(ANNOTATIONS_B_DIR):
    print("The annotations directory is required but doesn't exist: {}".format(ANNOTATIONS_B_DIR))
    sys.exit(1)

# check the base overlay B directory
OVERLAY_B_BASE_DIR = '{}/overlays'.format(ANNOTATIONS_B_DIR)
if not os.path.exists(OVERLAY_B_BASE_DIR):
    print("The overlay A directory is required but doesn't exist: {}".format(OVERLAY_B_BASE_DIR))
    sys.exit(1)

# check the composite tiles directory
COMPOSITE_TILE_DIR = '{}/composite'.format(TILE_LIST_DIR)
if os.path.exists(COMPOSITE_TILE_DIR):
    shutil.rmtree(COMPOSITE_TILE_DIR)
os.makedirs(COMPOSITE_TILE_DIR)

# load the font to use for labelling the overlays
if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

# for each tile in the tile list, find its A and B overlay, and create a composite of them
composite_tile_count = 0
for idx,row in enumerate(tile_list_df.itertuples()):
    # attributes of this tile
    tile_id = row.tile_id
    tile_frame_id = row.frame_id
    tile_rt = row.retention_time_secs
    tile_mz_lower = row.mz_lower
    tile_mz_upper = row.mz_upper
    tile_file_idx = row.file_idx
    tile_run_name = row.run_name
    tile_base_name = row.base_name

    overlay_a_name = '{}/{}'.format(OVERLAY_A_BASE_DIR, tile_base_name)
    overlay_b_name = '{}/{}'.format(OVERLAY_B_BASE_DIR, tile_base_name)
    composite_name_dir = '{}/{}/tile-{}'.format(COMPOSITE_TILE_DIR, tile_run_name, tile_id)
    composite_name = '{}/composite-{}'.format(composite_name_dir, tile_base_name)

    # make the composite
    if os.path.isfile(overlay_a_name) and os.path.isfile(overlay_b_name):
        if not os.path.exists(composite_name_dir):
            os.makedirs(composite_name_dir)
        cmd = "convert {} {} +append -background darkgrey -splice 10x0+910+0 {}".format(overlay_a_name, overlay_b_name, composite_name)
        os.system(cmd)
        composite_tile_count += 1
    else:
        if not os.path.isfile(overlay_a_name):
            print('could not find {} in {}'.format(tile_base_name, OVERLAY_A_BASE_DIR))
        if not os.path.isfile(overlay_b_name):
            print('could not find {} in {}'.format(tile_base_name, OVERLAY_B_BASE_DIR))

print('wrote {} composite tiles to {}'.format(composite_tile_count, COMPOSITE_TILE_DIR))
