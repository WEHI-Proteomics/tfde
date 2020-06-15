# This application creates everything YOLO needs in the training set. The output base directory should be symlinked to ~/darket/data/peptides on the training machine.
import json
from PIL import Image, ImageDraw, ImageChops, ImageFont
import os, shutil
import random
import argparse
import sqlite3
import pandas as pd
import sys
import pickle
import numpy as np
import logging
import time
import glob

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

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# charge states of interest
MIN_CHARGE = 2
MAX_CHARGE = 4

# number of isotopes of interest
MIN_ISOTOPES = 3
MAX_ISOTOPES = 7

# in YOLO a small object is smaller than 16x16 @ 416x416 image size.
SMALL_OBJECT_W = SMALL_OBJECT_H = 16/416

# allow for some buffer area around the features
MZ_BUFFER = 0.25
SCAN_BUFFER = 20

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

# get the m/z extent for the specified tile ID
def mz_range_for_tile(tile_id):
    assert (tile_id >= 0) and (tile_id <= TILES_PER_FRAME-1), "tile_id not in range"

    mz_lower = MZ_MIN + (tile_id * MZ_PER_TILE)
    mz_upper = mz_lower + MZ_PER_TILE
    return (mz_lower, mz_upper)

# get the tile ID and the x pixel coordinate for the specified m/z
def tile_pixel_x_from_mz(mz):
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"

    tile_id = int((mz - MZ_MIN) / MZ_PER_TILE)
    pixel_x = int(((mz - MZ_MIN) % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return (tile_id, pixel_x)

def scan_coords_for_single_charge_region(mz_lower, mz_upper):
    scan_for_mz_lower = -1 * ((1.2 * mz_lower) - 1252)
    scan_for_mz_upper = -1 * ((1.2 * mz_upper) - 1252)
    return (scan_for_mz_lower,scan_for_mz_upper)

def calculate_feature_class(isotopes, charge):
    assert ((isotopes >= MIN_ISOTOPES) and (isotopes <= MAX_ISOTOPES)), "isotopes must be between {} and {}".format(MIN_ISOTOPES, MAX_ISOTOPES)
    assert ((charge >= MIN_CHARGE) and (charge <= MAX_CHARGE)), "charge must be between {} and {}".format(MIN_CHARGE, MAX_CHARGE)
    charge_idx = charge - MIN_CHARGE
    isotope_idx = isotopes - MIN_ISOTOPES
    feature_class = charge_idx * (MAX_ISOTOPES-MIN_ISOTOPES+1) + isotope_idx
    return feature_class

def number_of_feature_classes():
    return (MAX_ISOTOPES-MIN_ISOTOPES+1) * (MAX_CHARGE-MIN_CHARGE+1)

def feature_names():
    names = []
    for ch in range(MIN_CHARGE,MAX_CHARGE+1):
        for iso in range(MIN_ISOTOPES,MAX_ISOTOPES+1):
            names.append('charge-{}-isotopes-{}'.format(ch, iso))
    return names

########################################

# python ./otf-peak-detect/yolo-feature-detection/training/create-training-set-from-tfd.py -eb ~/Downloads/experiments -en dwm-test -rn 190719_Hela_Ecoli_1to1_01 -tidx 34

parser = argparse.ArgumentParser(description='Create a YOLO training set from one or more annotations files.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tln','--tile_list_name', type=str, help='Name of the tile list.', required=True)
parser.add_argument('-as','--annotations_source', type=str, choices=['via','tfe','predictions'], help='Source of the annotations.', required=True)
args = parser.parse_args()

# store the command line arguments as metadata for later reference
tile_set_metadata = {'arguments':vars(args)}

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
else:
    print("Could not find the tile list's metadata file: {}".format(TILE_LIST_METADATA_FILE_NAME))
    sys.exit(1)

# check the raw tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, tile_list_metadata['arguments']['tile_set_name'])
if not os.path.exists(TILES_BASE_DIR):
    print("The raw tiles base directory is required but does not exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

# check the annotations directory
ANNOTATIONS_DIR = '{}/annotations-from-{}'.format(TILE_LIST_DIR, args.annotations_source)
if not os.path.exists(EXPERIMENT_DIR):
    print("The annotations directory is required but doesn't exist: {}".format(ANNOTATIONS_DIR))
    sys.exit(1)

# set up the training set directories
TRAINING_SET_BASE_DIR = '{}/training-set'.format(ANNOTATIONS_DIR)
if os.path.exists(TRAINING_SET_BASE_DIR):
    shutil.rmtree(TRAINING_SET_BASE_DIR)
os.makedirs(TRAINING_SET_BASE_DIR)

PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TRAINING_SET_BASE_DIR)
if os.path.exists(PRE_ASSIGNED_FILES_DIR):
    shutil.rmtree(PRE_ASSIGNED_FILES_DIR)
os.makedirs(PRE_ASSIGNED_FILES_DIR)

OVERLAY_FILES_DIR = '{}/overlays'.format(TRAINING_SET_BASE_DIR)
if os.path.exists(OVERLAY_FILES_DIR):
    shutil.rmtree(OVERLAY_FILES_DIR)
os.makedirs(OVERLAY_FILES_DIR)

MASK_FILES_DIR = '{}/masks'.format(TRAINING_SET_BASE_DIR)
if os.path.exists(MASK_FILES_DIR):
    shutil.rmtree(MASK_FILES_DIR)
os.makedirs(MASK_FILES_DIR)

SETS_BASE_DIR = '{}/sets'.format(TRAINING_SET_BASE_DIR)
if os.path.exists(SETS_BASE_DIR):
    shutil.rmtree(SETS_BASE_DIR)
os.makedirs(SETS_BASE_DIR)

TRAIN_SET_DIR = '{}/train'.format(SETS_BASE_DIR)
if os.path.exists(TRAIN_SET_DIR):
    shutil.rmtree(TRAIN_SET_DIR)
os.makedirs(TRAIN_SET_DIR)

VAL_SET_DIR = '{}/validation'.format(SETS_BASE_DIR)
if os.path.exists(VAL_SET_DIR):
    shutil.rmtree(VAL_SET_DIR)
os.makedirs(VAL_SET_DIR)

TEST_SET_DIR = '{}/test'.format(SETS_BASE_DIR)
if os.path.exists(TEST_SET_DIR):
    shutil.rmtree(TEST_SET_DIR)
os.makedirs(TEST_SET_DIR)

# set up logging
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
# set the file handler
file_handler = logging.FileHandler('{}/{}.log'.format(TRAINING_SET_BASE_DIR, parser.prog))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# set the console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("{} info: {}".format(parser.prog, tile_set_metadata))

# determine tile allocation proportions
train_proportion = 0.9
val_proportion = 0.1
test_proportion = 0.0  # there's no need for a test set because mAP is measured on the validation set
logger.info("set proportions: train {}, validation {}, test {}".format(train_proportion, val_proportion, test_proportion))

# load the font to use for labelling the overlays
if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

# copy the tiles from the tile set to the pre-assigned directory, create its overlay and label text file
classes_d = {}
small_objects = 0
total_objects = 0
tile_list = []
objects_per_tile = []
tile_metadata_l = []
digits = '0123456789'

# process all the annotations files
annotations_file_list = sorted(glob.glob("{}/annotations-run-*-tile-*.json".format(ANNOTATIONS_DIR)))
for annotation_file_name in annotations_file_list:
    # load the annotations file
    print('processing {}'.format(annotation_file_name))
    with open(annotation_file_name) as file:
        annotations = json.load(file)

    # for each tile in the annotations
    for tile in list(annotations.items()):
        tile_d = tile[1]
        tile_base_name = tile_d['source']['tile']['base_name']
        tile_full_path = '{}/{}'.format(TILES_BASE_DIR, tile_base_name)

        # determine the frame_id and tile_id
        splits = tile_base_name.split('-')
        run_name = splits[1]
        frame_id = int(splits[3])
        tile_id = int(splits[5].split('.')[0])

        # copy the tile to the pre-assigned directory
        destination_name = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, tile_base_name)
        shutil.copyfile(tile_full_path, destination_name)

        # create a feature mask
        mask_im_array = np.zeros([PIXELS_Y+1, PIXELS_X+1, 3], dtype=np.uint8)
        mask = Image.fromarray(mask_im_array.astype('uint8'), 'RGB')
        mask_draw = ImageDraw.Draw(mask)

        # fill in the charge-1 area that we want to preserve
        tile_mz_lower,tile_mz_upper = mz_range_for_tile(tile_id)
        mask_region_y_left,mask_region_y_right = scan_coords_for_single_charge_region(tile_mz_lower, tile_mz_upper)
        mask_draw.polygon(xy=[(0,0), (PIXELS_X,0), (PIXELS_X,mask_region_y_right), (0,mask_region_y_left)], fill='white', outline='white')

        # set up the YOLO annotatiions file
        annotations_filename = '{}.txt'.format(os.path.splitext(tile_base_name)[0])
        annotations_path = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, annotations_filename)

        # process each annotation for the tile
        number_of_objects_this_tile = 0
        tile_regions = tile_d['regions']
        if len(tile_regions) > 0:
            # load the tile from the tile set
            print("processing {}".format(tile_base_name))
            img = Image.open(tile_full_path)

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
                # calculate the YOLO coordinates for the text file
                yolo_x = (x + (width / 2)) / PIXELS_X
                yolo_y = (y + (height / 2)) / PIXELS_Y
                yolo_w = width / PIXELS_X
                yolo_h = height / PIXELS_Y
                # determine the attributes of this feature
                region_attributes = region['region_attributes']
                charge = int(''.join(c for c in region_attributes['charge'] if c in digits))
                isotopes = int(region_attributes['isotopes'])
                # label the charge states we want to detect
                if (charge >= MIN_CHARGE) and (charge <= MAX_CHARGE):
                    feature_class = calculate_feature_class(isotopes, charge)
                    # keep record of how many instances of each class
                    if feature_class in classes_d.keys():
                        classes_d[feature_class] += 1
                    else:
                        classes_d[feature_class] = 1
                    # add it to the list
                    feature_coordinates.append(("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(feature_class, yolo_x, yolo_y, yolo_w, yolo_h)))
                    # draw the rectangle on the overlay
                    draw.rectangle(xy=[(x, y), (x+width, y+height)], fill=None, outline=CLASS_COLOUR[feature_class])
                    # draw the feature class name
                    draw.rectangle(xy=[(x, y-12), (x+width, y)], fill='darkgrey', outline=None)
                    draw.text((x, y-12), feature_names()[feature_class], font=feature_label_font, fill=CLASS_COLOUR[feature_class])
                    # draw the mask
                    mask_draw.rectangle(xy=[(x, y), (x+width, y+height)], fill='white', outline='white')
                    # keep record of the 'small' objects
                    total_objects += 1
                    if (yolo_w <= SMALL_OBJECT_W) or (yolo_h <= SMALL_OBJECT_H):
                        small_objects += 1
                    # keep track of the number of objects in this tile
                    number_of_objects_this_tile += 1
                else:
                    logger.info("found a charge-{} feature - not included in the training set".format(charge))

            # finish drawing the annotations on the overlay
            del draw
            # write the overlay tile
            img.save('{}/{}'.format(OVERLAY_FILES_DIR, tile_base_name))

        # finish drawing the mask
        del mask_draw
        # ...and save the mask
        mask.save('{}/{}'.format(MASK_FILES_DIR, tile_base_name))

        # apply the mask to the tile
        img = Image.open("{}/{}".format(PRE_ASSIGNED_FILES_DIR, tile_base_name))
        masked_tile = ImageChops.multiply(img, mask)
        masked_tile.save("{}/{}".format(PRE_ASSIGNED_FILES_DIR, tile_base_name))

        # write the annotations text file for this tile
        with open(annotations_path, 'w') as f:
            for item in feature_coordinates:
                f.write("%s\n" % item)

        # keep stats of the objects on each tile
        objects_per_tile.append((tile_id, frame_id, number_of_objects_this_tile))

# at this point we have all the referenced tiles in the pre-assigned directory, the charge-1 cloud and all labelled features are masked, and each tile has an annotations file

# display the object counts for each class
names = feature_names()
for c in sorted(classes_d.keys()):
    logger.info("{} objects: {}".format(names[c], classes_d[c]))
if total_objects > 0:
    logger.info("{} out of {} objects ({}%) are small.".format(small_objects, total_objects, round(small_objects/total_objects*100,1)))
else:
    logger.info("note: there are no objects on these tiles")

# display the number of objects per tile
objects_per_tile_df = pd.DataFrame(objects_per_tile, columns=['tile_id','frame_id','number_of_objects'])
objects_per_tile_df.to_pickle('{}/objects_per_tile_df.pkl'.format(TRAINING_SET_BASE_DIR))
logger.info("There are {} tiles with no objects.".format(len(objects_per_tile_df[objects_per_tile_df.number_of_objects == 0])))
logger.info("On average there are {} objects per tile.".format(round(np.mean(objects_per_tile_df.number_of_objects),1)))

# assign the tiles to the training sets
train_n = round(len(tile_list) * train_proportion)
val_n = round(len(tile_list) * val_proportion)

train_set = random.sample(tile_list, train_n)
val_test_set = list(set(tile_list) - set(train_set))
val_set = random.sample(val_test_set, val_n)
test_set = list(set(val_test_set) - set(val_set))

logger.info("tile counts - train {}, validation {}, test {}".format(len(train_set), len(val_set), len(test_set)))
number_of_classes = MAX_CHARGE - MIN_CHARGE + 1
max_batches = max(6000, max(2000*number_of_classes, len(train_set)))  # recommendation from AlexeyAB
logger.info("set max_batches={}, steps={},{},{},{}".format(max_batches, int(0.4*max_batches), int(0.6*max_batches), int(0.8*max_batches), int(0.9*max_batches)))

# copy the training set tiles and their annotation files
print("copying the training set to {}".format(TRAIN_SET_DIR))
for file_pair in train_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(TRAIN_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(TRAIN_SET_DIR, file_pair[1]))

# copy the validation set tiles and their annotation files
print("copying the validation set to {}".format(VAL_SET_DIR))
for file_pair in val_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(VAL_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(VAL_SET_DIR, file_pair[1]))

# copy the test set tiles and their annotation files
print("copying the test set to {}".format(TEST_SET_DIR))
for file_pair in test_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(TEST_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(TEST_SET_DIR, file_pair[1]))

# create obj.names, for copying to ./darknet/data, with the object names, each one on a new line
LOCAL_NAMES_FILENAME = "{}/peptides-obj.names".format(TRAINING_SET_BASE_DIR)
print("writing {}".format(LOCAL_NAMES_FILENAME))

# class labels
with open(LOCAL_NAMES_FILENAME, 'w') as f:
    for name in feature_names():
        f.write("{}\n".format(name))

# create obj.data, for copying to ./darknet/data
LOCAL_DATA_FILENAME = "{}/peptides-obj.data".format(TRAINING_SET_BASE_DIR)
print("writing {}".format(LOCAL_DATA_FILENAME))

with open(LOCAL_DATA_FILENAME, 'w') as f:
    f.write("classes={}\n".format(number_of_feature_classes()))
    f.write("train=data/peptides/train.txt\n")
    f.write("valid=data/peptides/validation.txt\n")
    f.write("names=data/peptides/peptides-obj.names\n")
    f.write("backup=backup/\n")

# create the file list for each set
with open('{}/train.txt'.format(TRAINING_SET_BASE_DIR), 'w') as f:
    for file_pair in train_set:
        f.write('data/peptides/sets/train/{}\n'.format(file_pair[0]))

with open('{}/validation.txt'.format(TRAINING_SET_BASE_DIR), 'w') as f:
    for file_pair in val_set:
        f.write('data/peptides/sets/validation/{}\n'.format(file_pair[0]))

with open('{}/test.txt'.format(TRAINING_SET_BASE_DIR), 'w') as f:
    for file_pair in test_set:
        f.write('data/peptides/sets/test/{}\n'.format(file_pair[0]))

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
