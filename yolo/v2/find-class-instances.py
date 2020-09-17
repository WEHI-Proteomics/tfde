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

digits = '0123456789'

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
if not os.path.exists(ANNOTATIONS_DIR):
    print("The annotations directory is required but doesn't exist: {}".format(ANNOTATIONS_DIR))
    sys.exit(1)

# initialise the feature class names
feature_names = feature_names()

# load the annotations file(s)
features_l = []
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
        print("processing {}".format(tile_base_name))
        # process this tile if there are annotations for it
        if len(tile_regions) > 0:
            for region in tile_regions:
                # determine the attributes of this feature
                region_attributes = region['region_attributes']
                charge = int(''.join(c for c in region_attributes['charge'] if c in digits))
                isotopes = int(region_attributes['isotopes'])
                feature_class = calculate_feature_class(isotopes, charge)
                feature_class_name = feature_names[feature_class]
                features_l.append((tile_base_name,tile_id,frame_id,run_name,mz_lower,mz_upper,retention_time,charge,isotopes,feature_class,feature_class_name))

# write out the feature index
df = pd.DataFrame(features_l, columns=['tile_base_name','tile_id','frame_id','run_name','mz_lower','mz_upper','retention_time','charge','isotopes','feature_class','feature_class_name'])
for run_group_name,run_group_df in df.groupby(['run_name'], as_index=False):
    index_file_name = '{}/run-{}-feature-index.txt'.format(ANNOTATIONS_DIR, run_group_name)
    print('writing the feature index {}'.format(index_file_name))
    tfile = open(index_file_name, 'w')
    for group_name,group_df in run_group_df.groupby(['feature_class','charge','isotopes'], as_index=False):
        sorted_group_df = group_df.sort_values(by=['run_name','tile_id','frame_id'], ascending=True, inplace=False)
        tfile.write('\n\nfeature class {}, charge {}, isotopes {}\n\n'.format(group_name[0], group_name[1], group_name[2]))
        tfile.write(sorted_group_df.to_string(columns=['run_name','tile_id','frame_id'], index=False))
    tfile.close()
