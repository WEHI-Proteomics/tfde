import glob
import os
import argparse
import shutil
import sys
import json
import pandas as pd
from urllib import parse

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

args = parser.parse_args()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))


ANNOTATIONS_DIR = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/annotations/tile-sets/3-june-no-interp'


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
        tile_url = tile_d['filename']

        path_split = parse.urlsplit(tile_url).path.split('/')
        run_name = path_split[3]
        tile_id = int(path_split[5])
        frame_id = int(path_split[7])

        tile_regions = tile_d['regions']
        # process this tile if there are annotations for it
        if len(tile_regions) > 0:
            for region in tile_regions:
                # determine the attributes of this feature
                region_attributes = region['region_attributes']
                charge = int(''.join(c for c in region_attributes['charge'] if c in digits))
                isotopes = int(region_attributes['isotopes'])
                feature_class = calculate_feature_class(isotopes, charge)
                feature_class_name = feature_names[feature_class]
                features_l.append((tile_id,frame_id,run_name,charge,isotopes,feature_class,feature_class_name))

# write out the feature index
df = pd.DataFrame(features_l, columns=['tile_id','frame_id','run_name','charge','isotopes','feature_class','feature_class_name'])
for run_group_name,run_group_df in df.groupby(['run_name'], as_index=False):
    index_file_name = '{}/run-{}-feature-index.txt'.format(ANNOTATIONS_DIR, run_group_name)
    print('writing the feature index {}'.format(index_file_name))
    tfile = open(index_file_name, 'w')
    for group_name,group_df in run_group_df.groupby(['feature_class','charge','isotopes'], as_index=False):
        sorted_group_df = group_df.sort_values(by=['run_name','tile_id','frame_id'], ascending=True, inplace=False)
        tfile.write('\n\nfeature class {}, charge {}, isotopes {}\n\n'.format(group_name[0], group_name[1], group_name[2]))
        tfile.write(sorted_group_df.to_string(columns=['run_name','tile_id','frame_id'], index=False))
    tfile.close()

# which run,tile have the most instances of each feature class
top_5_file_name = '{}/top-5-feature-index.txt'.format(ANNOTATIONS_DIR)
tfile = open(top_5_file_name, 'w')
for feature_class_group_name,feature_class_group_df in df.groupby(['feature_class'], as_index=False):
    feature_class_name = feature_class_group_df.iloc[0].feature_class_name
    counts_l = []
    for run_group_name,run_group_df in feature_class_group_df.groupby(['run_name','tile_id'], as_index=False):
        counts_l.append((run_group_name[0],run_group_name[1],len(run_group_df)))
    counts_df = pd.DataFrame(counts_l, columns=['run_name','tile_id','count'])
    counts_df.sort_values(by=['count'], ascending=False, inplace=True)
    tfile.write('\n\nfeature class {}\n'.format(feature_class_name))
    tfile.write(counts_df.head(5).to_string(columns=['run_name','tile_id','count'], index=False))
tfile.close()
