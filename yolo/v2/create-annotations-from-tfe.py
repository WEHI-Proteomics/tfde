import json
import os, shutil
import random
import argparse
import sqlite3
import pandas as pd
import sys
import numpy as np
import time


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

# allow for some buffer area around the features
MZ_BUFFER = 0.25
SCAN_BUFFER = 20

# annotate the apex in RT
RT_EXTENT_MAX = 1.0

SERVER_URL = "http://spectra-server-lb-1653892276.ap-southeast-2.elb.amazonaws.com"

def tile_id_from_mz(mz):
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"
    tile_id = int((mz - MZ_MIN) / MZ_PER_TILE)
    return tile_id

def tile_pixel_x_from_mz(mz):
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"
    pixel_x = int(((mz - MZ_MIN) % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return pixel_x

def tile_pixel_y_from_scan(scan):
    assert (scan >= tile_set_scan_lower) and (scan <= tile_set_scan_upper), "scan not in range"
    pixel_y = int(((scan - tile_set_scan_lower) / (tile_set_scan_upper - tile_set_scan_lower)) * PIXELS_Y)
    return pixel_y

# determine the mapping between the percolator index and the run file name
def get_percolator_run_mapping(mapping_file_name):
    df = pd.read_csv(mapping_file_name)
    mapping_l = [tuple(r) for r in df.to_numpy()]
    return mapping_l

def file_idx_for_run(run_name):
    result = None
    mapping_l = get_percolator_run_mapping(MAPPING_FILE_NAME)
    for m in mapping_l:
        if m[1] == run_name:
            result = m[0]
            break
    return result

def run_name_for_file_idx(file_idx):
    result = None
    mapping_l = get_percolator_run_mapping(MAPPING_FILE_NAME)
    for m in mapping_l:
        if m[0] == file_idx:
            result = m[1]
            break
    return result

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


# for the tiles in the tile list, create annotations based on the features extracted with TFE

parser = argparse.ArgumentParser(description='Set up a training set from raw tiles.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tln','--tile_list_name', type=str, help='Name of the tile list.', required=True)
parser.add_argument('-sl','--scan_lower', type=int, default=1, help='Lower bound of the scan range.', required=False)
parser.add_argument('-su','--scan_upper', type=int, default=920, help='Upper bound of the scan range.', required=False)
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

# check the mapping file
MAPPING_FILE_NAME = "{}/recalibrated-percolator-output/percolator-idx-mapping.csv".format(EXPERIMENT_DIR)
if not os.path.isfile(MAPPING_FILE_NAME):
    print("The mapping file is required but doesn't exist: {}".format(MAPPING_FILE_NAME))
    sys.exit(1)

# check the extracted features directory
EXTRACTED_FEATURES_DIR = "{}/extracted-features".format(EXPERIMENT_DIR)
if not os.path.exists(EXTRACTED_FEATURES_DIR):
    print("The extracted features directory is required but doesn't exist: {}".format(EXTRACTED_FEATURES_DIR))
    sys.exit(1)

# check the extracted features database
EXTRACTED_FEATURES_DB_NAME = "{}/extracted-features.sqlite".format(EXTRACTED_FEATURES_DIR)
if not os.path.isfile(EXTRACTED_FEATURES_DB_NAME):
    print("The extracted features database is required but doesn't exist: {}".format(EXTRACTED_FEATURES_DB_NAME))
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
    print('loading the tile list metadata from {}'.format(TILE_LIST_METADATA_FILE_NAME))
    with open(TILE_LIST_METADATA_FILE_NAME) as json_file:
        tile_list_metadata = json.load(json_file)
        tile_list_df = pd.DataFrame(tile_list_metadata['tile_info'])
        tile_set_name = tile_list_metadata['arguments']['tile_set_name']
        tile_list_metadata.clear()  # no longer needed
else:
    print("Could not find the tile list's metadata file: {}".format(TILE_LIST_METADATA_FILE_NAME))
    sys.exit(1)

# # load the tile set metadata
# TILE_SET_BASE_DIR = '{}/tiles'.format(EXPERIMENT_DIR)
# TILE_SET_METADATA_FILE_NAME = '{}/{}/metadata.json'.format(TILE_SET_BASE_DIR, tile_set_name)
# if os.path.isfile(TILE_SET_METADATA_FILE_NAME):
#     print('loading the tile set metadata from {}'.format(TILE_SET_METADATA_FILE_NAME))
#     with open(TILE_SET_METADATA_FILE_NAME) as json_file:
#         tile_set_metadata = json.load(json_file)
#         tile_set_scan_lower = tile_set_metadata['arguments']['scan_lower']
#         tile_set_scan_upper = tile_set_metadata['arguments']['scan_upper']
#         tile_set_metadata.clear()  # no longer needed
# else:
#     print("Could not find the tile list's metadata file: {}".format(TILE_SET_METADATA_FILE_NAME))
#     sys.exit(1)

tile_set_scan_lower = args.scan_lower
tile_set_scan_upper = args.scan_upper

# check the annotations directory
ANNOTATIONS_DIR = '{}/annotations-from-tfe'.format(TILE_LIST_DIR)
if os.path.exists(ANNOTATIONS_DIR):
    shutil.rmtree(ANNOTATIONS_DIR)
os.makedirs(ANNOTATIONS_DIR)
os.makedirs('{}/predictions'.format(ANNOTATIONS_DIR))

# find the extent of the tile list in m/z, RT, runs
tile_list_mz_lower = tile_list_df.mz_lower.min()
tile_list_mz_upper = tile_list_df.mz_upper.max()

rt_lower = tile_list_df.retention_time_secs.min()
rt_upper = tile_list_df.retention_time_secs.max()

tile_list_df['file_idx'] = tile_list_df.apply(lambda row: file_idx_for_run(row.run_name), axis=1)
file_idxs_l = list(tile_list_df.file_idx.unique())
run_names_l = list(tile_list_df.run_name.unique())

print('creating indexes if not already existing on {}'.format(EXTRACTED_FEATURES_DB_NAME))
db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
src_c = db_conn.cursor()
src_c.execute("create index if not exists idx_create_annotations_1 on features (file_idx, rt_apex, monoisotopic_mz_centroid)")
db_conn.close()

print("reading the extracted features for runs {} from {}".format(run_names_l, EXTRACTED_FEATURES_DB_NAME))
sequences_df_l = []
for idx,file_idx in enumerate(file_idxs_l):
    print('processing run {} of {}'.format(idx+1, len(file_idxs_l)))
    db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
    sequences_df = pd.read_sql_query('select sequence,charge,run_name,file_idx,monoisotopic_mz_centroid,number_of_isotopes,rt_apex,mono_rt_bounds,mono_scan_bounds,isotope_1_rt_bounds,isotope_1_scan_bounds,isotope_2_rt_bounds,isotope_2_scan_bounds,isotope_intensities_l from features where file_idx == {} and rt_apex >= {} and rt_apex <= {} and monoisotopic_mz_centroid >= {} and monoisotopic_mz_centroid <= {}'.format(file_idx, rt_lower, rt_upper, tile_list_mz_lower, tile_list_mz_upper), db_conn)
    db_conn.close()
    print("loaded {} extracted features from {}".format(len(sequences_df), EXTRACTED_FEATURES_DB_NAME))

    # unpack the feature extents
    print("unpacking the feature extents")
    sequences_df.mono_rt_bounds = sequences_df.apply(lambda row: json.loads(row.mono_rt_bounds), axis=1)
    sequences_df.mono_scan_bounds = sequences_df.apply(lambda row: json.loads(row.mono_scan_bounds), axis=1)

    sequences_df.isotope_1_rt_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_1_rt_bounds), axis=1)
    sequences_df.isotope_1_scan_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_1_scan_bounds), axis=1)

    sequences_df.isotope_2_rt_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_2_rt_bounds), axis=1)
    sequences_df.isotope_2_scan_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_2_scan_bounds), axis=1)

    # annotate the feature around the apex of the monoisotopic peak, up to a maximum extent
    sequences_df['rt_lower'] = sequences_df.apply(lambda row: np.max(row.mono_rt_bounds[0], row.rt_apex-RT_EXTENT_MAX), axis=1)
    sequences_df['rt_upper'] = sequences_df.apply(lambda row: np.min(row.mono_rt_bounds[1], row.rt_apex+RT_EXTENT_MAX), axis=1)

    sequences_df['scan_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in [row.mono_scan_bounds,row.isotope_1_scan_bounds,row.isotope_2_scan_bounds]]), axis=1)
    sequences_df['scan_upper'] = sequences_df.apply(lambda row: np.max([i[1] for i in [row.mono_scan_bounds,row.isotope_1_scan_bounds,row.isotope_2_scan_bounds]]), axis=1)

    # remove columns no longer required
    sequences_df.drop(columns=['mono_rt_bounds','mono_scan_bounds','isotope_1_rt_bounds','isotope_1_scan_bounds','isotope_2_rt_bounds','isotope_2_scan_bounds'], inplace=True)

    sequences_df.isotope_intensities_l = sequences_df.apply(lambda row: json.loads(row.isotope_intensities_l), axis=1)
    sequences_df['mz_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in row.isotope_intensities_l[0][4]]), axis=1)  # [0][4] refers to the isotope points of the monoisotope; i[0] refers to the m/z values
    sequences_df['mz_upper'] = sequences_df.apply(lambda row: np.max([i[0] for i in row.isotope_intensities_l[row.number_of_isotopes-1][4]]), axis=1)
    sequences_df.drop(columns=['isotope_intensities_l'], inplace=True)  # to save memory

    sequences_df_l.append(sequences_df)

# concatenate the list into a single dataframe
sequences_df = pd.concat(sequences_df_l)

# for each tile in the list, find its intersecting features and create annotations for them
tiles_d = {}
classes_d = {}
classes_overall_d = {}
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
    tile_url = '{}/tile/run/{}/tile/{}/frame/{}'.format(SERVER_URL, tile_run_name, tile_id, tile_frame_id)

    # find the features intersecting with this tile
    intersecting_features_df = sequences_df[(sequences_df.file_idx == tile_file_idx) & (sequences_df.rt_lower <= tile_rt) & (sequences_df.rt_upper >= tile_rt) & (sequences_df.mz_lower >= tile_mz_lower) & (sequences_df.mz_upper <= tile_mz_upper)]
    # calculate the coordinates for the annotations file
    regions_l = []
    for idx,feature in intersecting_features_df.iterrows():
        t = tile_id_from_mz(feature.mz_lower - MZ_BUFFER)
        x0_buffer = tile_pixel_x_from_mz(feature.mz_lower - MZ_BUFFER)
        if t < tile_id:
            x0_buffer = 1
        t = tile_id_from_mz(feature.mz_upper + MZ_BUFFER)
        x1_buffer = tile_pixel_x_from_mz(feature.mz_upper + MZ_BUFFER)
        if t > tile_id:
            x1_buffer = PIXELS_X
        y0 = feature.scan_lower
        y0_buffer = tile_pixel_y_from_scan(max((y0 - SCAN_BUFFER), tile_set_scan_lower))
        y1 = feature.scan_upper
        y1_buffer = tile_pixel_y_from_scan(min((y1 + SCAN_BUFFER), tile_set_scan_upper))
        w = x1_buffer - x0_buffer
        h = y1_buffer - y0_buffer
        charge = feature.charge
        isotopes = feature.number_of_isotopes
        # label the charge states we want to detect
        if (charge >= MIN_CHARGE) and (charge <= MAX_CHARGE):
            feature_class = calculate_feature_class(isotopes, charge)
            # count how many instances of each class we've seen overall
            if feature_class in classes_overall_d.keys():
                classes_overall_d[feature_class] += 1
            else:
                classes_overall_d[feature_class] = 1
            # add it to the list of annotation regions
            charge_str = '{}+'.format(charge)
            isotopes_str = '{}'.format(isotopes)
            region = {'shape_attributes':{'name':'rect','x':x0_buffer, 'y':y0_buffer, 'width':w, 'height':h}, 'region_attributes':{'charge':charge_str, 'isotopes':isotopes_str}}
            regions_l.append(region)
        # else:
        #     print("found a charge-{} feature - not included in the annotations".format(charge))

    tiles_key = 'run-{}-tile-{}'.format(tile_run_name, tile_id)
    if not tiles_key in tiles_d:
        tiles_d[tiles_key] = {}
    tiles_d[tiles_key]['frame-{}'.format(tile_frame_id)] = {'filename':tile_url, 'size':-1, 'regions':regions_l, 'file_attributes':{'source':{'annotation':'TFE','tile':{'experiment_name':args.experiment_name,'tile_set':tile_set_name,'base_name':tile_base_name}}}}

# write out a separate JSON file for the annotations for each run and tile
print('writing annotation files to {}'.format(ANNOTATIONS_DIR))
for key, value in tiles_d.items():
    run_name = key.split('run-')[1].split('-tile')[0]
    tile_id = int(key.split('tile-')[1])
    annotations_file_name = '{}/annotations-run-{}-tile-{}.json'.format(ANNOTATIONS_DIR, run_name, tile_id)
    with open(annotations_file_name, 'w') as outfile:
        json.dump(value, outfile)
print("wrote out {} annotations files to {}".format(len(tiles_d), ANNOTATIONS_DIR))

# display the object counts for each class
names = feature_names()
print('\ninstances:')
for c in sorted(classes_overall_d.keys()):
    print("{} objects: {}".format(names[c], classes_overall_d[c]))

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
