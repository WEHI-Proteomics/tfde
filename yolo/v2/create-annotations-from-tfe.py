import json
import os, shutil
import random
import argparse
import sqlite3
import pandas as pd
import sys
import numpy as np
import logging
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
    assert (scan >= scan_lower) and (scan <= scan_upper), "scan not in range"
    pixel_y = int(((scan - scan_lower) / (scan_upper - scan_lower)) * PIXELS_Y)
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


# for the tiles in the tile list, create annotations based on the features extracted with TFE

parser = argparse.ArgumentParser(description='Set up a training set from raw tiles.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tln','--tile_list_name', type=str, help='Name of the tile list.', required=True)
args = parser.parse_args()

# store the command line arguments as metadata for later reference
metadata = {'arguments':vars(args)}

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
    with open(TILE_LIST_METADATA_FILE_NAME) as json_file:
        tile_list_metadata = json.load(json_file)
        tile_list_df = pd.DataFrame(tile_list_metadata['tile_info'])
        tile_set_name = tile_list_metadata['arguments']['tile_set_name']
else:
    print("Could not find the tile list's metadata file: {}".format(TILE_LIST_METADATA_FILE_NAME))
    sys.exit(1)

# load the tile set metadata
TILE_SET_BASE_DIR = '{}/tiles'.format(EXPERIMENT_DIR)
TILE_SET_METADATA_FILE_NAME = '{}/{}/metadata.json'.format(TILE_SET_BASE_DIR, tile_set_name)
if os.path.isfile(TILE_SET_METADATA_FILE_NAME):
    with open(TILE_SET_METADATA_FILE_NAME) as json_file:
        tile_set_metadata = json.load(json_file)
        scan_lower = tile_set_metadata['arguments']['scan_lower']
        scan_upper = tile_set_metadata['arguments']['scan_upper']
else:
    print("Could not find the tile list's metadata file: {}".format(TILE_SET_METADATA_FILE_NAME))
    sys.exit(1)

# check the annotations directory
ANNOTATIONS_DIR = '{}/annotations-from-tfe'.format(TILE_LIST_DIR)
if os.path.exists(ANNOTATIONS_DIR):
    shutil.rmtree(ANNOTATIONS_DIR)
os.makedirs(ANNOTATIONS_DIR)
os.makedirs('{}/predictions'.format(ANNOTATIONS_DIR))

# set up logging
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
# set the file handler
file_handler = logging.FileHandler('{}/{}.log'.format(TILE_LIST_DIR, parser.prog))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# set the console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("{} info: {}".format(parser.prog, metadata))

# find the extent of the tile list in m/z, RT, runs
tile_list_mz_lower = tile_list_df.mz_lower.min()
tile_list_mz_upper = tile_list_df.mz_upper.max()

rt_lower = tile_list_df.retention_time_secs.min()
rt_upper = tile_list_df.retention_time_secs.max()

tile_list_df['file_idx'] = tile_list_df.apply(lambda row: file_idx_for_run(row.run_name), axis=1)
file_idxs = list(tile_list_df.file_idx.unique())
if len(file_idxs) == 1:
    file_idxs = '({})'.format(file_idxs[0])
else:
    file_idxs = '{}'.format(tuple(file_idxs))
run_names_l = list(tile_list_df.run_name.unique())

# only load the extracted features that will appear in the tile list
logger.info("reading the extracted features for runs {} from {}".format(run_names_l, EXTRACTED_FEATURES_DB_NAME))
db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
sequences_df = pd.read_sql_query('select sequence,charge,run_name,file_idx,monoisotopic_mz_centroid,number_of_isotopes,rt_apex,mono_rt_bounds,mono_scan_bounds,isotope_1_rt_bounds,isotope_1_scan_bounds,isotope_2_rt_bounds,isotope_2_scan_bounds,isotope_intensities_l from features where file_idx in {} and rt_apex >= {} and rt_apex <= {} and monoisotopic_mz_centroid >= {} and monoisotopic_mz_centroid <= {}'.format(file_idxs, rt_lower, rt_upper, tile_list_mz_lower, tile_list_mz_upper), db_conn)
db_conn.close()
logger.info("loaded {} extracted features from {}".format(len(sequences_df), EXTRACTED_FEATURES_DB_NAME))

# unpack the feature extents
logger.info("unpacking the feature extents")
sequences_df.mono_rt_bounds = sequences_df.apply(lambda row: json.loads(row.mono_rt_bounds), axis=1)
sequences_df.mono_scan_bounds = sequences_df.apply(lambda row: json.loads(row.mono_scan_bounds), axis=1)

sequences_df.isotope_1_rt_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_1_rt_bounds), axis=1)
sequences_df.isotope_1_scan_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_1_scan_bounds), axis=1)

sequences_df.isotope_2_rt_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_2_rt_bounds), axis=1)
sequences_df.isotope_2_scan_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_2_scan_bounds), axis=1)

sequences_df.isotope_intensities_l = sequences_df.apply(lambda row: json.loads(row.isotope_intensities_l), axis=1)

sequences_df['rt_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in [row.mono_rt_bounds,row.isotope_1_rt_bounds,row.isotope_2_rt_bounds]]), axis=1)
sequences_df['rt_upper'] = sequences_df.apply(lambda row: np.max([i[1] for i in [row.mono_rt_bounds,row.isotope_1_rt_bounds,row.isotope_2_rt_bounds]]), axis=1)

sequences_df['scan_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in [row.mono_scan_bounds,row.isotope_1_scan_bounds,row.isotope_2_scan_bounds]]), axis=1)
sequences_df['scan_upper'] = sequences_df.apply(lambda row: np.max([i[1] for i in [row.mono_scan_bounds,row.isotope_1_scan_bounds,row.isotope_2_scan_bounds]]), axis=1)

sequences_df['mz_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in row.isotope_intensities_l[0][4]]), axis=1)  # [0][4] refers to the isotope points of the monoisotope; i[0] refers to the m/z values
sequences_df['mz_upper'] = sequences_df.apply(lambda row: np.max([i[0] for i in row.isotope_intensities_l[row.number_of_isotopes-1][4]]), axis=1)

# for each tile in the list, find its intersecting features and create annotations for them
tiles_d = {}
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
        y0_buffer = tile_pixel_y_from_scan(max((y0 - SCAN_BUFFER), scan_lower))
        y1 = feature.scan_upper
        y1_buffer = tile_pixel_y_from_scan(min((y1 + SCAN_BUFFER), scan_upper))
        w = x1_buffer - x0_buffer
        h = y1_buffer - y0_buffer
        charge = feature.charge
        isotopes = feature.number_of_isotopes
        # label the charge states we want to detect
        if (charge >= MIN_CHARGE) and (charge <= MAX_CHARGE):
            feature_class = calculate_feature_class(isotopes, charge)
            charge_str = '{}+'.format(charge)
            isotopes_str = '{}'.format(isotopes)
            region = {'shape_attributes':{'name':'rect','x':x0_buffer, 'y':y0_buffer, 'width':w, 'height':h}, 'region_attributes':{'charge':charge_str, 'isotopes':isotopes_str}}
            regions_l.append(region)
        else:
            logger.info("found a charge-{} feature - not included in the annotations".format(charge))

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

metadata["processed"] = time.ctime()
metadata["processor"] = parser.prog
print("{} metadata: {}".format(parser.prog, metadata))
