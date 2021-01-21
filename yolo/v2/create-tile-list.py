import glob
import os
import argparse
import shutil
import sys
import time
import json
import pandas as pd


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

# get the m/z extent for the specified tile ID
def mz_range_for_tile(tile_id):
    assert (tile_id >= 0) and (tile_id <= TILES_PER_FRAME-1), "tile_id not in range"

    mz_lower = MZ_MIN + (tile_id * MZ_PER_TILE)
    mz_upper = mz_lower + MZ_PER_TILE
    return (mz_lower, mz_upper)


parser = argparse.ArgumentParser(description='Create a tile list from a tile set.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tln','--tile_list_name', type=str, help='Name of the tile list.', required=True)
parser.add_argument('-tid','--tile_id_list', type=str, help='IDs of the tiles to use for the list. Can specify several ranges (e.g. 10-20,21-30,31-40), a single range (e.g. 10-24), individual tiles (e.g. 34,56,32), or a single tile (e.g. 54). Tile IDs must be between 0 and 89 inclusive', required=True)
parser.add_argument('-rn','--run_names', nargs='+', type=str, help='Space-separated names of runs to include.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)

args = parser.parse_args()

# store the arguments as metadata for later reference
metadata = {'arguments':vars(args)}

# parse the tile indexes
indexes_l = []
for item in args.tile_id_list.replace(" ", "").split(','):
    index_range = item.split('-')
    if all([i.isnumeric() for i in index_range]):  # only use the range if it's valid
        index_range = [int(i) for i in index_range]
        if len(index_range) == 2:
            index_lower = min(index_range)
            index_upper = max(index_range)
            indexes_l.append([i for i in range(index_lower, index_upper+1)])
        else:
            indexes_l.append(index_range)
indexes_l = [item for sublist in indexes_l for item in sublist]
if len(indexes_l) == 0:
    print("Need to specify at least one tile ID to include: {}".format(args.tile_id_list))
    sys.exit(1)

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the raw tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.tile_set_name)
if not os.path.exists(TILES_BASE_DIR):
    print("The raw tiles base directory is required but does not exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

# the base directory for tile lists
TILE_LIST_BASE_DIR = '{}/tile-lists'.format(EXPERIMENT_DIR)
if not os.path.exists(TILE_LIST_BASE_DIR):
    os.makedirs(TILE_LIST_BASE_DIR)

# the directory for this tile list
TILE_LIST_DIR = '{}/{}'.format(TILE_LIST_BASE_DIR, args.tile_list_name)
if os.path.exists(TILE_LIST_DIR):
    shutil.rmtree(TILE_LIST_DIR)
os.makedirs(TILE_LIST_DIR)
os.makedirs('{}/annotations-from-predictions'.format(TILE_LIST_DIR))    # annotations by converting predictions
os.makedirs('{}/annotations-from-via'.format(TILE_LIST_DIR))            # annotations edited by a human expert
os.makedirs('{}/annotations-from-tfe'.format(TILE_LIST_DIR))            # annotations from TFE features

# the file we will create
TILE_LIST_FILE_NAME = '{}/tile-list-{}.txt'.format(TILE_LIST_DIR, args.tile_list_name)
if os.path.isfile(TILE_LIST_FILE_NAME):
    os.remove(TILE_LIST_FILE_NAME)

# we will use all the tiles in retention time
file_list = []
for run_name in args.run_names:
    for idx in indexes_l:
        file_list += sorted(glob.glob("{}/run-{}-frame-*-tile-{}.png".format(TILES_BASE_DIR, run_name, idx)))

# write out the list of tiles
if len(file_list) > 0:
    print("writing {} entries to {}".format(len(file_list), TILE_LIST_FILE_NAME))
    with open(TILE_LIST_FILE_NAME, 'w') as f:
        for file in file_list:
            f.write('{}\n'.format(file))
else:
    print("could not find any tiles in the tile set directory: {}".format(TILES_BASE_DIR))

# create a DF to hold the metadata for each tile
tile_list_df = pd.DataFrame(file_list, columns=['full_path'])
tile_list_df['base_name'] = tile_list_df.apply(lambda row: os.path.basename(row.full_path), axis=1)

# load the tile set metadata so we can get the retention time for each tile
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.tile_set_name)
TILE_SET_METADATA_FILE_NAME = '{}/metadata.json'.format(TILES_BASE_DIR)
if os.path.isfile(TILE_SET_METADATA_FILE_NAME):
    with open(TILE_SET_METADATA_FILE_NAME) as json_file:
        tile_set_metadata = json.load(json_file)
        tile_set_tiles_df = pd.DataFrame(tile_set_metadata['tiles'])
        tile_set_tiles_df['base_name'] = tile_set_tiles_df.apply(lambda row: os.path.basename(row.tile_file_name), axis=1)
else:
    print("Could not find the tile list's metadata file: {}".format(TILE_SET_METADATA_FILE_NAME))
    sys.exit(1)

# find the extent of the tile list in m/z, RT, runs
print('collating metadata for each tile')
tile_list_df['run_name'] = tile_list_df.apply(lambda row: row.base_name.split('run-')[1].split('-frame')[0], axis=1)
tile_list_df['frame_id'] = tile_list_df.apply(lambda row: int(row.base_name.split('frame-')[1].split('-tile')[0]), axis=1)
tile_list_df['tile_id'] = tile_list_df.apply(lambda row: int(row.base_name.split('tile-')[1]), axis=1)
tile_list_df['mz_lower'] = tile_list_df.apply(lambda row: mz_range_for_tile(row.tile_id)[0], axis=1)
tile_list_df['mz_upper'] = tile_list_df.apply(lambda row: mz_range_for_tile(row.tile_id)[1], axis=1)
tile_list_df['retention_time_secs'] = tile_list_df.apply(lambda row: round(tile_set_tiles_df[tile_set_tiles_df.base_name == row.base_name].iloc[0].retention_time_secs,1), axis=1)
metadata['tile_info'] = tile_list_df.to_dict('records')

metadata["processed"] = time.ctime()
metadata["processor"] = parser.prog

# write out the metadata file
metadata_filename = '{}/metadata.json'.format(TILE_LIST_DIR)
print('writing the metadata file to {}'.format(metadata_filename))
with open(metadata_filename, 'w') as outfile:
    json.dump(metadata, outfile)
