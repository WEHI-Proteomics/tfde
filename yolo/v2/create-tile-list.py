import glob
import os
import argparse
import shutil
import sys
import time
import json

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

metadata["processed"] = time.ctime()
metadata["processor"] = parser.prog
print("{} metadata: {}".format(parser.prog, metadata))

# write out the metadata file
metadata_filename = '{}/metadata.json'.format(TILE_LIST_DIR)
with open(metadata_filename, 'w') as outfile:
    json.dump(metadata, outfile)
