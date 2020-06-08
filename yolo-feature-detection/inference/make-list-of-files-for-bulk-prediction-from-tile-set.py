import glob
import os
import argparse
import shutil
import sys

parser = argparse.ArgumentParser(description='Create a file containing the tiles from a tile set to be used for batch inference.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)

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

# check the raw tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.tile_set_name)
if not os.path.exists(TILES_BASE_DIR):
    print("The raw tiles base directory is required but does not exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

# check the predictions directory exists
PREDICTIONS_BASE_DIR = '{}/predictions/tile-sets/{}'.format(EXPERIMENT_DIR, args.tile_set_name)
if not os.path.exists(PREDICTIONS_BASE_DIR):
    os.makedirs(PREDICTIONS_BASE_DIR)

# the file we will create
TILE_LIST_FILE_NAME = '{}/batch-inference-tile-set-{}.txt'.format(PREDICTIONS_BASE_DIR, args.tile_set_name)

# we will use all the tiles in the tile set directory for inference
file_list = sorted(glob.glob("{}/run-*-frame-*-tile-*.png".format(TILES_BASE_DIR)))
if len(file_list) > 0:
    print("writing {} entries to {}".format(len(file_list), TILE_LIST_FILE_NAME))
    with open(TILE_LIST_FILE_NAME, 'w') as f:
        for file in file_list:
            f.write('{}\n'.format(file))
else:
    print("could not find any tiles in the tile set directory: {}".format(TILES_BASE_DIR))
