# This script gets all the tiles for a particular section of m/z and creates a file
# containing their filenames for the purpose of bulk inference.

import glob
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)
parser.add_argument('-tidx','--tile_idx_list', nargs='+', type=int, help='Space-separated indexes of the tiles to use for the training set.', required=True)

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
TILES_BASE_DIR = '{}/tiles/{}/{}'.format(EXPERIMENT_DIR, args.run_name, args.tile_set_name)
if not os.path.exists(TILES_BASE_DIR):
    print("The raw tiles base directory is required but does not exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

# copy the tile indexes to a staging area for inference
INFERENCE_STAGING_DIR = '{}/inference-staging'.format(TILES_BASE_DIR)
if os.path.exists(INFERENCE_STAGING_DIR):
    shutil.rmtree(INFERENCE_STAGING_DIR)
os.makedirs(INFERENCE_STAGING_DIR)

# check the tiles directory exists for each tile index we need
for tile_idx in args.tile_idx_list:
    tile_dir = "{}/tile-{}".format(TILES_BASE_DIR, tile_idx)
    if os.path.exists(tile_dir):
        # copy the raw tiles to the inference staging tiles directory
        file_list = sorted(glob.glob("{}/frame-*-tile-*.png".format(tile_dir)))
        print("copying {} tiles from {} to {}".format(len(file_list), tile_dir, INFERENCE_STAGING_DIR))
        for file in file_list:
            base_name = os.path.basename(file)
            destination_name = '{}/{}'.format(INFERENCE_STAGING_DIR, base_name)
            shutil.copyfile(file, destination_name)
    else:
        print("The tiles directory is required but does not exist: {}".format(tile_dir))
        sys.exit(1)

# write out the file list
OUTPUT_FILENAME = './inference-files-tile-idxs-{}.txt'.format('-'.join(map(str, args.tile_idx_list)))  # the name of the file containing the list to process
file_list = sorted(glob.glob("{}/frame-*.png".format(INFERENCE_STAGING_DIR)))

print("writing {} entries to {}".format(len(file_list), OUTPUT_FILENAME))
with open(OUTPUT_FILENAME, 'w') as f:
    for file in file_list:
        f.write('{}/{}\n'.format(INFERENCE_STAGING_DIR, os.path.basename(file)))
