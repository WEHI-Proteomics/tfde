import glob
import os
import argparse
import shutil
import sys

parser = argparse.ArgumentParser(description='Create a file containing the tiles to be used for batch inference.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tn','--training_set_name', type=str, help='Name of the training set.', required=True)
parser.add_argument('-tid','--tile_id', type=int, help='Index of the tile for inference.', required=True)

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

# check the training base directories
TRAINING_SET_BASE_DIR = '{}/training-sets/{}'.format(EXPERIMENT_DIR, args.training_set_name)
PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TRAINING_SET_BASE_DIR)

if not os.path.exists(TRAINING_SET_BASE_DIR):
    print("The training set directory is required but doesn't exist: {}".format(TRAINING_SET_BASE_DIR))
    sys.exit(1)

if not os.path.exists(PRE_ASSIGNED_FILES_DIR):
    print("The pre-assigned directory is required but doesn't exist: {}".format(PRE_ASSIGNED_FILES_DIR))
    sys.exit(1)

PREDICTIONS_BASE_DIR = '{}/predictions'.format(EXPERIMENT_DIR)
if not os.path.exists(PREDICTIONS_BASE_DIR):
    os.makedirs(PREDICTIONS_BASE_DIR)

TILE_PREDICTIONS_DIR = '{}/tile-{}'.format(PREDICTIONS_BASE_DIR, args.tile_id)
if os.path.exists(TILE_PREDICTIONS_DIR):
    shutil.rmtree(TILE_PREDICTIONS_DIR)
os.makedirs(TILE_PREDICTIONS_DIR)

TILE_FILE_NAME = '{}/batch-inference-tile-{}.txt'.format(TILE_PREDICTIONS_DIR, args.tile_id)

# we will use the tiles in the pre-assigned directory for inference
file_list = sorted(glob.glob("{}/frame-*-tile-{}.png".format(PRE_ASSIGNED_FILES_DIR, args.tile_id)))
if len(file_list) > 0:
    print("writing {} entries to {}".format(len(file_list), TILE_FILE_NAME))
    with open(TILE_FILE_NAME, 'w') as f:
        for file in file_list:
            f.write('{}\n'.format(file))
else:
    print("could not find any index {} tiles in the pre-assigned directory: {}".format(args.tile_id, PRE_ASSIGNED_FILES_DIR))
