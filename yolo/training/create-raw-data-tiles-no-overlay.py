import sqlite3
import pandas as pd
import numpy as np
import sys
from matplotlib import colors, cm, pyplot as plt
import argparse
import os, shutil
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import time
import ray

# Create a set of tiles without labels for training purposes. This version uses MkII of the tile rendering algorithm.
# Example: python ./otf-peak-detect/yolo/training/create-raw-data-tiles-no-overlay.py -eb ~/Downloads/experiments -en 190719_Hela_Ecoli -rn 190719_Hela_Ecoli_1to3_06 -tidx 33 34

MS1_CE = 10

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-rtl','--rt_lower', type=int, default=200, help='Lower bound of the RT range.', required=False)
parser.add_argument('-rtu','--rt_upper', type=int, default=800, help='Upper bound of the RT range.', required=False)
parser.add_argument('-tidx','--tile_idx_list', nargs='+', type=int, help='Space-separated indexes of the tiles to render.', required=True)
parser.add_argument('-np','--number_of_processors', type=int, default=8, help='The number of processors to use.', required=False)
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the converted database exists
CONVERTED_DATABASE_NAME = "{}/converted-databases/{}-converted.sqlite".format(EXPERIMENT_DIR, args.run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

# create the tiles base directory
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.run_name)
if os.path.exists(TILES_BASE_DIR):
    shutil.rmtree(TILES_BASE_DIR)
os.makedirs(TILES_BASE_DIR)
print("The tiles base directory was created: {}".format(TILES_BASE_DIR))

# set up a tile directory for each run
tile_dir_d = {}
for tile_idx in args.tile_idx_list:
    tile_dir = "{}/tile-{}".format(TILES_BASE_DIR, tile_idx)
    tile_dir_d[tile_idx] = tile_dir
    os.makedirs(tile_dir)
    print("Created {}".format(tile_dir))

start_run = time.time()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))

print("opening {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(args.rt_lower, args.rt_upper, MS1_CE), db_conn)
ms1_frame_ids = tuple(ms1_frame_properties_df.frame_id)
db_conn.close()

frame_delay = ms1_frame_properties_df.iloc[1].retention_time_secs - ms1_frame_properties_df.iloc[0].retention_time_secs
print("frame period: {} seconds".format(round(frame_delay,1)))

PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines
PIXELS_PER_BIN = 1
MZ_MIN = 100.0
MZ_MAX = 1700.0
SCAN_MAX = PIXELS_Y
SCAN_MIN = 1
MZ_PER_TILE = 18.0
MZ_BIN_WIDTH = MZ_PER_TILE / (PIXELS_X * PIXELS_PER_BIN)
TILES_PER_FRAME = int((MZ_MAX - MZ_MIN) / MZ_PER_TILE)

mz_bins = np.arange(start=MZ_MIN, stop=MZ_MAX+MZ_BIN_WIDTH, step=MZ_BIN_WIDTH)  # go slightly wider to accommodate the maximum value
MZ_BIN_COUNT = len(mz_bins)

if not ray.is_initialized():
    ray.init(num_cpus=args.number_of_processors)

@ray.remote
def render_frame(frame_id, tile_dir_d, idx):
    print("processing frame {} of {}".format(idx, len(ms1_frame_ids)))
    # read the raw points for the frame
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {}".format(frame_id), db_conn)
    db_conn.close()

    frame_intensity_array = np.zeros([SCAN_MAX+1, MZ_BIN_COUNT+1], dtype=np.uint16)  # scratchpad for the intensity value prior to image conversion
    for r in zip(raw_points_df.mz,raw_points_df.scan,raw_points_df.intensity):
        mz = r[0]
        scan = int(r[1])
        if (mz >= MZ_MIN) and (mz <= MZ_MAX) and (scan >= SCAN_MIN) and (scan <= SCAN_MAX):
            mz_array_idx = int(np.digitize(mz, mz_bins))-1
            scan_array_idx = scan
            intensity = int(r[2])
            frame_intensity_array[scan_array_idx,mz_array_idx] += intensity

    # calculate the colour to represent the intensity
    colour_map = cm.get_cmap(name='magma')
    norm = colors.LogNorm(vmin=1, vmax=5e3, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

    # convert the intensity array to a dataframe
    intensity_df = pd.DataFrame(frame_intensity_array).stack().rename_axis(['y', 'x']).reset_index(name='intensity')
    # remove all the zero-intensity elements
    intensity_df = intensity_df[intensity_df.intensity > 0]

    # calculate the colour to represent the intensity
    colour_l = []
    for r in zip(intensity_df.intensity):
        colour_l.append((colour_map(norm(r[0]), bytes=True)[:3]))
    intensity_df['colour'] = colour_l

    # create an image of the whole frame
    frame_im_array = np.zeros([PIXELS_Y+1, MZ_BIN_COUNT+1, 3], dtype=np.uint8)  # container for the image
    for r in zip(intensity_df.x, intensity_df.y, intensity_df.colour):
        x = r[0]
        y = r[1]
        c = r[2]
        frame_im_array[y,x,:] = c

    # extract the pixels for the frame for the specified tiles
    for tile_idx in tile_dir_d.keys():
        tile_idx_base = tile_idx * PIXELS_X
        tile_idx_width = PIXELS_X
        # extract the subset of the frame for this image
        tile_im_array = frame_im_array[:,tile_idx_base:tile_idx_base+tile_idx_width,:]
        tile = Image.fromarray(tile_im_array, 'RGB')

        mz_lower = MZ_MIN + (tile_idx * MZ_PER_TILE)
        mz_upper = mz_lower + MZ_PER_TILE

        tile.save('{}/frame-{}-tile-{}-mz-{}-{}.png'.format(tile_dir_d[tile_idx], frame_id, tile_idx, int(mz_lower), int(mz_upper)))


ray.get([render_frame.remote(frame_id, tile_dir_d, idx) for idx,frame_id in enumerate(ms1_frame_ids, start=1)])

stop_run = time.time()
info.append(("run processing time (sec)", round(stop_run-start_run,1)))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))
print("{} info: {}".format(parser.prog, info))

print("shutting down ray")
ray.shutdown()
