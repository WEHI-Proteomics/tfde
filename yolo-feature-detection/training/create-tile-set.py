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

# Create a set of tiles without labels for training purposes. This version uses Mk3 of the tile rendering algorithm.
# Example: python ./otf-peak-detect/yolo-feature-detection/training/create-raw-data-tiles.py -eb ~/Downloads/experiments -en 190719_Hela_Ecoli -rn 190719_Hela_Ecoli_1to3_06 -tidx 33 34

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

def mz_from_tile_pixel(tile_id, pixel_x):
    assert (pixel_x >= 0) and (pixel_x <= PIXELS_X), "pixel_x not in range"
    assert (tile_id >= 0) and (tile_id <= TILES_PER_FRAME-1), "tile_id not in range"

    mz = (tile_id * MZ_PER_TILE) + ((pixel_x / PIXELS_X) * MZ_PER_TILE) + MZ_MIN
    return mz

def tile_pixel_x_from_mz(mz):
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"

    tile_id = int((mz - MZ_MIN) / MZ_PER_TILE)
    pixel_x = int(((mz - MZ_MIN) % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return (tile_id, pixel_x)

def mz_range_for_tile(tile_id):
    assert (tile_id >= 0) and (tile_id <= TILES_PER_FRAME-1), "tile_id not in range"

    mz_lower = MZ_MIN + (tile_id * MZ_PER_TILE)
    mz_upper = mz_lower + MZ_PER_TILE
    return (mz_lower, mz_upper)

def interpolate_pixels(tile_im_array):
    z = np.array([0,0,0])
    for x in range(4, PIXELS_X-4):
        for y in range(4, PIXELS_Y-4):
            c = tile_im_array[y,x]
            if (c == z).all():  # only change pixels that are [0,0,0]
                # build the list of this pixel's neighbours to average
                n = []
                n.append(tile_im_array[y+1,x-1])
                n.append(tile_im_array[y+1,x])
                n.append(tile_im_array[y+1,x+1])

                n.append(tile_im_array[y-1,x-1])
                n.append(tile_im_array[y-1,x])
                n.append(tile_im_array[y-1,x+1])

                n.append(tile_im_array[y+2,x-1])
                n.append(tile_im_array[y+2,x])
                n.append(tile_im_array[y+2,x+1])

                n.append(tile_im_array[y-2,x-1])
                n.append(tile_im_array[y-2,x])
                n.append(tile_im_array[y-2,x+1])

                n.append(tile_im_array[y+3,x-1])
                n.append(tile_im_array[y+3,x])
                n.append(tile_im_array[y+3,x+1])

                n.append(tile_im_array[y-3,x-1])
                n.append(tile_im_array[y-3,x])
                n.append(tile_im_array[y-3,x+1])

                n.append(tile_im_array[y+4,x-1])
                n.append(tile_im_array[y+4,x])
                n.append(tile_im_array[y+4,x+1])

                n.append(tile_im_array[y-4,x-1])
                n.append(tile_im_array[y-4,x])
                n.append(tile_im_array[y-4,x+1])

                n.append(tile_im_array[y,x-1])
                n.append(tile_im_array[y,x+1])

                # set the pixel's colour to be the channel-wise mean of its neighbours
                neighbours_a = np.array(n)
                tile_im_array[y,x] = np.mean(neighbours_a, axis=0)

    return tile_im_array

def create_indexes(db_file_name):
    db_conn = sqlite3.connect(db_file_name)
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_tile_set_1 on frames (frame_id, mz)")
    db_conn.close()


# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)
parser.add_argument('-rtl','--rt_lower', type=int, default=200, help='Lower bound of the RT range.', required=False)
parser.add_argument('-rtu','--rt_upper', type=int, default=800, help='Upper bound of the RT range.', required=False)
parser.add_argument('-maxpi','--maximum_pixel_intensity', type=int, default=1000, help='Maximum pixel intensity for encoding, above which will be clipped.', required=False)
parser.add_argument('-minpi','--minimum_pixel_intensity', type=int, default=1, help='Minimum pixel intensity for encoding, below which will be clipped.', required=False)
parser.add_argument('-inp','--interpolate_neighbouring_pixels', action='store_true', help='Use the value of surrounding pixels to fill zero pixels.')
parser.add_argument('-tidl','--tile_idx_lower', type=int, help='Lower range of the tile indexes to render.', required=True)
parser.add_argument('-tidu','--tile_idx_upper', type=int, help='Upper range of the tile indexes to render.', required=True)
parser.add_argument('-np','--number_of_processors', type=int, default=8, help='The number of processors to use.', required=False)
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
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

# check the run directory exists
RUN_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

# check the converted database exists
CONVERTED_DATABASE_NAME = "{}/exp-{}-run-{}-converted.sqlite".format(RUN_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

# create the tiles base directory
TILES_BASE_DIR = '{}/tiles/{}/{}'.format(EXPERIMENT_DIR, args.run_name, args.tile_set_name)
if not os.path.exists(TILES_BASE_DIR):
    os.makedirs(TILES_BASE_DIR)
    print("The tiles base directory was created: {}".format(TILES_BASE_DIR))

# set up a tile directory for each run
tile_dir_d = {}
for tile_idx in range(args.tile_idx_lower, args.tile_idx_upper+1):
    tile_dir = "{}/tile-{}".format(TILES_BASE_DIR, tile_idx)
    tile_dir_d[tile_idx] = tile_dir
    if os.path.exists(tile_dir):
        shutil.rmtree(tile_dir)
    os.makedirs(tile_dir)
    print("Created {}".format(tile_dir))

start_run = time.time()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))

PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines
MZ_MIN = 100.0
MZ_MAX = 1700.0
SCAN_MAX = PIXELS_Y
SCAN_MIN = 1
MZ_PER_TILE = 18.0
TILES_PER_FRAME = int((MZ_MAX - MZ_MIN) / MZ_PER_TILE) + 1

print("creating indexes")
create_indexes(CONVERTED_DATABASE_NAME)


if not ray.is_initialized():
    ray.init(num_cpus=args.number_of_processors)

@ray.remote
def render_frame(frame_id, tile_dir_d, idx, total_frames):
    print("processing frame {} of {}".format(idx, total_frames))

    # find the mz range for the tiles specified
    frame_mz_lower = mz_range_for_tile(min(tile_dir_d.keys()))[0]  # the lower mz range for the lowest tile index specified
    frame_mz_upper = mz_range_for_tile(max(tile_dir_d.keys()))[1]  # the upper mz range for the highest tile index specified

    # read the raw points for the frame
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {} and mz >= {} and mz <= {}".format(frame_id, frame_mz_lower, frame_mz_upper), db_conn)
    db_conn.close()

    tile_pixels_df = pd.DataFrame(raw_points_df.apply(lambda row: tile_pixel_x_from_mz(row.mz), axis=1).tolist(), columns=['tile_id', 'pixel_x'])
    raw_points_df = pd.concat([raw_points_df, tile_pixels_df], axis=1)
    pixel_intensity_df = raw_points_df.groupby(by=['tile_id', 'pixel_x', 'scan'], as_index=False).intensity.sum()

    # create the colour map to convert intensity to colour
    colour_map = plt.get_cmap('rainbow')
    norm = colors.LogNorm(vmin=args.minimum_pixel_intensity, vmax=args.maximum_pixel_intensity, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

    # calculate the colour to represent the intensity
    colour_l = []
    for r in zip(pixel_intensity_df.intensity):
        colour_l.append((colour_map(norm(r[0]), bytes=True)[:3]))
    pixel_intensity_df['colour'] = colour_l

    # extract the pixels for the frame for the specified tiles
    for tile_idx in tile_dir_d.keys():
        tile_df = pixel_intensity_df[(pixel_intensity_df.tile_id == tile_idx)]

        # create an intensity array
        tile_im_array = np.zeros([PIXELS_Y+1, PIXELS_X+1, 3], dtype=np.uint8)  # container for the image
        for r in zip(tile_df.pixel_x, tile_df.scan, tile_df.colour):
            x = r[0]
            y = r[1]
            c = r[2]
            tile_im_array[y,x,:] = c

        if args.interpolate_neighbouring_pixels:
            # fill in zero pixels with interpolated values
            tile_im_array = interpolate_pixels(tile_im_array)

        # create an image of the intensity array
        tile = Image.fromarray(tile_im_array, 'RGB')
        mz_lower,mz_upper = mz_range_for_tile(tile_idx)
        tile.save('{}/frame-{}-tile-{}.png'.format(tile_dir_d[tile_idx], frame_id, tile_idx))

# get the ms1 frame ids within the specified retention time
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select Id,Time from frame_properties where Time >= {} and Time <= {} and MsMsType == {}".format(args.rt_lower, args.rt_upper, FRAME_TYPE_MS1), db_conn)
ms1_frame_ids = tuple(ms1_frame_properties_df.Id)
db_conn.close()

ray.get([render_frame.remote(frame_id, tile_dir_d, idx, len(ms1_frame_ids)) for idx,frame_id in enumerate(ms1_frame_ids, start=1)])

stop_run = time.time()
info.append(("run processing time (sec)", round(stop_run-start_run,1)))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))
print("{} info: {}".format(parser.prog, info))

print("shutting down ray")
ray.shutdown()

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
