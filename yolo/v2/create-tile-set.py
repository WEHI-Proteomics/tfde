import sqlite3
import pandas as pd
import numpy as np
import sys
from matplotlib import colors, pyplot as plt
import argparse
import os, shutil
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import time
import ray
import json
import multiprocessing as mp
from cmcrameri import cm


# Create a set of tiles without labels for training purposes. This version uses Mk3 of the tile rendering algorithm.
# Example: python ./otf-peak-detect/yolo-feature-detection/training/create-raw-data-tiles.py -eb ~/Downloads/experiments -en 190719_Hela_Ecoli -rn 190719_Hela_Ecoli_1to3_06 -tidx 33 34

PIXELS_X = 910
PIXELS_Y = 910
MZ_MIN = 100.0
MZ_MAX = 1702.0
MZ_PER_TILE = 18.0
TILES_PER_FRAME = int((MZ_MAX - MZ_MIN) / MZ_PER_TILE) + 1
MIN_TILE_IDX = 0
MAX_TILE_IDX = TILES_PER_FRAME-1


# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

def mz_from_tile_pixel(tile_id, pixel_x):
    assert (pixel_x >= 0) and (pixel_x <= PIXELS_X), "pixel_x not in range"
    assert (tile_id >= 0) and (tile_id <= TILES_PER_FRAME-1), "tile_id not in range"
    mz = (tile_id * MZ_PER_TILE) + ((pixel_x / PIXELS_X) * MZ_PER_TILE) + MZ_MIN
    return mz

def tile_id_from_mz(mz):
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"
    tile_id = int((mz - MZ_MIN) / MZ_PER_TILE)
    return tile_id

def tile_pixel_x_from_mz(mz):
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"
    pixel_x = int(((mz - MZ_MIN) % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return pixel_x

def tile_pixel_y_from_scan(scan):
    assert (scan >= args.scan_lower) and (scan <= args.scan_upper), "scan not in range"
    pixel_y = int(((scan - args.scan_lower) / (args.scan_upper - args.scan_lower)) * PIXELS_Y)
    return pixel_y

def scan_from_tile_pixel(pixel_y):
    assert (pixel_y >= 0) and (pixel_y <= PIXELS_Y), "pixel_y not in range"
    scan = int(pixel_y / PIXELS_Y * (args.scan_upper - args.scan_lower))
    return scan

def mz_range_for_tile(tile_id):
    assert (tile_id >= 0) and (tile_id <= TILES_PER_FRAME-1), "tile_id not in range"
    mz_lower = MZ_MIN + (tile_id * MZ_PER_TILE)
    mz_upper = mz_lower + MZ_PER_TILE
    return (mz_lower, mz_upper)

def create_indexes(db_file_name):
    db_conn = sqlite3.connect(db_file_name)
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_tile_set_1 on frames (frame_id, mz)")
    src_c.execute("create index if not exists idx_tile_set_2 on frame_properties (Time, MsMsType)")
    db_conn.close()

@ray.remote
def render_frame(run_name, converted_db_name, frame_id, retention_time_secs, min_tile_id, max_tile_id, frame_idx, total_frames):
    print("processing frame {} of {} for run {}".format(frame_idx, total_frames, run_name))

    # find the mz range for the tiles specified
    frame_mz_lower = mz_range_for_tile(min_tile_id)[0]  # the lower mz range for the lowest tile index specified
    frame_mz_upper = mz_range_for_tile(max_tile_id)[1]  # the upper mz range for the highest tile index specified
    print('frame m/z range {},{} for tiles {},{}'.format(frame_mz_lower,frame_mz_upper,min_tile_id,max_tile_id))

    # read the raw points for the frame
    db_conn = sqlite3.connect(converted_db_name)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {} and mz >= {} and mz <= {}".format(frame_id, frame_mz_lower, frame_mz_upper), db_conn)
    db_conn.close()

    tile_list = []
    if len(raw_points_df) > 0:
        # assign a tile_id and a pixel x value to each raw point
        raw_points_df['tile_id'] = raw_points_df.apply(lambda row: tile_id_from_mz(row.mz), axis=1)
        raw_points_df['pixel_x'] = raw_points_df.apply(lambda row: tile_pixel_x_from_mz(row.mz), axis=1)
        raw_points_df['pixel_y'] = raw_points_df.apply(lambda row: tile_pixel_y_from_scan(row.scan), axis=1)

        # sum the intensity of raw points that have been assigned to each pixel
        pixel_intensity_df = raw_points_df.groupby(by=['tile_id', 'pixel_x', 'pixel_y'], as_index=False).intensity.sum()

        # create the colour map to convert intensity to colour
        colour_map = cm.lajolla_r
        norm = colors.LogNorm(vmin=args.minimum_pixel_intensity, vmax=args.maximum_pixel_intensity, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

        # calculate the colour to represent the intensity
        colours_l = []
        for i in pixel_intensity_df.intensity.unique():
            colours_l.append((i, colour_map(norm(i), bytes=True)[:3]))
        colours_df = pd.DataFrame(colours_l, columns=['intensity','colour'])
        pixel_intensity_df = pd.merge(pixel_intensity_df, colours_df, how='left', left_on=['intensity'], right_on=['intensity'])

        # extract the pixels for the frame for the specified tiles
        for tile_id in range(min_tile_id, max_tile_id+1):
            tile_df = pixel_intensity_df[(pixel_intensity_df.tile_id == tile_id)]

            # create an intensity array
            tile_im_array = np.zeros([PIXELS_Y+1, PIXELS_X+1, 3], dtype=np.uint8)  # container for the image
            for r in zip(tile_df.pixel_x, tile_df.pixel_y, tile_df.colour):
                x = r[0]
                y = r[1]
                c = r[2]
                tile_im_array[y,x,:] = c

            # create an image of the intensity array
            tile = Image.fromarray(tile_im_array, 'RGB')
            tile_file_name = '{}/run-{}-frame-{}-tile-{}.png'.format(TILES_BASE_DIR, run_name, frame_id, tile_id)
            tile.save(tile_file_name)

            tile_mz_lower,tile_mz_upper = mz_range_for_tile(tile_id)
            tile_list.append({'run_name':run_name, 'frame_id':frame_id, 'tile_id':tile_id, 'mz_lower':tile_mz_lower, 'mz_upper':tile_mz_upper, 'retention_time_secs':retention_time_secs, 'tile_file_name':tile_file_name})

    return tile_list

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers


##################################
parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_names', nargs='+', type=str, help='Space-separated names of runs to include.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)
parser.add_argument('-rtl','--rt_lower', type=int, default=200, help='Lower bound of the RT range.', required=False)
parser.add_argument('-rtu','--rt_upper', type=int, default=800, help='Upper bound of the RT range.', required=False)
parser.add_argument('-sl','--scan_lower', type=int, default=1, help='Lower bound of the scan range.', required=False)
parser.add_argument('-su','--scan_upper', type=int, default=910, help='Upper bound of the scan range.', required=False)
parser.add_argument('-maxpi','--maximum_pixel_intensity', type=int, default=500, help='Maximum pixel intensity for encoding, above which will be clipped.', required=False)
parser.add_argument('-minpi','--minimum_pixel_intensity', type=int, default=1, help='Minimum pixel intensity for encoding, below which will be clipped.', required=False)
parser.add_argument('-tidl','--tile_idx_lower', type=int, default=20, help='Lower range of the tile indexes to render. Must be between {} and {}'.format(MIN_TILE_IDX,MAX_TILE_IDX), required=False)
parser.add_argument('-tidu','--tile_idx_upper', type=int, default=40, help='Upper range of the tile indexes to render. Must be between {} and {}'.format(MIN_TILE_IDX,MAX_TILE_IDX), required=False)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], default='cluster', help='The Ray mode to use.', required=False)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.6, help='Proportion of the machine\'s cores to use for this program.', required=False)
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

# store the arguments as metadata for later reference
tile_set_metadata = {'arguments':vars(args)}

# check the tile index range is valid
if (args.tile_idx_lower < MIN_TILE_IDX) or (args.tile_idx_lower > MAX_TILE_IDX) or (args.tile_idx_upper < MIN_TILE_IDX) or (args.tile_idx_upper > MAX_TILE_IDX):
    print("The tile index must be between {} and {} inclusive".format(MIN_TILE_IDX, MAX_TILE_IDX))
    sys.exit(1)

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the converted database directory exists
CONVERTED_DATABASE_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(CONVERTED_DATABASE_DIR):
    print("The converted database directory is required but doesn't exist: {}".format(CONVERTED_DATABASE_DIR))
    sys.exit(1)

# create the tiles base directory
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.tile_set_name)
if os.path.exists(TILES_BASE_DIR):
    shutil.rmtree(TILES_BASE_DIR)
os.makedirs(TILES_BASE_DIR)


##############################
start_run = time.time()

print("setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(object_store_memory=20000000000,
                    redis_max_memory=25000000000,
                    num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

tile_metadata_l = []
for run_name in args.run_names:
    # check the converted database exists
    CONVERTED_DATABASE_NAME = "{}/exp-{}-run-{}-converted.sqlite".format(CONVERTED_DATABASE_DIR, args.experiment_name, run_name)
    if not os.path.isfile(CONVERTED_DATABASE_NAME):
        print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
        sys.exit(1)

    print("creating indexes in {} if they don't already exist".format(CONVERTED_DATABASE_NAME))
    create_indexes(CONVERTED_DATABASE_NAME)

    print("rendering tiles {} to {} from run {}, with m/z range {} to {}".format(args.tile_idx_lower, args.tile_idx_upper, run_name, round(mz_range_for_tile(args.tile_idx_lower)[0],1), round(mz_range_for_tile(args.tile_idx_upper)[1],1)))

    # get the ms1 frame ids within the specified retention time
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    ms1_frame_properties_df = pd.read_sql_query("select Id,Time from frame_properties where Time >= {} and Time <= {} and MsMsType == {}".format(args.rt_lower, args.rt_upper, FRAME_TYPE_MS1), db_conn)
    ms1_frame_ids = tuple(ms1_frame_properties_df.Id)
    db_conn.close()

    l = ray.get([render_frame.remote(run_name, CONVERTED_DATABASE_NAME, frame_id, ms1_frame_properties_df[ms1_frame_properties_df.Id == frame_id].iloc[0].Time, args.tile_idx_lower, args.tile_idx_upper, frame_idx, len(ms1_frame_ids)) for frame_idx,frame_id in enumerate(ms1_frame_ids, start=1)])
    for item in l:
        tile_metadata_l.append(item)

tile_metadata_l = [item for sublist in tile_metadata_l for item in sublist]  # tile_metadata_l is a list of lists, so we need to flatten it

stop_run = time.time()
tile_set_metadata["run processing time (sec)"] = round(stop_run-start_run,1)
tile_set_metadata["processed"] = time.ctime()
tile_set_metadata["processor"] = parser.prog
print("{} info: {}".format(parser.prog, tile_set_metadata))

# write out the metadata file
tile_set_metadata['tiles'] = tile_metadata_l
metadata_filename = '{}/metadata.json'.format(TILES_BASE_DIR)
with open(metadata_filename, 'w') as outfile:
    json.dump(tile_set_metadata, outfile)

print("shutting down ray")
ray.shutdown()

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
