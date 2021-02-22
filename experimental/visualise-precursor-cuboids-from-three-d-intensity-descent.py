import pandas as pd
import numpy as np
from matplotlib import colors, cm, text, pyplot as plt
import matplotlib.patches as patches
import os
import time
from cmcrameri import cm
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import shutil
from cmcrameri import cm
import sqlite3

# generate a tile for each frame, annotating intersecting precursor cuboids


MZ_MIN = 748        # default is 748
MZ_MAX = 766        # default is 766
SCAN_MIN = 350      # default is 1
SCAN_MAX = 850      # default is 920
RT_MIN = 2000
RT_MAX = 2200

PIXELS_X = 800
PIXELS_Y = 800

PIXELS_PER_MZ = PIXELS_X / (MZ_MAX - MZ_MIN)
PIXELS_PER_SCAN = PIXELS_Y / (SCAN_MAX - SCAN_MIN)

minimum_pixel_intensity = 1
maximum_pixel_intensity = 250

TILES_BASE_DIR = '/home/ubuntu/precursor-cuboid-3did-tiles'

# check the experiment directory exists
experiment_base_dir = '/data2/experiments'
experiment_name = 'P3856'
run_name = 'P3856_YHE211_1_Slot1-1_1_5104'

EXPERIMENT_DIR = "{}/{}".format(experiment_base_dir, experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the converted databases directory exists
CONVERTED_DATABASE_NAME = "{}/converted-databases/exp-{}-run-{}-converted.sqlite".format(EXPERIMENT_DIR, experiment_name, run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

CUBOIDS_DIR = "{}/precursor-cuboids-3did".format(EXPERIMENT_DIR)
CUBOIDS_FILE = '{}/exp-{}-run-{}-mz-100-1700-precursor-cuboids.pkl'.format(CUBOIDS_DIR, experiment_name, run_name)

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

def pixel_x_from_mz(mz):
    pixel_x = int((mz - MZ_MIN) * PIXELS_PER_MZ)
    return pixel_x

def pixel_y_from_scan(scan):
    pixel_y = int((scan - SCAN_MIN) * PIXELS_PER_SCAN)
    return pixel_y


print('loading raw data from {}'.format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
raw_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {}".format(FRAME_TYPE_MS1, MZ_MIN, MZ_MAX, SCAN_MIN, SCAN_MAX, RT_MIN, RT_MAX), db_conn)
db_conn.close()

raw_df['pixel_x'] = raw_df.apply(lambda row: pixel_x_from_mz(row.mz), axis=1)
raw_df['pixel_y'] = raw_df.apply(lambda row: pixel_y_from_scan(row.scan), axis=1)

# sum the intensity of raw points that have been assigned to each pixel
pixel_intensity_df = raw_df.groupby(by=['frame_id', 'pixel_x', 'pixel_y'], as_index=False).intensity.sum()
print('intensity range {}..{}'.format(pixel_intensity_df.intensity.min(), pixel_intensity_df.intensity.max()))

# create the colour map to convert intensity to colour
colour_map = plt.get_cmap('ocean')
# colour_map = cm.batlow
norm = colors.LogNorm(vmin=minimum_pixel_intensity, vmax=maximum_pixel_intensity, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

# calculate the colour to represent the intensity
colours_l = []
for i in pixel_intensity_df.intensity.unique():
    colours_l.append((i, colour_map(norm(i), bytes=True)[:3]))
colours_df = pd.DataFrame(colours_l, columns=['intensity','colour'])
pixel_intensity_df = pd.merge(pixel_intensity_df, colours_df, how='left', left_on=['intensity'], right_on=['intensity'])

# create the tiles base directory
if os.path.exists(TILES_BASE_DIR):
    shutil.rmtree(TILES_BASE_DIR)
os.makedirs(TILES_BASE_DIR)

# load the precursor cuboids
print('loading the precursor cuboids from {}'.format(CUBOIDS_FILE))
precursor_cuboids_df = pd.read_pickle(CUBOIDS_FILE)

# add a buffer around the edges
x_buffer = 5
y_buffer = 5

tile_id=1
print('generating the tiles')
for group_name,group_df in pixel_intensity_df.groupby(['frame_id'], as_index=False):
    tile_rt = raw_df[(raw_df.frame_id == group_name)].iloc[0].retention_time_secs

    # create an intensity array
    tile_im_array = np.zeros([PIXELS_Y+1, PIXELS_X+1, 3], dtype=np.uint8)  # container for the image
    for r in zip(group_df.pixel_x, group_df.pixel_y, group_df.colour):
        x = r[0]
        y = r[1]
        c = r[2]
        tile_im_array[y,x,:] = c

    # create an image of the intensity array
    tile = Image.fromarray(tile_im_array, 'RGB')
    enhancer_object = ImageEnhance.Brightness(tile)
    tile = enhancer_object.enhance(1.1)

    # get a drawing context for the bounding boxes
    draw = ImageDraw.Draw(tile)

    # find the intersecting precursor cuboids for this tile; can be partial overlap in the m/z and scan dimensions
    intersecting_cuboids_df = precursor_cuboids_df[
                (precursor_cuboids_df.rt_lower <= tile_rt) & (precursor_cuboids_df.rt_upper >= tile_rt) & 
                ((precursor_cuboids_df.mz_lower >= MZ_MIN) & (precursor_cuboids_df.mz_lower <= MZ_MAX) | 
                (precursor_cuboids_df.mz_upper >= MZ_MIN) & (precursor_cuboids_df.mz_upper <= MZ_MAX)) & 
                ((precursor_cuboids_df.scan_lower >= SCAN_MIN) & (precursor_cuboids_df.scan_lower <= SCAN_MAX) |
                (precursor_cuboids_df.scan_upper >= SCAN_MIN) & (precursor_cuboids_df.scan_upper <= SCAN_MAX))
                ]

    for idx,cuboid in intersecting_cuboids_df.iterrows():
        # get the coordinates for the bounding box
        x0 = pixel_x_from_mz(cuboid.mz_lower)
        x1 = pixel_x_from_mz(cuboid.mz_upper)
        y0 = pixel_y_from_scan(cuboid.scan_lower)
        y1 = pixel_y_from_scan(cuboid.scan_upper)
        # draw the bounding box
        draw.rectangle(xy=[(x0-x_buffer, y0-y_buffer), (x1+x_buffer, y1+y_buffer)], fill=None, outline='limegreen')

    # save the tile
    tile_file_name = '{}/tile-{}.png'.format(TILES_BASE_DIR, tile_id)
    tile.save(tile_file_name)
    tile_id += 1

    print('.', end='', flush=True)

print()
print('saved {} tiles to {}'.format(tile_id, TILES_BASE_DIR))
