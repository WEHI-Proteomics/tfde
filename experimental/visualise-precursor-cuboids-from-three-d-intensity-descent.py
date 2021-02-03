import pandas as pd
import numpy as np
from matplotlib import colors, cm, text, pyplot as plt
import matplotlib.patches as patches
import os
import time
from cmcrameri import cm
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import shutil

# generate a tile for each frame, annotating intersecting precursor cuboids


MZ_MIN = 748
MZ_MAX = 766
SCAN_MIN = 1
SCAN_MAX = 920
PIXELS_X = 1000
PIXELS_Y = 1000

PIXELS_PER_MZ = PIXELS_X / (MZ_MAX - MZ_MIN)
PIXELS_PER_SCAN = PIXELS_Y / (SCAN_MAX - SCAN_MIN)

minimum_pixel_intensity = 50
maximum_pixel_intensity = 200

TILES_BASE_DIR = '/Users/darylwilding-mcbride/Downloads/three-d-intensity-descent-tiles'

def pixel_x_from_mz(mz):
    pixel_x = int((mz - MZ_MIN) * PIXELS_PER_MZ)
    return pixel_x

def pixel_y_from_scan(scan):
    pixel_y = int((scan - SCAN_MIN) * PIXELS_PER_SCAN)
    return pixel_y

raw_df = pd.read_pickle('/Users/darylwilding-mcbride/Downloads/YHE211_1-mz-748-766-rt-2000-2200.pkl')
raw_df = raw_df[(raw_df.frame_type == 0) & (raw_df.intensity >= 50)]

raw_df['pixel_x'] = raw_df.apply(lambda row: pixel_x_from_mz(row.mz), axis=1)
raw_df['pixel_y'] = raw_df.apply(lambda row: pixel_y_from_scan(row.scan), axis=1)

# sum the intensity of raw points that have been assigned to each pixel
pixel_intensity_df = raw_df.groupby(by=['retention_time_secs', 'pixel_x', 'pixel_y'], as_index=False).intensity.sum()

# create the colour map to convert intensity to colour
colour_map = plt.get_cmap('rainbow')
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
CUBOIDS_FILE = '/Users/darylwilding-mcbride/Downloads/precursor-cuboids.pkl'
precursor_cuboids_df = pd.read_pickle(CUBOIDS_FILE)

# add a buffer around the edges
x_buffer = 5
y_buffer = 5

tile_id=1
for group_name,group_df in pixel_intensity_df.groupby(['retention_time_secs'], as_index=False):
    tile_rt = group_name

    # create an intensity array
    tile_im_array = np.zeros([PIXELS_Y+1, PIXELS_X+1, 3], dtype=np.uint8)  # container for the image
    for r in zip(group_df.pixel_x, group_df.pixel_y, group_df.colour):
        x = r[0]
        y = r[1]
        c = r[2]
        tile_im_array[y,x,:] = c

    # create an image of the intensity array
    tile = Image.fromarray(tile_im_array, 'RGB')
    draw = ImageDraw.Draw(tile)

    # find the intersecting precursor cuboids for this tile
    intersecting_cuboids_df = precursor_cuboids_df[(precursor_cuboids_df.rt_lower <= tile_rt) & (precursor_cuboids_df.rt_upper >= tile_rt) & (precursor_cuboids_df.mz_lower >= MZ_MIN) & (precursor_cuboids_df.mz_upper <= MZ_MAX)]

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
