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

# generate a tile for each frame, annotating intersecting precursor cuboids


MZ_MIN = 748        # default is 748
MZ_MAX = 766        # default is 766
SCAN_MIN = 350      # default is 1
SCAN_MAX = 850      # default is 920
PIXELS_X = 800
PIXELS_Y = 800

PIXELS_PER_MZ = PIXELS_X / (MZ_MAX - MZ_MIN)
PIXELS_PER_SCAN = PIXELS_Y / (SCAN_MAX - SCAN_MIN)

minimum_pixel_intensity = 1
maximum_pixel_intensity = 250

TILES_BASE_DIR = '/Users/darylwilding-mcbride/Downloads/three-d-intensity-descent-tiles'

def pixel_x_from_mz(mz):
    pixel_x = int((mz - MZ_MIN) * PIXELS_PER_MZ)
    return pixel_x

def pixel_y_from_scan(scan):
    pixel_y = int((scan - SCAN_MIN) * PIXELS_PER_SCAN)
    return pixel_y

raw_df = pd.read_pickle('/Users/darylwilding-mcbride/Downloads/YHE211_1-mz-748-766-rt-2000-2200.pkl')
raw_df = raw_df[(raw_df.frame_type == 0) & (raw_df.mz >= MZ_MIN) & (raw_df.mz <= MZ_MAX) & (raw_df.scan >= SCAN_MIN) & (raw_df.scan <= SCAN_MAX)]

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
CUBOIDS_FILE = '/Users/darylwilding-mcbride/Downloads/precursor-cuboids.pkl'
precursor_cuboids_df = pd.read_pickle(CUBOIDS_FILE)

# add a buffer around the edges
x_buffer = 5
y_buffer = 5

tile_id=1
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
