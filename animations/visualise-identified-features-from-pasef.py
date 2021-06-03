import pandas as pd
import numpy as np
from matplotlib import colors, pyplot as plt
import os
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import sqlite3
import json
import shutil
from os.path import expanduser
import sys
import pickle

# generate a tile for each frame around a selected feature from the PASEF method

MAXIMUM_Q_VALUE = 0.01

# for focus on a particular feature
pasef_feature_id = None

# visualisation offsets from the feature apexes
VIS_MZ_OFFSET_LOWER = 5
VIS_MZ_OFFSET_UPPER = 5
VIS_SCAN_OFFSET_LOWER = 150
VIS_SCAN_OFFSET_UPPER = 150
VIS_RT_OFFSET_LOWER = 15
VIS_RT_OFFSET_UPPER = 15


experiment_name = 'P3856'
feature_detection_method = 'pasef'
run_name = 'P3856_YHE211_1_Slot1-1_1_5104'


RESULTS_BASE_DIR = '/media/big-ssd/results-P3856'
IDENTIFICATIONS_DIR = '{}/P3856-results-cs-true-fmdw-true-2021-05-20-02-44-34/identifications-pasef'.format(RESULTS_BASE_DIR)
IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.pkl'.format(IDENTIFICATIONS_DIR, experiment_name, feature_detection_method)

# load the identifications
with open(IDENTIFICATIONS_FILE, 'rb') as handle:
    d = pickle.load(handle)
identifications_df = d['identifications_df']
identifications_df = identifications_df[(identifications_df['percolator q-value'] <= MAXIMUM_Q_VALUE)]

if pasef_feature_id is not None:
    selected_feature = identifications_df[(identifications_df.feature_id == pasef_feature_id)].iloc[0]
    # for a selected feature
    limits = {
    'MZ_MIN': selected_feature.monoisotopic_mz - VIS_MZ_OFFSET_LOWER,
    'MZ_MAX': selected_feature.monoisotopic_mz + VIS_MZ_OFFSET_UPPER,
    'SCAN_MIN': selected_feature.scan_apex - VIS_SCAN_OFFSET_LOWER,
    'SCAN_MAX': selected_feature.scan_apex + VIS_SCAN_OFFSET_UPPER,
    'RT_MIN': selected_feature.rt_apex - VIS_RT_OFFSET_LOWER,
    'RT_MAX': selected_feature.rt_apex + VIS_RT_OFFSET_UPPER
    }
    print('coordinates for comparison:\n{}'.format(limits))
else:
    # for any feature
    limits = {
    'MZ_MIN': 700,
    'MZ_MAX': 710,
    'SCAN_MIN': 450,
    'SCAN_MAX': 920,
    'RT_MIN': 1650,
    'RT_MAX': 2200
    }

PIXELS_X = 800
PIXELS_Y = 800

PIXELS_PER_MZ = PIXELS_X / (limits['MZ_MAX'] - limits['MZ_MIN'])
PIXELS_PER_SCAN = PIXELS_Y / (limits['SCAN_MAX'] - limits['SCAN_MIN'])

minimum_pixel_intensity = 1
maximum_pixel_intensity = 250

# add a buffer around the edges of the bounding box
BB_MZ_BUFFER = 0.2
BB_SCAN_BUFFER = 5

EXPERIMENT_DIR = '/media/big-ssd/experiments/{}'.format(experiment_name)
CONVERTED_DATABASE_NAME = '/media/big-ssd/experiments/P3856/converted-databases/exp-P3856-run-{}-converted.sqlite'.format(run_name)

TILES_BASE_DIR = '{}/tiles/feature-tiles-pasef'.format(expanduser('~'))


if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print('the converted database is required but is not present: {}'.format(CONVERTED_DATABASE_NAME))
    sys.exit(1)


# font paths for overlay labels
UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'

def pixel_x_from_mz(mz):
    pixel_x = int((mz - limits['MZ_MIN']) * PIXELS_PER_MZ)
    return pixel_x

def pixel_y_from_scan(scan):
    pixel_y = int((scan - limits['SCAN_MIN']) * PIXELS_PER_SCAN)
    return pixel_y

# load the raw data for the region of interest
print('loading the raw data from {}'.format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
raw_df = pd.read_sql_query("select * from frames where frame_type == 0 and mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {};".format(limits['MZ_MIN'], limits['MZ_MAX'], limits['SCAN_MIN'], limits['SCAN_MAX'], limits['RT_MIN'], limits['RT_MAX']), db_conn)
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

# load the font to use for labelling the overlays
if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

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

    # draw the CCS markers
    ccs_marker_each = 50
    range_l = round(limits['SCAN_MIN'] / ccs_marker_each) * ccs_marker_each
    range_u = round(limits['SCAN_MAX'] / ccs_marker_each) * ccs_marker_each
    for marker_scan in np.arange(range_l,range_u+ccs_marker_each,ccs_marker_each):
        marker_y = pixel_y_from_scan(marker_scan)
        draw.text((10, marker_y-6), str(round(marker_scan)), font=feature_label_font, fill='lawngreen')
        draw.line((0,marker_y, 5,marker_y), fill='lawngreen', width=1)

    # draw the m/z markers
    mz_marker_each = 1
    range_l = round(limits['MZ_MIN'] / mz_marker_each) * mz_marker_each
    range_u = round(limits['MZ_MAX'] / mz_marker_each) * mz_marker_each
    for marker_mz in np.arange(range_l,range_u+mz_marker_each,mz_marker_each):
        marker_x = pixel_x_from_mz(marker_mz)
        draw.text((marker_x-10, 8), str(round(marker_mz)), font=feature_label_font, fill='lawngreen')
        draw.line((marker_x,0, marker_x,5), fill='lawngreen', width=1)

    # draw the tile info
    info_box_x_inset = 200
    info_box_y_inset = 24
    space_per_line = 12
    draw.rectangle(xy=[(PIXELS_X-info_box_x_inset, info_box_y_inset), (PIXELS_X, 3*space_per_line)], fill=(20,20,20), outline=None)
    draw.text((PIXELS_X-info_box_x_inset, (0*space_per_line)+info_box_y_inset), 'PASEF-seeded', font=feature_label_font, fill='lawngreen')
    draw.text((PIXELS_X-info_box_x_inset, (1*space_per_line)+info_box_y_inset), '{}'.format(run_name), font=feature_label_font, fill='lawngreen')
    draw.text((PIXELS_X-info_box_x_inset, (2*space_per_line)+info_box_y_inset), '{} secs'.format(round(tile_rt,1)), font=feature_label_font, fill='lawngreen')

    # find the intersecting features for this tile; can be partial overlap in the m/z and scan dimensions
    if pasef_feature_id is not None:
        # the selected feature
        intersecting_features_df = identifications_df[
                    (identifications_df.feature_id == pasef_feature_id) &
                    (identifications_df.rt_lower <= tile_rt) & (identifications_df.rt_upper >= tile_rt) & 
                    (identifications_df.monoisotopic_mz >= limits['MZ_MIN']) & (identifications_df.monoisotopic_mz <= limits['MZ_MAX']) & 
                    (identifications_df.scan_apex >= limits['SCAN_MIN']) & (identifications_df.scan_apex <= limits['SCAN_MAX'])
                    ]
    else:
        # any feature
        intersecting_features_df = identifications_df[
                    (identifications_df.rt_lower <= tile_rt) & (identifications_df.rt_upper >= tile_rt) & 
                    (identifications_df.monoisotopic_mz >= limits['MZ_MIN']) & (identifications_df.monoisotopic_mz <= limits['MZ_MAX']) & 
                    (identifications_df.scan_apex >= limits['SCAN_MIN']) & (identifications_df.scan_apex <= limits['SCAN_MAX'])
                    ]

    for idx,feature in intersecting_features_df.iterrows():
        # get the coordinates for the bounding box
        envelope = json.loads(feature.envelope)
        x0 = pixel_x_from_mz(envelope[0][0] - BB_MZ_BUFFER)
        x1 = pixel_x_from_mz(envelope[-1][0] + BB_MZ_BUFFER)
        y0 = pixel_y_from_scan(feature.scan_lower - BB_SCAN_BUFFER)
        y1 = pixel_y_from_scan(feature.scan_upper + BB_SCAN_BUFFER)
        # draw the bounding box
        draw.rectangle(xy=[(x0, y0), (x1, y1)], fill=None, outline='deepskyblue')
        # draw the bounding box label
        draw.text((x0, y0-(2*space_per_line)), 'feature {}'.format(feature.feature_id), font=feature_label_font, fill='lawngreen')
        draw.text((x0, y0-(1*space_per_line)), '{}, {}+'.format(feature.sequence, feature.charge), font=feature_label_font, fill='lawngreen')

    # save the tile
    tile_file_name = '{}/tile-{:05d}.png'.format(TILES_BASE_DIR, tile_id)
    tile.save(tile_file_name)
    tile_id += 1

    print('.', end='', flush=True)

print()
print('saved {} tiles to {}'.format(tile_id, TILES_BASE_DIR))
