import pandas as pd
import numpy as np
import sys
import pickle
import glob
import os
import shutil
import sqlite3
import matplotlib.pyplot as plt
import argparse
from matplotlib import colors, cm, pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tempfile
import zipfile
import json
from cmcrameri import cm


def pixel_xy(mz, scan, mz_lower, mz_upper, scan_lower, scan_upper):
    x_pixels_per_mz = (args.pixels_x-1) / (mz_upper - mz_lower)
    y_pixels_per_scan = (args.pixels_y-1) / (scan_upper - scan_lower)
    
    pixel_x = int((mz - mz_lower) * x_pixels_per_mz)
    pixel_y = int((scan - scan_lower) * y_pixels_per_scan)
    return (pixel_x, pixel_y)

# loads the metadata, ms1 points, and ms2 points from the specified zip file
def load_precursor_cuboid_zip(filename):
    temp_dir = tempfile.TemporaryDirectory().name
    with zipfile.ZipFile(filename, "r") as zf:
        zf.extractall(path=temp_dir)
        names = zf.namelist()
        with open('{}/{}'.format(temp_dir, names[0])) as json_file:
            metadata = json.load(json_file)
        ms1_df = pd.read_pickle('{}/{}'.format(temp_dir, names[1]))
        ms2_df = pd.read_pickle('{}/{}'.format(temp_dir, names[2]))
    # clean up the temp directory
    shutil.rmtree(temp_dir)
    return (metadata, ms1_df, ms2_df)


# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# font paths for overlay labels
UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'

if os.path.isfile(UBUNTU_FONT_PATH):
    feature_label_font = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
else:
    feature_label_font = ImageFont.truetype(MACOS_FONT_PATH, 10)

# for drawing on tiles
TINT_COLOR = (0, 0, 0)  # Black
OPACITY = int(255 * 0.1)  # lower opacity means more transparent

# how far either side of the feature coordinates should the images extend
OFFSET_MZ_LOWER = 10.0
OFFSET_MZ_UPPER = 20.0

OFFSET_CCS_LOWER = 150
OFFSET_CCS_UPPER = 150

OFFSET_RT_LOWER = 5
OFFSET_RT_UPPER = 5


###########################
parser = argparse.ArgumentParser(description='Visualise the raw data in a precursor cuboid.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-pid','--precursor_id', type=int, help='The selected precursor.', required=True)
parser.add_argument('-px','--pixels_x', type=int, default=800, help='The dimension of the images on the x axis.', required=False)
parser.add_argument('-py','--pixels_y', type=int, default=800, help='The dimension of the images on the y axis.', required=False)
parser.add_argument('-minint','--minimum_intensity', type=int, default=100, help='The minimum intensity to be included in the image.', required=False)
parser.add_argument('-mic','--maximum_intensity_clipping', type=int, default=200, help='The maximum intensity to map before clipping.', required=False)

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

# check the precursor cuboid exists
file_names_l = glob.glob('{}/precursor-cuboids/{}/exp-{}-run-{}-precursor-{}-of-*.zip'.format(EXPERIMENT_DIR, args.run_name, args.experiment_name, args.run_name, args.precursor_id))
CUBOID_ZIP_FILE = file_names_l[0]
if not os.path.isfile(CUBOID_ZIP_FILE):
    print("The precursor cuboid zip file is required but doesn't exist: {}".format(CUBOID_ZIP_FILE))
    sys.exit(1)

# clear out any previous precursor cuboid slices
ENCODED_CUBOIDS_DIR = '{}/encoded-precursor-cuboids/{}/precursor-{}'.format(EXPERIMENT_DIR, args.run_name, args.precursor_id)
if os.path.exists(ENCODED_CUBOIDS_DIR):
    shutil.rmtree(ENCODED_CUBOIDS_DIR)
os.makedirs(ENCODED_CUBOIDS_DIR)

# create the colour mapping
colour_map = cm.batlow
norm = colors.LogNorm(vmin=args.minimum_intensity, vmax=args.maximum_intensity_clipping, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

# extract the components of the cuboid
precursor_metadata, ms1_points_df, ms2_points_df = load_precursor_cuboid_zip(CUBOID_ZIP_FILE)

# determine the cuboid dimensions
mz_lower = ms1_points_df.mz.min()
mz_upper = ms1_points_df.mz.max()
scan_lower = ms1_points_df.scan.min()
scan_upper = ms1_points_df.scan.max()
rt_lower = ms1_points_df.retention_time_secs.min()
rt_upper = ms1_points_df.retention_time_secs.max()

x_pixels_per_mz = (args.pixels_x-1) / (mz_upper - mz_lower)
y_pixels_per_scan = (args.pixels_y-1) / (scan_upper - scan_lower)

# calculate the raw point coordinates in scaled pixels
pixel_df = pd.DataFrame(ms1_points_df.apply(lambda row: pixel_xy(row.mz, row.scan, mz_lower, mz_upper, scan_lower, scan_upper), axis=1).tolist(), columns=['pixel_x','pixel_y'])
raw_pixel_df = pd.concat([ms1_points_df, pixel_df], axis=1)

# sum the intensity of raw points that have been assigned to each pixel
pixel_intensity_df = raw_pixel_df.groupby(by=['frame_id', 'pixel_x', 'pixel_y'], as_index=False).intensity.sum()

# calculate the colour to represent the intensity
colours_l = []
for i in pixel_intensity_df.intensity.unique():
    colours_l.append((i, colour_map(norm(i), bytes=True)[:3]))
colours_df = pd.DataFrame(colours_l, columns=['intensity','colour'])
pixel_intensity_df = pd.merge(pixel_intensity_df, colours_df, how='left', left_on=['intensity'], right_on=['intensity'])

# write out the images to files
slice_number = 0
for group_name,group_df in pixel_intensity_df.groupby(['frame_id'], as_index=False):
    frame_rt = ms1_points_df[(ms1_points_df.frame_id == group_name)].iloc[0].retention_time_secs
    
    # create an intensity array
    tile_im_array = np.zeros([args.pixels_y, args.pixels_x, 3], dtype=np.uint8)  # container for the image
    for r in zip(group_df.pixel_x, group_df.pixel_y, group_df.colour):
        x = r[0]
        y = r[1]
        c = r[2]
        tile_im_array[y:int(y+y_pixels_per_scan),x,:] = c

    # create an image of the intensity array
    slice_number += 1
    tile = Image.fromarray(tile_im_array, mode='RGB')
    draw = ImageDraw.Draw(tile)
    
    # draw the CCS markers
    ccs_marker_each = 10
    ccs_marker_lower = int(round(scan_lower / 10) * 10)  # round to nearest 10
    ccs_marker_upper = int(round(scan_upper / 10) * 10)
    for ccs in range(ccs_marker_lower,ccs_marker_upper,ccs_marker_each):
        pixel_y = int((ccs - scan_lower) * y_pixels_per_scan)
        draw.line((0,pixel_y, 5,pixel_y), fill='yellow', width=1)
        draw.text((10, pixel_y-6), str(ccs), font=feature_label_font, fill='yellow')
    
    # draw the m/z markers
    mz_marker_each = 0.5
    mz_marker_lower = round(mz_lower * 2) / 2  # round to nearest 0.5
    mz_marker_upper = round(mz_upper * 2) / 2
    for mz in np.arange(mz_marker_lower, mz_marker_upper, step=mz_marker_each):
        pixel_x = int((mz - mz_lower) * x_pixels_per_mz)
        draw.line((pixel_x,0, pixel_x,10), fill='yellow', width=1)
        draw.text((pixel_x-10,12), round(mz,1).astype('str'), font=feature_label_font, fill='yellow')

    # draw the info box
    info_box_x_inset = 200
    space_per_line = 12
    draw.rectangle(xy=[(args.pixels_x-info_box_x_inset, 0), (args.pixels_x, 3*space_per_line)], fill=(20,20,20), outline=None)
    draw.text((args.pixels_x-info_box_x_inset,0*space_per_line), 'precursor {}'.format(args.precursor_id), font=feature_label_font, fill='lawngreen')
    draw.text((args.pixels_x-info_box_x_inset,1*space_per_line), '{}, {}'.format(args.experiment_name, '_'.join(args.run_name.split('_Slot')[0].split('_')[1:])), font=feature_label_font, fill='lawngreen')
    draw.text((args.pixels_x-info_box_x_inset,2*space_per_line), round(frame_rt,1).astype('str'), font=feature_label_font, fill='lawngreen')
        
    # save the image as a file
    tile_file_name = '{}/precursor-cuboid-slice-{:03d}.png'.format(ENCODED_CUBOIDS_DIR, slice_number)
    tile.save(tile_file_name)
