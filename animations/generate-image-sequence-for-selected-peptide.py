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


def get_run_names(experiment_dir):

    # process all the runs
    database_names_l = glob.glob("{}/converted-databases/*.sqlite".format(experiment_dir))

    # convert the raw databases
    run_names = []
    for database_name in database_names_l:
        run_name = os.path.basename(database_name).split('.sqlite')[0].split('run-')[1].split('-converted')[0]
        run_names.append(run_name)

    return run_names

def pixel_xy(mz, scan, mz_lower, mz_upper, scan_lower, scan_upper):
    x_pixels_per_mz = (PIXELS_X-1) / (mz_upper - mz_lower)
    y_pixels_per_scan = (PIXELS_Y-1) / (scan_upper - scan_lower)
    
    pixel_x = int((mz - mz_lower) * x_pixels_per_mz)
    pixel_y = int((scan - scan_lower) * y_pixels_per_scan)
    return (pixel_x, pixel_y)



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

# image dimensions
PIXELS_X = 600
PIXELS_Y = 600


###########################
parser = argparse.ArgumentParser(description='Visualise the raw data at a peptide\'s estimated coordinates and its extraction coordinates.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-seq','--sequence', type=str, help='The selected sequence.', required=True)
parser.add_argument('-seqchr','--sequence_charge', type=int, help='The charge for the selected sequence.', required=True)

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

# get the run names for the experiment
run_names = get_run_names(EXPERIMENT_DIR)
print("found {} runs for this experiment: {}".format(len(run_names), run_names))

# create the colour mapping
colour_map = plt.get_cmap('rainbow')
norm = colors.LogNorm(vmin=1, vmax=5000, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

# generate a sequence of images for the selected peptide in each run of the experiment
for run_name in run_names:

    CONVERTED_DB = '{}/converted-databases/exp-{}-run-{}-converted.sqlite'.format(EXPERIMENT_DIR, args.experiment_name, run_name)
    ENCODED_FEATURES_DIR = '{}/encoded-features/{}'.format(EXPERIMENT_DIR, run_name)
    FEATURE_SLICES_DIR = '{}/slices'.format(ENCODED_FEATURES_DIR)

    # clear out any previous feature slices
    if os.path.exists(FEATURE_SLICES_DIR):
        shutil.rmtree(FEATURE_SLICES_DIR)
    os.makedirs(FEATURE_SLICES_DIR)

    estimated_coords_df = pd.read_pickle('{}/target-decoy-models/library-sequences-in-run-{}.pkl'.format(EXPERIMENT_DIR, run_name))
    estimated_coords = estimated_coords_df[(estimated_coords_df.sequence == args.sequence) & (estimated_coords_df.charge == args.sequence_charge)].iloc[0].target_coords

    extracted_coords = estimated_coords_df[(estimated_coords_df.sequence == args.sequence) & (estimated_coords_df.charge == args.sequence_charge)].iloc[0].attributes
    extracted_rt_apex = extracted_coords['rt_apex']
    extracted_scan_apex = extracted_coords['scan_apex']
    extracted_mz = extracted_coords['monoisotopic_mz_centroid']

    # determine the cuboid dimensions
    mz_lower = estimated_coords['mono_mz'] - OFFSET_MZ_LOWER
    mz_upper = estimated_coords['mono_mz'] + OFFSET_MZ_UPPER
    scan_lower = estimated_coords['scan_apex'] - OFFSET_CCS_LOWER
    scan_upper = estimated_coords['scan_apex'] + OFFSET_CCS_UPPER
    rt_apex = estimated_coords['rt_apex']
    rt_lower = estimated_coords['rt_apex'] - OFFSET_RT_LOWER
    rt_upper = estimated_coords['rt_apex'] + OFFSET_RT_UPPER

    x_pixels_per_mz = (PIXELS_X-1) / (mz_upper - mz_lower)
    y_pixels_per_scan = (PIXELS_Y-1) / (scan_upper - scan_lower)

    # get the raw data for this feature
    db_conn = sqlite3.connect(CONVERTED_DB)
    raw_df = pd.read_sql_query('select mz,scan,intensity,frame_id,retention_time_secs from frames where intensity > 100 and mz >= {} and mz <= {} and scan >= {} and scan <= {} and frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {}'.format(mz_lower, mz_upper, scan_lower, scan_upper, FRAME_TYPE_MS1, rt_lower, rt_upper), db_conn)
    if len(raw_df) == 0:
        print("found no raw points")
        sys.exit(1)
    db_conn.close()

    # get the frame ID closest to the estimated RT apex
    apex_frame_id = int(raw_df.iloc[(raw_df['retention_time_secs'] - rt_apex).abs().argsort()[:1]].sort_values(by=['retention_time_secs'], ascending=[True], inplace=False).iloc[0].frame_id)

    # get the frame ID closest to the extracted RT apex
    extracted_apex_frame_id = int(raw_df.iloc[(raw_df['retention_time_secs'] - extracted_rt_apex).abs().argsort()[:1]].sort_values(by=['retention_time_secs'], ascending=[True], inplace=False).iloc[0].frame_id)

    # calculate the raw point coordinates in scaled pixels
    pixel_df = pd.DataFrame(raw_df.apply(lambda row: pixel_xy(row.mz, row.scan, mz_lower, mz_upper, scan_lower, scan_upper), axis=1).tolist(), columns=['pixel_x','pixel_y'])
    raw_pixel_df = pd.concat([raw_df, pixel_df], axis=1)

    # sum the intensity of raw points that have been assigned to each pixel
    pixel_intensity_df = raw_pixel_df.groupby(by=['frame_id', 'pixel_x', 'pixel_y'], as_index=False).intensity.sum()

    # calculate the colour to represent the intensity
    colours_l = []
    for i in pixel_intensity_df.intensity.unique():
        colours_l.append((i, colour_map(norm(i), bytes=True)[:3]))
    colours_df = pd.DataFrame(colours_l, columns=['intensity','colour'])
    pixel_intensity_df = pd.merge(pixel_intensity_df, colours_df, how='left', left_on=['intensity'], right_on=['intensity'])

    estimated_x, estimated_y = pixel_xy(estimated_coords['mono_mz'], estimated_coords['scan_apex'], mz_lower, mz_upper, scan_lower, scan_upper)

    # write out the images to files
    feature_slice = 0
    for group_name,group_df in pixel_intensity_df.groupby(['frame_id'], as_index=False):
        frame_rt = raw_df[(raw_df.frame_id == group_name)].iloc[0].retention_time_secs
        
        # create an intensity array
        tile_im_array = np.zeros([PIXELS_Y, PIXELS_X, 3], dtype=np.uint8)  # container for the image
        for r in zip(group_df.pixel_x, group_df.pixel_y, group_df.colour):
            x = r[0]
            y = r[1]
            c = r[2]
            tile_im_array[y:int(y+y_pixels_per_scan),x,:] = c

        # create an image of the intensity array
        feature_slice += 1
        tile = Image.fromarray(tile_im_array, 'RGB')
        draw = ImageDraw.Draw(tile)
        
        # if this is the estimated apex frame, highlight the estimated coordinates
        if group_name == apex_frame_id:
            draw.line((estimated_x,0, estimated_x,PIXELS_Y), fill='green', width=1)
            draw.line((0,estimated_y, PIXELS_X,estimated_y), fill='green', width=1)
        
        # if this is the extracted apex frame, highlight the extracted coordinates
        if group_name == extracted_apex_frame_id:
            # draw the extracted apex
            extracted_x, extracted_y = pixel_xy(extracted_coords['monoisotopic_mz_centroid'], extracted_coords['scan_apex'], mz_lower, mz_upper, scan_lower, scan_upper)
            draw.line((extracted_x,0, extracted_x,PIXELS_Y), fill='blue', width=1)
            draw.line((0,extracted_y, PIXELS_X,extracted_y), fill='blue', width=1)

        # draw the CCS markers
        ccs_marker_each = 10
        draw.line((0,estimated_y, 10,estimated_y), fill='yellow', width=3)
        draw.text((15, estimated_y-6), round(estimated_coords['scan_apex'],1).astype('str'), font=feature_label_font, fill='yellow')
        for i in range(1,10):
            ccs_marker_y_pixels = y_pixels_per_scan * ccs_marker_each * i
            draw.line((0,estimated_y-ccs_marker_y_pixels, 5,estimated_y-ccs_marker_y_pixels), fill='yellow', width=1)
            draw.line((0,estimated_y+ccs_marker_y_pixels, 5,estimated_y+ccs_marker_y_pixels), fill='yellow', width=1)

        # draw the m/z markers
        mz_marker_each = 1
        draw.line((estimated_x,0, estimated_x,10), fill='yellow', width=3)
        draw.text((estimated_x-10,12), round(estimated_coords['mono_mz'],1).astype('str'), font=feature_label_font, fill='yellow')
        for i in range(1,10):
            mz_marker_x_pixels = x_pixels_per_mz * mz_marker_each * i
            draw.line((estimated_x-mz_marker_x_pixels,0, estimated_x-mz_marker_x_pixels,5), fill='yellow', width=1)
            draw.line((estimated_x+mz_marker_x_pixels,0, estimated_x+mz_marker_x_pixels,5), fill='yellow', width=1)

        # draw the info box
        info_box_x_inset = 200
        space_per_line = 12
        draw.rectangle(xy=[(PIXELS_X-info_box_x_inset, 0), (PIXELS_X, 4*space_per_line)], fill=TINT_COLOR+(OPACITY,), outline=None)
        draw.text((PIXELS_X-info_box_x_inset,0*space_per_line), args.sequence, font=feature_label_font, fill='lawngreen')
        draw.text((PIXELS_X-info_box_x_inset,1*space_per_line), 'charge {}'.format(args.sequence_charge), font=feature_label_font, fill='lawngreen')
        draw.text((PIXELS_X-info_box_x_inset,2*space_per_line), '{}, {}'.format(args.experiment_name, '_'.join(run_name.split('_Slot')[0].split('_')[1:])), font=feature_label_font, fill='lawngreen')
        draw.text((PIXELS_X-info_box_x_inset,3*space_per_line), round(frame_rt,1).astype('str'), font=feature_label_font, fill='lawngreen')
            
        # save the image as a file
        tile_file_name = '{}/feature-slice-{:03d}.png'.format(FEATURE_SLICES_DIR, feature_slice)
        tile.save(tile_file_name)
        