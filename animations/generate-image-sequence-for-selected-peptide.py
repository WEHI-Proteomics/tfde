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
import json
import random
from cmcrameri import cm

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
    x_pixels_per_mz = (args.pixels_x-1) / (mz_upper - mz_lower)
    y_pixels_per_scan = (args.pixels_y-1) / (scan_upper - scan_lower)
    
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


###########################
parser = argparse.ArgumentParser(description='Visualise the raw data at a peptide\'s estimated coordinates and its extraction coordinates.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-seq','--sequence', type=str, help='The selected sequence.', required=True)
parser.add_argument('-seqchr','--sequence_charge', type=int, help='The charge for the selected sequence.', required=True)
parser.add_argument('-px','--pixels_x', type=int, default=800, help='The dimension of the images on the x axis.', required=False)
parser.add_argument('-py','--pixels_y', type=int, default=800, help='The dimension of the images on the y axis.', required=False)
parser.add_argument('-minint','--minimum_intensity', type=int, default=100, help='The minimum intensity to be included in the image.', required=False)
parser.add_argument('-oml','--offset_mz_lower', type=float, default=10.0, help='How far to the left of the selected peptide to display, in m/z.', required=False)
parser.add_argument('-omu','--offset_mz_upper', type=float, default=20.0, help='How far to the right of the selected peptide to display, in m/z.', required=False)
parser.add_argument('-osl','--offset_scan_lower', type=int, default=150, help='How far on the lower side of the selected peptide to display, in scans.', required=False)
parser.add_argument('-osu','--offset_scan_upper', type=int, default=150, help='How far on the upper side of the selected peptide to display, in scans.', required=False)
parser.add_argument('-orl','--offset_rt_lower', type=int, default=5, help='How far on the lower side of the selected peptide to display, in retention time seconds.', required=False)
parser.add_argument('-oru','--offset_rt_upper', type=int, default=5, help='How far on the upper side of the selected peptide to display, in retention time seconds.', required=False)
parser.add_argument('-mic','--maximum_intensity_clipping', type=int, default=200, help='The maximum intensity to map before clipping.', required=False)
parser.add_argument('-rn','--run_names_to_process', nargs='+', type=str, help='Space-separated names of runs to include.', required=True)

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
print("found {} runs for this experiment; processing {}".format(len(run_names), len(args.run_names_to_process)))
if not set(args.run_names_to_process).issubset(set(run_names)):
    print("Not all the runs specified are part of this experiment: {}".format(args.run_names_to_process))
    sys.exit(1)

# load the extractions
EXTRACTED_FEATURES_DB_NAME = '{}/extracted-features/extracted-features.sqlite'.format(EXPERIMENT_DIR)
if not os.path.isfile(EXTRACTED_FEATURES_DB_NAME):
    print("The extractions file doesn't exist: {}".format(EXTRACTED_FEATURES_DB_NAME))
    sys.exit(1)

print('creating indexes if not already existing on {}'.format(EXTRACTED_FEATURES_DB_NAME))
db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
src_c = db_conn.cursor()
src_c.execute("create index if not exists idx_extractions_1 on features (sequence, charge, classed_as)")
src_c.execute("create index if not exists idx_extractions_2 on features (run_name, classed_as)")
db_conn.close()

print('loading the extractions from {}'.format(EXTRACTED_FEATURES_DB_NAME))
db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
ext_df = pd.read_sql_query("select * from features where sequence == \'{}\' and charge == \'{}\' and classed_as == \'target\'".format(args.sequence, args.sequence_charge), db_conn)
db_conn.close()

# create the colour mapping
colour_map = plt.get_cmap('rainbow')
norm = colors.LogNorm(vmin=args.minimum_intensity, vmax=args.maximum_intensity_clipping, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

# clear out any previous feature slices
ENCODED_FEATURES_DIR = '{}/encoded-features'.format(EXPERIMENT_DIR)
if os.path.exists(ENCODED_FEATURES_DIR):
    shutil.rmtree(ENCODED_FEATURES_DIR)
os.makedirs(ENCODED_FEATURES_DIR)

# generate a sequence of images for the selected peptide in each run of the experiment
for run_name in args.run_names_to_process:
    print('processing {}'.format(run_name))

    CONVERTED_DB = '{}/converted-databases/exp-{}-run-{}-converted.sqlite'.format(EXPERIMENT_DIR, args.experiment_name, run_name)
    if not os.path.isfile(CONVERTED_DB):
        print("The converted database is required but doesn't exist: {}".format(CONVERTED_DB))
        sys.exit(1)

    # create the output directory
    FEATURE_SLICES_DIR = '{}/{}/slices'.format(ENCODED_FEATURES_DIR, run_name)
    os.makedirs(FEATURE_SLICES_DIR)

    extracted_sequence_in_run = ext_df[(ext_df.run_name == run_name)]
    if len(extracted_sequence_in_run) == 0:
        print('could not find the sequence in {} - moving on'.format(run_name))
        break

    # get the estimated coordinates
    estimated_coords = json.loads(extracted_sequence_in_run.iloc[0].target_coords)

    # get the extracted coordinates
    extracted_rt_apex = extracted_sequence_in_run.iloc[0].rt_apex
    extracted_scan_apex = extracted_sequence_in_run.iloc[0].scan_apex
    extracted_mz = extracted_sequence_in_run.iloc[0].monoisotopic_mz_centroid

    # determine the cuboid dimensions of the selected peptide
    mz_lower = estimated_coords['mono_mz'] - args.offset_mz_lower
    mz_upper = estimated_coords['mono_mz'] + args.offset_mz_upper
    scan_lower = estimated_coords['scan_apex'] - args.offset_scan_lower
    scan_upper = estimated_coords['scan_apex'] + args.offset_scan_upper
    rt_apex = estimated_coords['rt_apex']
    rt_lower = estimated_coords['rt_apex'] - args.offset_rt_lower
    rt_upper = estimated_coords['rt_apex'] + args.offset_rt_upper

    # find the other peptides that have an apex in this peptide's cuboid, so we can show them as well
    db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
    run_ext_df = pd.read_sql_query("select * from features where run_name == \'{}\' and classed_as == \'target\'".format(run_name), db_conn)
    db_conn.close()
    intersecting_df = run_ext_df[(run_ext_df.monoisotopic_mz_centroid >= mz_lower) & (run_ext_df.monoisotopic_mz_centroid <= mz_upper) & (run_ext_df.rt_apex >= rt_lower) & (run_ext_df.rt_apex <= rt_upper) & (run_ext_df.scan_apex >= scan_lower) & (run_ext_df.scan_apex <= scan_upper)]
    print('there are {} peptides that have an apex in this peptide\'s cuboid'.format(len(intersecting_df)))

    # pixel scaling factors
    x_pixels_per_mz = (args.pixels_x-1) / (mz_upper - mz_lower)
    y_pixels_per_scan = (args.pixels_y-1) / (scan_upper - scan_lower)

    # get the raw data for this feature
    db_conn = sqlite3.connect(CONVERTED_DB)
    raw_df = pd.read_sql_query('select mz,scan,intensity,frame_id,retention_time_secs from frames where intensity > {} and mz >= {} and mz <= {} and scan >= {} and scan <= {} and frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {}'.format(args.minimum_intensity, mz_lower, mz_upper, scan_lower, scan_upper, FRAME_TYPE_MS1, rt_lower, rt_upper), db_conn)
    db_conn.close()
    if len(raw_df) == 0:
        print("found no raw points")
        break

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
        tile_im_array = np.zeros([args.pixels_y, args.pixels_x, 3], dtype=np.uint8)  # container for the image
        for r in zip(group_df.pixel_x, group_df.pixel_y, group_df.colour):
            x = r[0]
            y = r[1]
            c = r[2]
            tile_im_array[y:int(y+y_pixels_per_scan),x,:] = c

        # create an image of the intensity array
        feature_slice += 1
        tile = Image.fromarray(tile_im_array, mode='RGB')
        draw = ImageDraw.Draw(tile)
        
        # if this is the estimated apex frame, highlight the estimated coordinates
        if group_name == apex_frame_id:
            line_colour = (100,200,100)
            draw.line((estimated_x,0, estimated_x,args.pixels_y), fill=line_colour, width=1)
            draw.line((0,estimated_y, args.pixels_x,estimated_y), fill=line_colour, width=1)
        
        # if this is the extracted apex frame, highlight the extracted coordinates
        if group_name == extracted_apex_frame_id:
            # draw the extracted apex
            extracted_x, extracted_y = pixel_xy(extracted_mz, extracted_scan_apex, mz_lower, mz_upper, scan_lower, scan_upper)
            line_colour = (100,100,200)
            draw.line((extracted_x,0, extracted_x,args.pixels_y), fill=line_colour, width=1)
            draw.line((0,extracted_y, args.pixels_x,extracted_y), fill=line_colour, width=1)

        # draw the CCS markers
        ccs_marker_each = 10
        draw.line((0,estimated_y, 10,estimated_y), fill='yellow', width=3)
        draw.text((15, estimated_y-6), str(round(estimated_coords['scan_apex'],1)), font=feature_label_font, fill='yellow')
        for i in range(1,10):
            ccs_marker_y_pixels = y_pixels_per_scan * ccs_marker_each * i
            draw.line((0,estimated_y-ccs_marker_y_pixels, 5,estimated_y-ccs_marker_y_pixels), fill='yellow', width=1)
            draw.line((0,estimated_y+ccs_marker_y_pixels, 5,estimated_y+ccs_marker_y_pixels), fill='yellow', width=1)

        # draw the m/z markers
        mz_marker_each = 1
        draw.line((estimated_x,0, estimated_x,10), fill='yellow', width=3)
        draw.text((estimated_x-10,12), str(round(estimated_coords['mono_mz'],1)), font=feature_label_font, fill='yellow')
        for i in range(1,10):
            mz_marker_x_pixels = x_pixels_per_mz * mz_marker_each * i
            draw.line((estimated_x-mz_marker_x_pixels,0, estimated_x-mz_marker_x_pixels,5), fill='yellow', width=1)
            draw.line((estimated_x+mz_marker_x_pixels,0, estimated_x+mz_marker_x_pixels,5), fill='yellow', width=1)

        # draw the info box
        info_box_x_inset = 200
        space_per_line = 12
        draw.rectangle(xy=[(args.pixels_x-info_box_x_inset, 0), (args.pixels_x, 4*space_per_line)], fill=(20,20,20), outline=None)
        draw.text((args.pixels_x-info_box_x_inset,0*space_per_line), args.sequence, font=feature_label_font, fill='lawngreen')
        draw.text((args.pixels_x-info_box_x_inset,1*space_per_line), 'charge {}'.format(args.sequence_charge), font=feature_label_font, fill='lawngreen')
        draw.text((args.pixels_x-info_box_x_inset,2*space_per_line), '{}, {}'.format(args.experiment_name, '_'.join(run_name.split('_Slot')[0].split('_')[1:])), font=feature_label_font, fill='lawngreen')
        draw.text((args.pixels_x-info_box_x_inset,3*space_per_line), str(round(frame_rt,1)), font=feature_label_font, fill='lawngreen')

        # draw the other extractions that have an apex in this frame
        rt_wobble = 1.0
        radius_px = 10
        other_ext_df = intersecting_df[(intersecting_df.rt_apex >= frame_rt-rt_wobble) & (intersecting_df.rt_apex <= frame_rt+rt_wobble)]
        for row in other_ext_df.itertuples():
            px_x, px_y = pixel_xy(row.monoisotopic_mz_centroid, row.scan_apex, mz_lower, mz_upper, scan_lower, scan_upper)
            draw.ellipse((px_x-radius_px, px_y-radius_px, px_x+radius_px, px_y+radius_px), fill = None, outline ='orange')
            draw.text((px_x+radius_px+2,px_y-6), row.sequence, font=feature_label_font, fill='lawngreen')
            
        # save the image as a file
        tile_file_name = '{}/feature-slice-{:03d}.png'.format(FEATURE_SLICES_DIR, feature_slice)
        tile.save(tile_file_name)
