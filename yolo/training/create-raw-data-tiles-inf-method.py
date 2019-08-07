import sqlite3
import pandas as pd
import numpy as np
import sys
from matplotlib import colors, cm, pyplot as plt
import argparse
import ray
import os, shutil
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import time
import statistics

MS1_CE = 10

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-cdb','--converted_database', type=str, help='Path to the raw converted database.', required=True)
parser.add_argument('-tb','--tile_base', type=str, help='Path to the base directory of the training set.', required=True)
parser.add_argument('-ff','--features_file', type=str, help='Path to the pickle file containing the features detected.', required=True)
parser.add_argument('-rtl','--rt_lower', type=int, help='Lower bound of the RT range.', required=True)
parser.add_argument('-rtu','--rt_upper', type=int, help='Upper bound of the RT range.', required=True)
parser.add_argument('-tm','--test_mode', action='store_true', help='A small subset of the data for testing purposes.')
parser.add_argument('-agm','--adjust_to_global_mobility_median', action='store_true', help='Align with the global mobility median, for video purposes.')
args = parser.parse_args()

PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(args.tile_base)
OVERLAY_FILES_DIR = '{}/overlay'.format(args.tile_base)

UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'

start_run = time.time()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))

print("opening {}".format(args.converted_database))
db_conn = sqlite3.connect(args.converted_database)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(args.rt_lower, args.rt_upper, MS1_CE), db_conn)
ms1_frame_ids = tuple(ms1_frame_properties_df.frame_id)
db_conn.close()

frame_delay = ms1_frame_properties_df.iloc[1].retention_time_secs - ms1_frame_properties_df.iloc[0].retention_time_secs

if args.adjust_to_global_mobility_median:
    # calculate the median of the mobility through the whole extent in RT
    db_conn = sqlite3.connect(args.converted_database)
    points_df = pd.read_sql_query("select frame_id,scan from frames where frame_id in {}".format(ms1_frame_ids), db_conn)
    global_mobility_median = statistics.median(points_df.scan)
    del points_df
    db_conn.close()

def delta_scan_for_frame(frame_raw_scans):
    frame_mobility_median = statistics.median(frame_raw_scans)
    delta_scan = int(global_mobility_median - frame_mobility_median)
    return delta_scan

MZ_MIN = 100.0
MZ_MAX = 1700.0
MZ_BIN_WIDTH = 0.1
SCAN_MIN = 1
SCAN_MAX = 910
SCAN_LENGTH_MINIMUM = SCAN_MAX * 0.05  # filter out the small-extent features

mz_bins = np.arange(start=MZ_MIN, stop=MZ_MAX+MZ_BIN_WIDTH, step=MZ_BIN_WIDTH)  # go slightly wider to accommodate the maximum value

MZ_BIN_COUNT = len(mz_bins)
DELTA_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
MZ_TOLERANCE_PPM = 5
MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4
MIN_ISOTOPE_CORRELATION = 0.9
RT_EACH_SIDE = 1.0  # proportion of RT length / 2 used for the bounding box

# load the features detected
features_df = pd.read_pickle(args.features_file)
features_df['mz_lower'] = features_df.envelope.apply(lambda x: x[0][0])
features_df['mz_upper'] = features_df.envelope.apply(lambda x: x[len(x)-1][0])
features_df['isotope_count'] = features_df.envelope.apply(lambda x: len(x))
features_df = features_df[(features_df.isotope_count > 2) & ((features_df.scan_upper - features_df.scan_lower) > SCAN_LENGTH_MINIMUM)]

print("charge states: {} to {}".format(features_df.charge.min(), features_df.charge.max()))

MAX_CHARGE_STATE = int(features_df.charge.max())

mz_ppm_tolerance_l = []
binned_mz_lower_l = []
binned_mz_upper_l = []
binned_mz_idx_lower_l = []
binned_mz_idx_upper_l = []
scan_lower_l = []
scan_upper_l = []

# preprocess the features by calculating their rectangles
for r in zip(features_df.feature_id,features_df.charge,features_df.isotope_count,features_df.mz_lower,features_df.mz_upper,features_df.scan_lower,features_df.scan_upper):
    feature_id = int(r[0])
    charge_state = int(r[1])
    isotope_count = int(r[2])
    mz_lower = r[3]
    mz_upper = r[4]
    scan_lower = r[5]
    scan_upper = r[6]

    # determine the bounding box coordinates for m/z and scan in real space
    mz_ppm_tolerance = mz_lower * MZ_TOLERANCE_PERCENT / 100
    mz_ppm_tolerance_l.append((mz_ppm_tolerance))

    # find the bin edges for the feature's mz
    binned_mz_idx_lower = int(np.digitize(mz_lower, mz_bins))-1
    binned_mz_idx_upper = int(np.digitize(mz_upper, mz_bins))
    rect_mz_lower = mz_bins[binned_mz_idx_lower]
    rect_mz_upper = mz_bins[binned_mz_idx_upper]
    rect_mz_range = rect_mz_upper - rect_mz_lower
    binned_mz_lower_l.append(rect_mz_lower)
    binned_mz_upper_l.append(rect_mz_upper)
    binned_mz_idx_lower_l.append(binned_mz_idx_lower)
    binned_mz_idx_upper_l.append(binned_mz_idx_upper)

features_df['mz_ppm_tolerance'] = mz_ppm_tolerance_l
features_df['binned_rect_mz_lower'] = binned_mz_lower_l
features_df['binned_rect_mz_upper'] = binned_mz_upper_l
features_df['binned_rect_mz_idx_lower'] = binned_mz_idx_lower_l
features_df['binned_rect_mz_idx_upper'] = binned_mz_idx_upper_l

PIXELS_PER_MZ_BIN = 5
PIXELS_PER_SCAN = 1

# will stretch the image to these dimensions
TILE_HEIGHT = SCAN_MAX
TILE_WIDTH = TILE_HEIGHT

PIXELS_X = MZ_BIN_COUNT * PIXELS_PER_MZ_BIN
PIXELS_Y = SCAN_MAX * PIXELS_PER_SCAN
MZ_BINS_PER_TILE = int(TILE_WIDTH / PIXELS_PER_MZ_BIN)
TILES_PER_FRAME = int(MZ_BIN_COUNT / MZ_BINS_PER_TILE)
RESIZE_FACTOR_X = TILE_WIDTH / MZ_BINS_PER_TILE

# ### Generate tiles for all frames

# initialise the directories required for the data set creation
if os.path.exists(args.tile_base):
    shutil.rmtree(args.tile_base)
os.makedirs(args.tile_base)

if os.path.exists(PRE_ASSIGNED_FILES_DIR):
    shutil.rmtree(PRE_ASSIGNED_FILES_DIR)
os.makedirs(PRE_ASSIGNED_FILES_DIR)

if os.path.exists(OVERLAY_FILES_DIR):
    shutil.rmtree(OVERLAY_FILES_DIR)
os.makedirs(OVERLAY_FILES_DIR)

# calculate the colour to represent the intensity
colour_map = cm.get_cmap(name='magma')
norm = colors.LogNorm(vmin=1, vmax=1e4, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

if not ray.is_initialized():
    ray.init()

@ray.remote
def render_tile_for_frame(frame_r):
    frame_id = int(frame_r[0])
    frame_rt = frame_r[1]

    print("processing frame {}".format(frame_id))

    # count the instances by class
    instances_df = pd.DataFrame([(x,0) for x in range(1,MAX_CHARGE_STATE+1)], columns=['charge','instances'])

    # get the features overlapping this frame
    features_frame_overlap_df = features_df[(features_df.rt_lower <= frame_rt) & (features_df.rt_upper >= frame_rt)].copy()

    # load the raw frame points
    db_conn = sqlite3.connect(args.converted_database)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {}".format(frame_id), db_conn)
    db_conn.close()

    scan_delta = delta_scan_for_frame(frame_raw_scans=raw_points_df.scan)

    if args.adjust_to_global_mobility_median:
        # adjust the mobility of all the raw points in the frame to align with the global mobility
        raw_points_df.scan += scan_delta

    # convert the raw points to an intensity array
    frame_intensity_array = np.zeros([SCAN_MAX+1, MZ_BIN_COUNT+1], dtype=np.uint16)  # scratchpad for the intensity value prior to image conversion
    for r in zip(raw_points_df.mz,raw_points_df.scan,raw_points_df.intensity):
        mz = r[0]
        scan = int(r[1])
        if (mz >= MZ_MIN) and (mz <= MZ_MAX) and (scan >= SCAN_MIN) and (scan <= SCAN_MAX):
            mz_array_idx = int(np.digitize(mz, mz_bins))-1
            scan_array_idx = scan
            intensity = int(r[2])
            frame_intensity_array[scan_array_idx,mz_array_idx] += intensity

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
    frame_im_array = np.zeros([TILE_HEIGHT+1, MZ_BIN_COUNT+1, 3], dtype=np.uint8)  # container for the image
    for r in zip(intensity_df.x, intensity_df.y, intensity_df.colour):
        x = r[0]
        y = r[1]
        c = r[2]
        frame_im_array[y,x,:] = c

    # load the font to use for labelling the overlays
    if os.path.isfile(UBUNTU_FONT_PATH):
        feature_label = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
    else:
        feature_label = ImageFont.truetype(MACOS_FONT_PATH, 10)

    # write out the image tiles for the frame
    for tile_idx in range(TILES_PER_FRAME):
        # tile m/z coordinates
        tile_base_mz = mz_bins[tile_idx * MZ_BINS_PER_TILE]
        tile_width_mz = MZ_BINS_PER_TILE * MZ_BIN_WIDTH
        # tile index coordinates
        tile_idx_base = int(tile_idx * MZ_BINS_PER_TILE)
        tile_idx_width = MZ_BINS_PER_TILE
        # extract the subset of the frame for this image
        tile_im_array = frame_im_array[:,tile_idx_base:tile_idx_base+tile_idx_width,:]
        tile = Image.fromarray(tile_im_array, 'RGB')
        tile_with_overlay = Image.fromarray(tile_im_array, 'RGB')

        # stretch the image to be square
        stretched_tile = tile.resize((TILE_WIDTH, TILE_HEIGHT))
        stretched_tile_with_overlay = tile_with_overlay.resize((TILE_WIDTH, TILE_HEIGHT))

        # get the features that fully fit in the tile
        feature_coordinates = []
        ap_df = features_frame_overlap_df
        for feature_r in zip(ap_df.feature_id, ap_df.charge, ap_df.isotope_count, ap_df.binned_rect_mz_idx_lower, ap_df.binned_rect_mz_idx_upper, ap_df.scan_lower, ap_df.scan_upper):
            feature_id = int(feature_r[0])
            feature_charge_state = int(feature_r[1])
            isotope_count = int(feature_r[2])
            binned_rect_mz_idx_lower = int(feature_r[3]) - 1  # go a bit wider in m/z to make sure we get the whole width
            binned_rect_mz_idx_upper = int(feature_r[4]) + 1
            scan_lower = int(feature_r[5]) - 2  # and a bit wider in mobility for a bigger margin
            scan_upper = int(feature_r[6]) + 2

            if args.adjust_to_global_mobility_median:
                # adjust the mobility of all the raw points in the frame to align with the global mobility
                scan_lower += scan_delta
                scan_upper += scan_delta

            # draw the features overlay
            draw = ImageDraw.Draw(stretched_tile_with_overlay)
            x0 = (binned_rect_mz_idx_lower - tile_idx_base) * RESIZE_FACTOR_X
            x1 = (binned_rect_mz_idx_upper - tile_idx_base) * RESIZE_FACTOR_X
            y0 = scan_lower
            y1 = scan_upper
            # text file coordinates
            x = (x0 + ((x1 - x0) / 2)) / TILE_WIDTH  # YOLO expects x,y to be the centre point of the object
            y = (y0 + ((y1 - y0) / 2)) / TILE_HEIGHT
            width = (x1 - x0) / TILE_WIDTH
            height = (y1 - y0) / TILE_HEIGHT
            object_class = feature_charge_state-1
            # object_class = 0
            # draw the MQ feature if its centre is within the tile
            if ((x > 0) and (x < 1) and (y > 0) and (y < 1)):
                draw.rectangle(xy=[(x0, y0), (x1, y1)], fill=None, outline='red')
                draw.text((x0, y0-12), "{}, +{}, {} iso".format(feature_id,feature_charge_state,isotope_count), font=feature_label, fill='red')
                feature_coordinates.append(("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(object_class, x, y, width, height)))
                instances_df.loc[(instances_df.charge == feature_charge_state),'instances'] += 1

        # write them out
        train_filename = '{}/frame-{}-tile-{}-mz-{}-{}.png'.format(PRE_ASSIGNED_FILES_DIR, frame_id, tile_idx, int(tile_base_mz), int(tile_base_mz+tile_width_mz))
        train_text_filename = '{}/frame-{}-tile-{}-mz-{}-{}.txt'.format(PRE_ASSIGNED_FILES_DIR, frame_id, tile_idx, int(tile_base_mz), int(tile_base_mz+tile_width_mz))
        overlay_filename = '{}/frame-{}-tile-{}-mz-{}-{}.png'.format(OVERLAY_FILES_DIR, frame_id, tile_idx, int(tile_base_mz), int(tile_base_mz+tile_width_mz))
        stretched_tile.save(train_filename)
        stretched_tile_with_overlay.save(overlay_filename)
        # write the text file
        with open(train_text_filename, 'w') as f:
            for item in feature_coordinates:
                f.write("%s\n" % item)

    return instances_df


if args.test_mode:
    ms1_frame_properties_df = ms1_frame_properties_df[:20]

instances_l = ray.get([render_tile_for_frame.remote(frame_r) for frame_r in zip(ms1_frame_properties_df.frame_id, ms1_frame_properties_df.retention_time_secs)])
print("labelled instances: {}, total {}".format(list(sum(instances_l).instances), sum(instances_l).instances.sum()))

stop_run = time.time()
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))
print("{} info: {}".format(parser.prog, info))

print("shutting down ray")
ray.shutdown()
