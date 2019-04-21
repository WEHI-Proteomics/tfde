import sqlite3
import pandas as pd
import numpy as np
import sys
from matplotlib import colors, cm, pyplot as plt
import argparse
import ray
import os, shutil
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


MS1_CE = 10

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-rtl','--rt_lower', type=int, help='Lower bound of the RT range.', required=True)
parser.add_argument('-rtu','--rt_upper', type=int, help='Upper bound of the RT range.', required=True)
args = parser.parse_args()

BASE_NAME = "/home/ubuntu/HeLa_20KInt-rt-{}-{}".format(args.rt_lower, args.rt_upper)
# BASE_NAME = "/Users/darylwilding-mcbride/Downloads/HeLa_20KInt-rt-{}-{}".format(args.rt_lower, args.rt_upper)
CONVERTED_DATABASE_NAME = '{}/HeLa_20KInt.sqlite'.format(BASE_NAME)
ALLPEPTIDES_FILENAME = '/home/ubuntu/maxquant_results/txt/allPeptides.txt'
# ALLPEPTIDES_FILENAME = '/Users/darylwilding-mcbride/Downloads/maxquant_results/txt/allPeptides.txt'

# TILE_BASE = '/Users/darylwilding-mcbride/Downloads/yolo-train'
TILE_BASE = '/home/ubuntu/yolo-train-rt-{}-{}'.format(args.rt_lower, args.rt_upper)
PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TILE_BASE)
OVERLAY_FILES_DIR = '{}/overlay'.format(TILE_BASE)

# if not input("This will erase the overlay and pre-assigned directories in {}. Are you sure? (y/n): ".format(TILE_BASE)).lower().strip()[:1] == "y": sys.exit(1)

db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(args.rt_lower, args.rt_upper, MS1_CE), db_conn)
db_conn.close()

frame_delay = ms1_frame_properties_df.iloc[1].retention_time_secs - ms1_frame_properties_df.iloc[0].retention_time_secs

MZ_MIN = 100.0
MZ_MAX = 1700.0
MZ_BIN_WIDTH = 0.1
SCAN_MIN = 1
SCAN_MAX = 910

mz_bins = np.arange(start=MZ_MIN, stop=MZ_MAX+MZ_BIN_WIDTH, step=MZ_BIN_WIDTH)  # go slightly wider to accomodate the maximum value

MZ_BIN_COUNT = len(mz_bins)
DELTA_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
MZ_TOLERANCE_PPM = 5
MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4
MIN_ISOTOPE_CORRELATION = 0.9
RT_EACH_SIDE = 1.0  # proportion of RT length / 2 used for the bounding box

allpeptides_df = pd.read_csv(ALLPEPTIDES_FILENAME, sep='\t')
allpeptides_df.rename(columns={'Number of isotopic peaks':'isotope_count', 'm/z':'mz', 'Number of data points':'number_data_points', 'Intensity':'intensity', 'Ion mobility index':'scan', 'Ion mobility index length':'scan_length', 'Ion mobility index length (FWHM)':'scan_length_fwhm', 'Retention time':'rt', 'Retention length':'rt_length', 'Retention length (FWHM)':'rt_length_fwhm', 'Charge':'charge_state', 'Number of pasef MS/MS':'number_pasef_ms2_ids', 'Isotope correlation':'isotope_correlation'}, inplace=True)
allpeptides_df = allpeptides_df[allpeptides_df.intensity.notnull()].copy()  # remove all the null intensity rows
allpeptides_df = allpeptides_df[allpeptides_df.intensity.notnull() & (allpeptides_df.isotope_correlation >= MIN_ISOTOPE_CORRELATION) & (allpeptides_df.rt >= args.rt_lower) & (allpeptides_df.rt <= args.rt_upper)].copy()

allpeptides_df["rt_delta"] = allpeptides_df.rt_length / 2
allpeptides_df["rt_lower"] = allpeptides_df.rt - (allpeptides_df.rt_delta * RT_EACH_SIDE)
allpeptides_df["rt_upper"] = allpeptides_df.rt + (allpeptides_df.rt_delta * RT_EACH_SIDE)

# sort the features by decreasing intensity and give them an ID
allpeptides_df.sort_values(by=['intensity'], ascending=False, inplace=True)
allpeptides_df["mq_feature_id"] = np.arange(start=1, stop=len(allpeptides_df)+1)

print("charge states: {} to {}".format(allpeptides_df.charge_state.min(), allpeptides_df.charge_state.max()))

MAX_CHARGE_STATE = int(allpeptides_df.charge_state.max())

# ### Calculate the binned rectangle coordinates for all the MQ features

mz_ppm_tolerance_l = []
binned_mz_lower_l = []
binned_mz_upper_l = []
binned_mz_idx_lower_l = []
binned_mz_idx_upper_l = []
scan_lower_l = []
scan_upper_l = []

# calculate the MQ feature rectangles
for r in zip(allpeptides_df.mq_feature_id,allpeptides_df.charge_state,allpeptides_df.isotope_count,allpeptides_df.mz,allpeptides_df.scan,allpeptides_df.scan_length):
    mq_feature_id = int(r[0])
    charge_state = int(r[1])
    isotope_count = int(r[2])
    mq_feature_mz = r[3]
    mq_feature_scan = int(r[4])
    mq_feature_scan_length = int(r[5])

    expected_isotope_spacing_mz = DELTA_MZ / charge_state

    # determine the bounding box coordinates for m/z and scan in real space
    mz_ppm_tolerance = mq_feature_mz * MZ_TOLERANCE_PERCENT / 100
    mz_ppm_tolerance_l.append((mz_ppm_tolerance))
    
    # find the bin edges for the feature's mz
    binned_mz_idx_lower = int(np.digitize(mq_feature_mz, mz_bins))-1
    binned_mz_idx_upper = int(np.digitize(mq_feature_mz + ((isotope_count-1) * expected_isotope_spacing_mz), mz_bins))
    rect_mz_lower = mz_bins[binned_mz_idx_lower]
    rect_mz_upper = mz_bins[binned_mz_idx_upper]
    rect_mz_range = rect_mz_upper - rect_mz_lower
    binned_mz_lower_l.append(rect_mz_lower)
    binned_mz_upper_l.append(rect_mz_upper)
    binned_mz_idx_lower_l.append(binned_mz_idx_lower)
    binned_mz_idx_upper_l.append(binned_mz_idx_upper)

    rect_scan = mq_feature_scan
    rect_scan_delta = mq_feature_scan_length / 2
    rect_scan_lower = int(rect_scan - rect_scan_delta)
    rect_scan_upper = int(rect_scan + rect_scan_delta)
    rect_scan_range = int(mq_feature_scan_length)
    scan_lower_l.append(rect_scan_lower)
    scan_upper_l.append(rect_scan_upper)
    
allpeptides_df['mz_ppm_tolerance'] = mz_ppm_tolerance_l
allpeptides_df['binned_rect_mz_lower'] = binned_mz_lower_l
allpeptides_df['binned_rect_mz_upper'] = binned_mz_upper_l
allpeptides_df['binned_rect_mz_idx_lower'] = binned_mz_idx_lower_l
allpeptides_df['binned_rect_mz_idx_upper'] = binned_mz_idx_upper_l
allpeptides_df['scan_lower'] = scan_lower_l
allpeptides_df['scan_upper'] = scan_upper_l

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
if os.path.exists(TILE_BASE):
    shutil.rmtree(TILE_BASE)
os.makedirs(TILE_BASE)

if os.path.exists(PRE_ASSIGNED_FILES_DIR):
    shutil.rmtree(PRE_ASSIGNED_FILES_DIR)
os.makedirs(PRE_ASSIGNED_FILES_DIR)

if os.path.exists(OVERLAY_FILES_DIR):
    shutil.rmtree(OVERLAY_FILES_DIR)
os.makedirs(OVERLAY_FILES_DIR)

# calculate the colour to represent the intensity
a = np.arange(start=0, stop=200, step=2, dtype=np.int)  # use up the darker colours for the low intensity points
b = np.arange(start=200, stop=13000, step=200, dtype=np.int)
bounds = np.concatenate([a,b])
colour_map = cm.get_cmap(name='magma', lut=len(bounds))
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=colour_map.N, clip=True)

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
    allpeptides_frame_overlap_df = allpeptides_df[(allpeptides_df.rt_lower <= frame_rt) & (allpeptides_df.rt_upper >= frame_rt)].copy()

    # load the raw frame points
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {}".format(frame_id), db_conn)
    db_conn.close()
    
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
    # feature_label = ImageFont.truetype('/Library/Fonts/Arial.ttf', 10)
    feature_label = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 10)

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

        # get the MQ features that fully fit in the tile
        feature_coordinates = []
        ap_df = allpeptides_frame_overlap_df
        for feature_r in zip(ap_df.mq_feature_id, ap_df.mz, ap_df.charge_state, ap_df.isotope_count, ap_df.binned_rect_mz_idx_lower, ap_df.binned_rect_mz_idx_upper, ap_df.scan_lower, ap_df.scan_upper):
            mq_feature_id = int(feature_r[0])
            mq_feature_mz = feature_r[1]
            mq_feature_charge_state = int(feature_r[2])
            isotope_count = int(feature_r[3])
            binned_rect_mz_idx_lower = int(feature_r[4]) - 1  # go a bit wider in m/z to make sure we get the whole width
            binned_rect_mz_idx_upper = int(feature_r[5]) + 1
            scan_lower = int(feature_r[6]) - 2  # and a bit wider in mobility for a bigger margin
            scan_upper = int(feature_r[7]) + 2

            # draw the MQ features overlay
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
            object_class = mq_feature_charge_state-1
            # draw the MQ feature if its centre is within the tile
            if ((x >= 0) and (x <= 1) and (y >= 0) and (y <= 1)):
                draw.rectangle(xy=[(x0, y0), (x1, y1)], fill=None, outline='red')
                draw.text((x0, y0-12), "{}, +{}, {} iso".format(mq_feature_id,mq_feature_charge_state,isotope_count), font=feature_label, fill='red')
                feature_coordinates.append(("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(object_class, x, y, width, height)))
                instances_df.loc[(instances_df.charge == mq_feature_charge_state),'instances'] += 1

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

    print("frame {}: {}".format(frame_id, instances_df))
    print("frame {}: total number of labelled instances: {}".format(frame_id, instances_df.instances.sum()))
    return instances_df


# ray.get([render_tile_for_frame.remote(frame_r) for frame_r in zip(ms1_frame_properties_df.iloc[:1].frame_id, ms1_frame_properties_df.iloc[:1].retention_time_secs)])
ray.get([render_tile_for_frame.remote(frame_r) for frame_r in zip(ms1_frame_properties_df.frame_id, ms1_frame_properties_df.retention_time_secs)])

print("shutting down ray")
ray.shutdown()
