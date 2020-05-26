# This application creates everything YOLO needs in the training set. The output base directory should be copied to ~/darket/data/peptides on the training machine with scp -rp. Prior to this step, the raw tiles must be created with create-raw-data-tiles.py.
import json
from PIL import Image, ImageDraw, ImageChops
import os, shutil
import random
import argparse
import glob
import sqlite3
import pandas as pd
import sys
import pickle
import numpy as np
import logging
import time
import multiprocessing as mp
import ray


PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines
MZ_MIN = 100.0
MZ_MAX = 1700.0
SCAN_MAX = PIXELS_Y
SCAN_MIN = 1
MZ_PER_TILE = 18.0
TILES_PER_FRAME = int((MZ_MAX - MZ_MIN) / MZ_PER_TILE) + 1
MIN_TILE_IDX = 0
MAX_TILE_IDX = TILES_PER_FRAME-1

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# charge states of interest
MIN_CHARGE = 2
MAX_CHARGE = 4

# in YOLO a small object is smaller than 16x16 @ 416x416 image size.
SMALL_OBJECT_W = SMALL_OBJECT_H = 16/416

# allow for some buffer area around the features
MZ_BUFFER = 0.25
SCAN_BUFFER = 5

# get the m/z extent for the specified tile ID
def mz_range_for_tile(tile_id):
    assert (tile_id >= 0) and (tile_id <= TILES_PER_FRAME-1), "tile_id not in range"

    mz_lower = MZ_MIN + (tile_id * MZ_PER_TILE)
    mz_upper = mz_lower + MZ_PER_TILE
    return (mz_lower, mz_upper)

# get the tile ID and the x pixel coordinate for the specified m/z
def tile_pixel_x_from_mz(mz):
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"

    tile_id = int((mz - MZ_MIN) / MZ_PER_TILE)
    pixel_x = int(((mz - MZ_MIN) % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return (tile_id, pixel_x)

# decide whether there is sufficient data in this region to justify labelling it for the training set
def label_this_object(frame_id, feature):
    # are there any points in this region of the frame?
    db_conn = sqlite3.connect(CONVERTED_DATAbasename)
    points_df = pd.read_sql_query("select * from frames where frame_id == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {}".format(frame_id, feature.mz_lower, feature.mz_upper, feature.scan_lower, feature.scan_upper), db_conn)
    db_conn.close()
    return (len(points_df) > 0)

# determine the mapping between the percolator index and the run file name
def get_percolator_run_mapping(mapping_file_name):
    df = pd.read_csv(mapping_file_name)
    mapping_l = [tuple(r) for r in df.to_numpy()]
    return mapping_l

def file_idx_for_run(run_name):
    result = None
    mapping_l = get_percolator_run_mapping(MAPPING_FILE_NAME)
    for m in mapping_l:
        if m[1] == run_name:
            result = m[0]
            break
    return result

def run_name_for_file_idx(file_idx):
    result = None
    mapping_l = get_percolator_run_mapping(MAPPING_FILE_NAME)
    for m in mapping_l:
        if m[0] == file_idx:
            result = m[1]
            break
    return result

def scan_coords_for_single_charge_region(mz_lower, mz_upper):
    scan_for_mz_lower = -1 * ((1.2 * mz_lower) - 1252)
    scan_for_mz_upper = -1 * ((1.2 * mz_upper) - 1252)
    return (scan_for_mz_lower,scan_for_mz_upper)

@ray.remote
def apply_feature_mask(file_pair):
    global train_set_object_count

    # copy the tiles to their training set directory
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(TRAIN_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(TRAIN_SET_DIR, file_pair[1]))

    # find this tile in the tile metadata
    basename = file_pair[0]
    found = False
    for tile in tile_metadata_l:
        if tile['basename'] == basename:
            mask_region_y_left = tile['mask_region_y_left']
            mask_region_y_right = tile['mask_region_y_right']
            tile_features_l = tile['tile_features_l']
            found = True
            break
    assert(found == True), "could not find the metadata for tile {}".format(basename)

    # create a feature mask
    mask_im_array = np.zeros([PIXELS_Y+1, PIXELS_X+1, 3], dtype=np.uint8)
    mask = Image.fromarray(mask_im_array.astype('uint8'), 'RGB')
    mask_draw = ImageDraw.Draw(mask)

    # draw a mask for each features on this tile
    for feature in tile_features_l:
        # draw the mask for this feature
        x0_buffer = feature['x0_buffer']
        y0_buffer = feature['y0_buffer']
        x1_buffer = feature['x1_buffer']
        y1_buffer = feature['y1_buffer']
        mask_draw.rectangle(xy=[(x0_buffer, y0_buffer), (x1_buffer, y1_buffer)], fill='white', outline='white')

    # save the bare mask
    mask.save('{}/{}'.format(MASK_FILES_DIR, basename))

    # apply the mask to the tile
    img = Image.open("{}/{}".format(TRAIN_SET_DIR, basename))
    masked_tile = ImageChops.multiply(img, mask)
    masked_tile.save("{}/{}".format(TRAIN_SET_DIR, basename))

    # count how many objects there are in this set
    train_set_object_count += len(tile_features_l)

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers


# python ./otf-peak-detect/yolo-feature-detection/training/create-training-set-from-tfd.py -eb ~/Downloads/experiments -en dwm-test -rn 190719_Hela_Ecoli_1to1_01 -tidx 34

parser = argparse.ArgumentParser(description='Set up a training set from raw tiles.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)
parser.add_argument('-tn','--training_set_name', type=str, default='yolo', help='Name of the training set.', required=False)
parser.add_argument('-rtl','--rt_lower', type=int, default=200, help='Lower bound of the RT range.', required=False)
parser.add_argument('-rtu','--rt_upper', type=int, default=800, help='Upper bound of the RT range.', required=False)
parser.add_argument('-tidx','--tile_idx_list', type=str, help='Indexes of the tiles to use for the training set. Can specify several ranges (e.g. 10-20,21-30,31-40), a single range (e.g. 10-24), individual indexes (e.g. 34,56,32), or a single index (e.g. 54). Indexes must be between {} and {}'.format(MIN_TILE_IDX,MAX_TILE_IDX), required=True)
parser.add_argument('-inf','--inference_mode', action='store_true', help='This set of labelled tiles is for testing a model\'s inference rather than for training a new model.')
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.')
parser.add_argument('-ssms','--small_set_mode_size', type=int, default='100', help='The number of tiles to sample for small set mode.', required=False)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], default='cluster', help='The Ray mode to use.', required=False)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.6, help='Proportion of the machine\'s cores to use for this program.', required=False)
args = parser.parse_args()

# store the command line arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

start_run = time.time()

# parse the tile indexes
indexes_l = []
for item in args.tile_idx_list.replace(" ", "").split(','):
    index_range = item.split('-')
    if all([i.isnumeric() for i in index_range]):  # only use the range if it's valid
        index_range = [int(i) for i in index_range]
        if len(index_range) == 2:
            index_lower = min(index_range)
            index_upper = max(index_range)
            indexes_l.append([i for i in range(index_lower, index_upper+1)])
            info.append("tile range {}-{}, with m/z range {} to {}".format(index_lower, index_upper, round(mz_range_for_tile(index_lower)[0],1), round(mz_range_for_tile(index_upper)[1],1)))
        else:
            indexes_l.append(index_range)
            info.append("tile index {}, with m/z range {} to {}".format(index_range[0], round(mz_range_for_tile(index_range[0])[0],1), round(mz_range_for_tile(index_range[0])[1],1)))
indexes_l = [item for sublist in indexes_l for item in sublist]
if len(indexes_l) == 0:
    print("Need to specify at least one tile index to include training set: {}".format(args.tile_idx_list))
    sys.exit(1)

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the run directory exists
CONVERTED_DATABASE_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(CONVERTED_DATABASE_DIR):
    print("The run directory is required but doesn't exist: {}".format(CONVERTED_DATABASE_DIR))
    sys.exit(1)

# check the converted database exists
CONVERTED_DATAbasename = "{}/exp-{}-run-{}-converted.sqlite".format(CONVERTED_DATABASE_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(CONVERTED_DATAbasename):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATAbasename))
    sys.exit(1)

# check the extracted features directory
EXTRACTED_FEATURES_DIR = "{}/extracted-features".format(EXPERIMENT_DIR)
if not os.path.exists(EXTRACTED_FEATURES_DIR):
    print("The extracted features directory is required but doesn't exist: {}".format(EXTRACTED_FEATURES_DIR))
    sys.exit(1)

# check the extracted features database
EXTRACTED_FEATURES_DB_NAME = "{}/extracted-features.sqlite".format(EXTRACTED_FEATURES_DIR)
if not os.path.isfile(EXTRACTED_FEATURES_DB_NAME):
    print("The extracted features database is required but doesn't exist: {}".format(EXTRACTED_FEATURES_DB_NAME))
    sys.exit(1)

MAPPING_FILE_NAME = "{}/recalibrated-percolator-output/percolator-idx-mapping.csv".format(EXPERIMENT_DIR)

# set up the training base directories
TRAINING_SET_BASE_DIR = '{}/training-sets/{}'.format(EXPERIMENT_DIR, args.training_set_name)
PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TRAINING_SET_BASE_DIR)
OVERLAY_FILES_DIR = '{}/overlays'.format(TRAINING_SET_BASE_DIR)
MASK_FILES_DIR = '{}/masks'.format(TRAINING_SET_BASE_DIR)

# make sure they're empty for a fresh start
if os.path.exists(TRAINING_SET_BASE_DIR):
    shutil.rmtree(TRAINING_SET_BASE_DIR)
os.makedirs(TRAINING_SET_BASE_DIR)

if os.path.exists(PRE_ASSIGNED_FILES_DIR):
    shutil.rmtree(PRE_ASSIGNED_FILES_DIR)
os.makedirs(PRE_ASSIGNED_FILES_DIR)

if os.path.exists(OVERLAY_FILES_DIR):
    shutil.rmtree(OVERLAY_FILES_DIR)
os.makedirs(OVERLAY_FILES_DIR)

if os.path.exists(MASK_FILES_DIR):
    shutil.rmtree(MASK_FILES_DIR)
os.makedirs(MASK_FILES_DIR)

# check the raw tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}/{}'.format(EXPERIMENT_DIR, args.run_name, args.tile_set_name)
if not os.path.exists(TILES_BASE_DIR):
    print("The raw tiles base directory is required but does not exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

# set up the training sets directories
SETS_BASE_DIR = '{}/sets'.format(TRAINING_SET_BASE_DIR)
TRAIN_SET_DIR = '{}/train'.format(SETS_BASE_DIR)
VAL_SET_DIR = '{}/validation'.format(SETS_BASE_DIR)
TEST_SET_DIR = '{}/test'.format(SETS_BASE_DIR)

if os.path.exists(TRAIN_SET_DIR):
    shutil.rmtree(TRAIN_SET_DIR)
os.makedirs(TRAIN_SET_DIR)

if os.path.exists(VAL_SET_DIR):
    shutil.rmtree(VAL_SET_DIR)
os.makedirs(VAL_SET_DIR)

if os.path.exists(TEST_SET_DIR):
    shutil.rmtree(TEST_SET_DIR)
os.makedirs(TEST_SET_DIR)

# create a place for the predictions for this inference set
if args.inference_mode:
    PREDICTIONS_BASE_DIR = '{}/predictions/{}'.format(EXPERIMENT_DIR, args.training_set_name)
    if os.path.exists(PREDICTIONS_BASE_DIR):
        shutil.rmtree(PREDICTIONS_BASE_DIR)
    os.makedirs(PREDICTIONS_BASE_DIR)

# set up logging
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
# set the file handler
file_handler = logging.FileHandler('{}/{}.log'.format(TRAINING_SET_BASE_DIR, parser.prog))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# set the console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("{} info: {}".format(parser.prog, info))

# determine tile allocation proportions - not using the test set at the moment because we'll create separate inference sets
if args.inference_mode:
    train_proportion = 0.0
    val_proportion = 1.0  # darknet mAP calculation uses the validation set
    test_proportion = 0.0
else:
    train_proportion = 0.8
    val_proportion = 0.2
    test_proportion = 0.0
logger.info("set proportions: train {}, validation {}, test {}".format(train_proportion, val_proportion, test_proportion))

print("setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(object_store_memory=20000000000,
                    redis_max_memory=25000000000,
                    num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

# load the extracted features for the specified run
logger.info("reading the extracted features from {}".format(EXTRACTED_FEATURES_DB_NAME))
db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
sequences_df = pd.read_sql_query('select sequence,charge,run_name,monoisotopic_mz_centroid,number_of_isotopes,mono_rt_bounds,mono_scan_bounds,isotope_1_rt_bounds,isotope_1_scan_bounds,isotope_2_rt_bounds,isotope_2_scan_bounds,isotope_intensities_l from features where file_idx=={}'.format(file_idx_for_run(args.run_name)), db_conn)
db_conn.close()
logger.info("loaded {} extracted features from {}".format(len(sequences_df), EXTRACTED_FEATURES_DB_NAME))

# unpack the feature extents
logger.info("unpacking the feature extents")
sequences_df.mono_rt_bounds = sequences_df.apply(lambda row: json.loads(row.mono_rt_bounds), axis=1)
sequences_df.mono_scan_bounds = sequences_df.apply(lambda row: json.loads(row.mono_scan_bounds), axis=1)

sequences_df.isotope_1_rt_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_1_rt_bounds), axis=1)
sequences_df.isotope_1_scan_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_1_scan_bounds), axis=1)

sequences_df.isotope_2_rt_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_2_rt_bounds), axis=1)
sequences_df.isotope_2_scan_bounds = sequences_df.apply(lambda row: json.loads(row.isotope_2_scan_bounds), axis=1)

sequences_df.isotope_intensities_l = sequences_df.apply(lambda row: json.loads(row.isotope_intensities_l), axis=1)

sequences_df['rt_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in [row.mono_rt_bounds,row.isotope_1_rt_bounds,row.isotope_2_rt_bounds]]), axis=1)
sequences_df['rt_upper'] = sequences_df.apply(lambda row: np.max([i[1] for i in [row.mono_rt_bounds,row.isotope_1_rt_bounds,row.isotope_2_rt_bounds]]), axis=1)

sequences_df['scan_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in [row.mono_scan_bounds,row.isotope_1_scan_bounds,row.isotope_2_scan_bounds]]), axis=1)
sequences_df['scan_upper'] = sequences_df.apply(lambda row: np.max([i[1] for i in [row.mono_scan_bounds,row.isotope_1_scan_bounds,row.isotope_2_scan_bounds]]), axis=1)

sequences_df['mz_lower'] = sequences_df.apply(lambda row: np.min([i[0] for i in row.isotope_intensities_l[0][4]]), axis=1)  # [0][4] refers to the isotope points of the monoisotope; i[0] refers to the m/z values
sequences_df['mz_upper'] = sequences_df.apply(lambda row: np.max([i[0] for i in row.isotope_intensities_l[row.number_of_isotopes-1][4]]), axis=1)

# get the frame properties so we can map frame ID to RT
logger.info("reading frame IDs from {}".format(CONVERTED_DATAbasename))
db_conn = sqlite3.connect(CONVERTED_DATAbasename)
ms1_frame_properties_df = pd.read_sql_query("select Id,Time from frame_properties where Time >= {} and Time <= {} and MsMsType == {}".format(args.rt_lower, args.rt_upper, FRAME_TYPE_MS1), db_conn)
min_frame_id = ms1_frame_properties_df.Id.min()
max_frame_id = ms1_frame_properties_df.Id.max()
db_conn.close()

# check the tiles directory exists for each tile index we need
for tile_idx in indexes_l:
    tile_count = 0
    tile_dir = "{}/tile-{}".format(TILES_BASE_DIR, tile_idx)
    if os.path.exists(tile_dir):
        # copy the raw tiles to the pre-assigned tiles directory
        file_list = sorted(glob.glob("{}/frame-*-tile-*.png".format(tile_dir)))
        logger.info("copying tiles within the RT range from {} to {}".format(tile_dir, PRE_ASSIGNED_FILES_DIR))
        for file in file_list:
            basename = os.path.basename(file)
            # if the frame_id is within the specified RT range
            frame_id = int(basename.split('-')[1])
            if (frame_id >= min_frame_id) and (frame_id <= max_frame_id):
                destination_name = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, basename)
                shutil.copyfile(file, destination_name)
                tile_count += 1
        logger.info("copied {} tiles for index {}".format(tile_count, tile_idx))
    else:
        print("The tiles directory is required but does not exist: {}".format(tile_dir))
        sys.exit(1)

# get all the tiles that have been generated from the raw data
tile_filename_list = sorted(glob.glob("{}/frame-*-tile-*.png".format(PRE_ASSIGNED_FILES_DIR)))
if len(tile_filename_list) == 0:
    print("Found no tiles to process in the pre-assigned directory: {}".format(PRE_ASSIGNED_FILES_DIR))
    sys.exit(1)

# limit the number of tiles for small set mode
if args.small_set_mode:
    tile_filename_list = tile_filename_list[:args.small_set_mode_size]

classes_d = {}
small_objects = 0
total_objects = 0
tile_list = []
objects_per_tile = []
tile_metadata_l = []

# for each raw tile, create its overlay and label text file
for idx,tile_filename in enumerate(tile_filename_list):
    if idx % 100 == 0:
        logger.info("processing {} of {} tiles".format(idx+1, len(tile_filename_list)))

    basename = os.path.basename(tile_filename)
    frame_id = int(basename.split('-')[1])
    tile_id = int(basename.split('-')[3].split('.')[0])

    number_of_objects_this_tile = 0

    # get the m/z range for this tile
    (tile_mz_lower,tile_mz_upper) = mz_range_for_tile(tile_id)
    # define the charge-1 region
    mask_region_y_left,mask_region_y_right = scan_coords_for_single_charge_region(tile_mz_lower, tile_mz_upper)

    # store metadata for this tile
    tile_metadata = {'frame_id':frame_id, 'tile_id':tile_id, 'basename':basename, 'mask_region_y_left':mask_region_y_left, 'mask_region_y_right':mask_region_y_right}

    annotations_filename = '{}.txt'.format(os.path.splitext(basename)[0])
    annotations_path = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, annotations_filename)
    tile_list.append((basename, annotations_filename))

    # get the retention time for this frame
    frame_rt = ms1_frame_properties_df[ms1_frame_properties_df.Id == frame_id].iloc[0].Time
    # find the features intersecting with this frame
    intersecting_features_df = sequences_df[(sequences_df.rt_lower <= frame_rt) & (sequences_df.rt_upper >= frame_rt) & (sequences_df.mz_lower >= tile_mz_lower) & (sequences_df.mz_upper <= tile_mz_upper)]
    # remember the coordinates so we can write them to the annotations file
    feature_coordinates = []
    # store the features for each tile so we can mask them later
    tile_features_l = []
    # draw the labels on the raw tile
    img = Image.open(tile_filename)
    draw = ImageDraw.Draw(img)
    for idx,feature in intersecting_features_df.iterrows():
        (t,x0_buffer) = tile_pixel_x_from_mz(feature.mz_lower - MZ_BUFFER)
        if t < tile_id:
            x0_buffer = 1
        (t,x1_buffer) = tile_pixel_x_from_mz(feature.mz_upper + MZ_BUFFER)
        if t > tile_id:
            x1_buffer = PIXELS_X
        y0 = feature.scan_lower
        y0_buffer = max((y0 - SCAN_BUFFER), SCAN_MIN)
        y1 = feature.scan_upper
        y1_buffer = min((y1 + SCAN_BUFFER), SCAN_MAX)
        w = x1_buffer - x0_buffer
        h = y1_buffer - y0_buffer
        charge = feature.charge
        # calculate the annotation coordinates for the text file
        yolo_x = (x0_buffer + (w / 2)) / PIXELS_X
        yolo_y = (y0_buffer + (h / 2)) / PIXELS_Y
        yolo_w = w / PIXELS_X
        yolo_h = h / PIXELS_Y
        # label this object if it meets the criteria
        if label_this_object(frame_id, feature):
            # we are only interested in charge 2 and higher
            if (charge >= MIN_CHARGE) and (charge <= MAX_CHARGE):
                feature_class = charge - MIN_CHARGE
                # keep record of how many instances of each class
                if feature_class in classes_d.keys():
                    classes_d[feature_class] += 1
                else:
                    classes_d[feature_class] = 1
                # add it to the list
                feature_coordinates.append(("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(feature_class, yolo_x, yolo_y, yolo_w, yolo_h)))
                # draw the rectangle on the overlay
                draw.rectangle(xy=[(x0_buffer, y0_buffer), (x1_buffer, y1_buffer)], fill=None, outline='red')
                # store the pixel coords for each feature for this tile so we can mask the features later
                tile_features_l.append({'x0_buffer':x0_buffer, 'y0_buffer':y0_buffer, 'x1_buffer':x1_buffer, 'y1_buffer':y1_buffer})
                # keep record of the 'small' objects
                total_objects += 1
                if (yolo_w <= SMALL_OBJECT_W) or (yolo_h <= SMALL_OBJECT_H):
                    small_objects += 1
                # keep track of the number of objects in this tile
                number_of_objects_this_tile += 1
            else:
                logger.info("found a charge-{} feature - not included in the training set".format(charge))

    # add it to the list
    tile_metadata['tile_features_l'] = tile_features_l
    tile_metadata_l.append(tile_metadata)

    # write the overlay tile
    img.save('{}/{}'.format(OVERLAY_FILES_DIR, basename))

    # write the annotations text file
    with open(annotations_path, 'w') as f:
        for item in feature_coordinates:
            f.write("%s\n" % item)
    
    objects_per_tile.append((tile_id, frame_id, number_of_objects_this_tile))

# display the object counts for each class
for c in sorted(classes_d.keys()):
    logger.info("charge {} objects: {}".format(c+2, classes_d[c]))
logger.info("{} out of {} objects ({}%) are small.".format(small_objects, total_objects, round(small_objects/total_objects*100,1)))

# display the number of objects per tile
objects_per_tile_df = pd.DataFrame(objects_per_tile, columns=['tile_id','frame_id','number_of_objects'])
objects_per_tile_df.to_pickle('{}/objects_per_tile_df.pkl'.format(TRAINING_SET_BASE_DIR))
logger.info("There are {} tiles with no objects.".format(len(objects_per_tile_df[objects_per_tile_df.number_of_objects == 0])))
logger.info("On average there are {} objects per tile.".format(round(np.mean(objects_per_tile_df.number_of_objects),1)))

# assign the tiles to the training sets
train_n = round(len(tile_list) * train_proportion)
val_n = round(len(tile_list) * val_proportion)

train_set = random.sample(tile_list, train_n)
val_test_set = list(set(tile_list) - set(train_set))
val_set = random.sample(val_test_set, val_n)
test_set = list(set(val_test_set) - set(val_set))

logger.info("tile counts - train {}, validation {}, test {}".format(len(train_set), len(val_set), len(test_set)))
number_of_classes = MAX_CHARGE - MIN_CHARGE + 1
max_batches = max(6000, max(2000*number_of_classes, len(train_set)))  # recommendation from AlexeyAB
logger.info("set max_batches={}, steps={},{},{},{}".format(max_batches, int(0.4*max_batches), int(0.6*max_batches), int(0.8*max_batches), int(0.9*max_batches)))

# copy the training set tiles and their annotation files to the training set directory
print("copying the training set to {}".format(TRAIN_SET_DIR))
train_set_object_count = 0
_ = ray.get([apply_feature_mask.remote(file_pair) for file_pair in train_set])

# copy the validation set tiles and their annotation files to the validation set directory
print("copying the validation set to {}".format(VAL_SET_DIR))
valid_set_object_count = 0
for file_pair in val_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(VAL_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(VAL_SET_DIR, file_pair[1]))

    # find this tile in the tile metadata
    basename = file_pair[0]
    found = False
    for tile in tile_metadata_l:
        if tile['basename'] == basename:
            mask_region_y_left = tile['mask_region_y_left']
            mask_region_y_right = tile['mask_region_y_right']
            tile_features_l = tile['tile_features_l']
            found = True
            break
    assert(found == True), "could not find the metadata for tile {}".format(basename)

    # count how many objects there are in this set
    valid_set_object_count += len(tile_features_l)

# copy the test set tiles and their annotation files to the test set directory
print("copying the test set to {}".format(TEST_SET_DIR))
test_set_object_count = 0
for file_pair in test_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(TEST_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(TEST_SET_DIR, file_pair[1]))

    # find this tile in the tile metadata
    basename = file_pair[0]
    found = False
    for tile in tile_metadata_l:
        if tile['basename'] == basename:
            mask_region_y_left = tile['mask_region_y_left']
            mask_region_y_right = tile['mask_region_y_right']
            tile_features_l = tile['tile_features_l']
            found = True
            break
    assert(found == True), "could not find the metadata for tile {}".format(basename)

    # count how many objects there are in this set
    test_set_object_count += len(tile_features_l)

logger.info("set object counts - train {}, validation {}, test {}".format(train_set_object_count, valid_set_object_count, test_set_object_count))

# create obj.names, for copying to ./darknet/data, with the object names, each one on a new line
LOCAL_NAMES_FILENAME = "{}/peptides-obj.names".format(TRAINING_SET_BASE_DIR)
print("writing {}".format(LOCAL_NAMES_FILENAME))

# class labels
with open(LOCAL_NAMES_FILENAME, 'w') as f:
    for charge in range(MIN_CHARGE, MAX_CHARGE+1):
        f.write("charge-{}\n".format(charge))

# create obj.data, for copying to ./darknet/data
LOCAL_DATA_FILENAME = "{}/peptides-obj.data".format(TRAINING_SET_BASE_DIR)
print("writing {}".format(LOCAL_DATA_FILENAME))

with open(LOCAL_DATA_FILENAME, 'w') as f:
    f.write("classes={}\n".format(MAX_CHARGE - MIN_CHARGE + 1))
    f.write("train=data/peptides/train.txt\n")
    f.write("valid=data/peptides/validation.txt\n")
    f.write("names=data/peptides/peptides-obj.names\n")
    f.write("backup=backup/\n")

# create the file list for each set
with open('{}/train.txt'.format(TRAINING_SET_BASE_DIR), 'w') as f:
    for file_pair in train_set:
        f.write('data/peptides/sets/train/{}\n'.format(file_pair[0]))

with open('{}/validation.txt'.format(TRAINING_SET_BASE_DIR), 'w') as f:
    for file_pair in val_set:
        f.write('data/peptides/sets/validation/{}\n'.format(file_pair[0]))

with open('{}/test.txt'.format(TRAINING_SET_BASE_DIR), 'w') as f:
    for file_pair in test_set:
        f.write('data/peptides/sets/test/{}\n'.format(file_pair[0]))

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
