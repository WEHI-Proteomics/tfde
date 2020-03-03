# This application creates everything YOLO needs in the training set. The output base directory should be copied to ~/darket/data/peptides on the training machine with scp -rp. Prior to this step, the raw tiles must be created with create-raw-data-tiles.py.
import json
from PIL import Image, ImageDraw
import os, shutil
import random
import argparse
import glob
import sqlite3
import pandas as pd
import sys
import pickle
import numpy as np
import ray

PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines
MZ_MIN = 100.0
MZ_MAX = 1700.0
SCAN_MAX = PIXELS_Y
SCAN_MIN = 1
MZ_PER_TILE = 18.0
TILES_PER_FRAME = int((MZ_MAX - MZ_MIN) / MZ_PER_TILE) + 1

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
CARBON_MASS_DIFFERENCE = 1.003355

# in YOLO a small object is smaller than 16x16 @ 416x416 image size.
SMALL_OBJECT_W = SMALL_OBJECT_H = 16/416

# takes a numpy array of intensity, and another of mz
def mz_centroid(_int_f, _mz_f):
    try:
        return ((_int_f/_int_f.sum()) * _mz_f).sum()
    except:
        print("exception in mz_centroid")
        return None

def estimate_target_coordinates(row_as_series, mz_estimator, scan_estimator, rt_estimator):
    sequence_estimation_attribs_s = row_as_series[['theoretical_mz','experiment_rt_mean','experiment_rt_std_dev','experiment_scan_mean','experiment_scan_std_dev','experiment_intensity_mean','experiment_intensity_std_dev']]
    sequence_estimation_attribs = np.reshape(sequence_estimation_attribs_s.values, (1, -1))  # make it 2D

    # estimate the raw monoisotopic m/z
    mz_delta_ppm_estimated = mz_estimator.predict(sequence_estimation_attribs)[0]
    theoretical_mz = sequence_estimation_attribs_s.theoretical_mz
    estimated_monoisotopic_mz = (mz_delta_ppm_estimated / 1e6 * theoretical_mz) + theoretical_mz

    # estimate the raw monoisotopic scan
    estimated_scan_delta = scan_estimator.predict(sequence_estimation_attribs)[0]
    experiment_scan_mean = sequence_estimation_attribs_s.experiment_scan_mean
    estimated_scan_apex = (estimated_scan_delta * experiment_scan_mean) + experiment_scan_mean

    # estimate the raw monoisotopic RT
    estimated_rt_delta = rt_estimator.predict(sequence_estimation_attribs)[0]
    experiment_rt_mean = sequence_estimation_attribs_s.experiment_rt_mean
    estimated_rt_apex = (estimated_rt_delta * experiment_rt_mean) + experiment_rt_mean

    return {"mono_mz":estimated_monoisotopic_mz, "scan_apex":estimated_scan_apex, "rt_apex":estimated_rt_apex}

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
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    points_df = pd.read_sql_query("select * from frames where frame_id == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {}".format(frame_id, feature.mz_lower, feature.mz_upper, feature.scan_lower, feature.scan_upper), db_conn)
    db_conn.close()
    return (len(points_df) > 0)

# determine the mapping between the percolator index and the run file name
def get_percolator_run_mapping(mapping_file_name):
    df = pd.read_csv(mapping_file_name)
    mapping_l = list(df.itertuples(index=False, name=None))
    return mapping_l

def file_idx_for_run(run_name):
    result = None
    mapping_l = get_percolator_run_mapping(MAPPING_FILE_NAME)
    for m in mapping_l:
        if m[1] == run_name:
            result = m[0]
            break
    return result

# Find the feature extents for labelling
@ray.remote
def calculate_feature_extents(row):
    experiment_scan_peak_width = row.experiment_scan_peak_width
    experiment_rt_peak_width = row.experiment_rt_peak_width
    charge = row.charge
    sequence = row.sequence

    print("processing {}, charge {}".format(sequence, charge))

    coordinates_d = row.target_coords
    mono_mz = coordinates_d['mono_mz']
    scan_apex = coordinates_d['scan_apex']
    rt_apex = coordinates_d['rt_apex']

    # distance for looking either side of the scan and RT apex, based on the other times this sequence has been seen in this experiment
    SCAN_WIDTH = experiment_scan_peak_width
    RT_WIDTH = experiment_rt_peak_width

    # the width to use for isotopic width, in Da
    MZ_TOLERANCE_PPM = 5  # +/- this amount
    MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4
    MS1_PEAK_DELTA = mono_mz * MZ_TOLERANCE_PERCENT / 100

    # the number of isotopes to look for in the m/z dimension - the theoretical model includes 7 (the monoisotopic plus 6 isotopes)
    NUMBER_OF_ISOTOPES = 7
    expected_spacing_mz = CARBON_MASS_DIFFERENCE / charge

    # define the region we will look in for the feature
    feature_region_mz_lower = mono_mz - MS1_PEAK_DELTA
    feature_region_mz_upper = mono_mz + (NUMBER_OF_ISOTOPES * expected_spacing_mz) + MS1_PEAK_DELTA
    scan_lower = scan_apex - (2 * SCAN_WIDTH)
    scan_upper = scan_apex + (2 * SCAN_WIDTH)
    rt_lower = rt_apex - (2 * RT_WIDTH)
    rt_upper = rt_apex + (2 * RT_WIDTH)

    # extract the raw data within this area of interest
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    feature_region_raw_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {} and frame_type == {}".format(feature_region_mz_lower, feature_region_mz_upper, scan_lower, scan_upper, rt_lower, rt_upper, FRAME_TYPE_MS1), db_conn)
    db_conn.close()

    # derive peaks for the monoisotopic and the isotopes
    isotope_peaks_l = []
    isotope_raw_points_l = []
    for isotope_idx in range(NUMBER_OF_ISOTOPES):
        estimated_isotope_midpoint = mono_mz + (isotope_idx * expected_spacing_mz)
        isotope_mz_lower = estimated_isotope_midpoint - MS1_PEAK_DELTA
        isotope_mz_upper = estimated_isotope_midpoint + MS1_PEAK_DELTA
        isotope_raw_points_df = feature_region_raw_points_df[(feature_region_raw_points_df.mz >= isotope_mz_lower) & (feature_region_raw_points_df.mz <= isotope_mz_upper)].copy()
        # add the isotope's raw points to the list
        isotope_raw_points_l.append(isotope_raw_points_df)
        if len(isotope_raw_points_df) > 0:
            # centroid the raw points to get the peak for the isotope
            isotope_raw_points_a = isotope_raw_points_df[['mz','intensity']].values
            mz_cent = mz_centroid(isotope_raw_points_a[:,1], isotope_raw_points_a[:,0])
            summed_intensity = isotope_raw_points_a[:,1].sum()
        else:
            mz_cent = None
            summed_intensity = 0
        # add the peak to the list of isotopic peaks
        isotope_peaks_l.append((mz_cent, summed_intensity))
    isotope_peaks_df = pd.DataFrame(isotope_peaks_l, columns=['mz_centroid','summed_intensity'])

    # determine the number of isotopes by finding the the first one past isotope 1 that is more intense than the one before it
    a = isotope_peaks_df.summed_intensity.values
    d = np.diff(a[1:])
    idxs = np.where(d > 0)[0]
    index_of_last_isotope = len(isotope_peaks_df) - 1
    if len(idxs) > 0:
        index_of_last_isotope = np.where(d > 0)[0][0] + 1

    mz_lower = mono_mz
    mz_upper = isotope_peaks_df.iloc[index_of_last_isotope].mz_centroid

    return (sequence, charge, mz_lower, mz_upper, scan_lower, scan_upper, rt_lower, rt_upper, index_of_last_isotope+1)


# python ./otf-peak-detect/yolo-feature-detection/training/create-training-set-from-tfd.py -eb ~/Downloads/experiments -en dwm-test -rn 190719_Hela_Ecoli_1to1_01 -tidx 34 -np 4

parser = argparse.ArgumentParser(description='Set up a training set from raw tiles.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)
parser.add_argument('-tn','--training_set_name', type=str, default='yolo', help='Name of the training set.', required=False)
parser.add_argument('-rtl','--rt_lower', type=int, default=200, help='Lower bound of the RT range.', required=False)
parser.add_argument('-rtu','--rt_upper', type=int, default=800, help='Upper bound of the RT range.', required=False)
parser.add_argument('-tidx','--tile_idx_list', nargs='+', type=int, help='Space-separated indexes of the tiles to use for the training set.', required=True)
parser.add_argument('-np','--number_of_processors', type=int, default=8, help='The number of processors to use.', required=False)
parser.add_argument('-cnf','--create_new_features', action='store_true', help='Create a new features file for labelling, even if one already exists.')
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.')
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
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

# check the run directory exists
CONVERTED_DATABASE_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(CONVERTED_DATABASE_DIR):
    print("The run directory is required but doesn't exist: {}".format(CONVERTED_DATABASE_DIR))
    sys.exit(1)

# check the converted database exists
CONVERTED_DATABASE_NAME = "{}/exp-{}-run-{}-converted.sqlite".format(CONVERTED_DATABASE_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

# set up the coordinate estimators directory
COORDINATE_ESTIMATORS_DIR = "{}/coordinate-estimators".format(EXPERIMENT_DIR)
if not os.path.exists(COORDINATE_ESTIMATORS_DIR):
    print("The coordinate estimators directory is required but doesn't exist: {}".format(COORDINATE_ESTIMATORS_DIR))
    sys.exit(1)

# load the sequence library
SEQUENCE_LIBRARY_DIR = "{}/sequence-library".format(EXPERIMENT_DIR)
SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.pkl".format(SEQUENCE_LIBRARY_DIR)
if not os.path.isfile(SEQUENCE_LIBRARY_FILE_NAME):
    print("The sequences library file doesn't exist: {}".format(SEQUENCE_LIBRARY_FILE_NAME))
    sys.exit(1)
else:
    library_sequences_df = pd.read_pickle(SEQUENCE_LIBRARY_FILE_NAME)
    if args.small_set_mode:
        library_sequences_df = library_sequences_df.sample(n=20)
    print("loaded {} sequences from the library".format(len(library_sequences_df)))

# load the coordinate estimators
MZ_ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, args.run_name, 'mz')
if not os.path.isfile(MZ_ESTIMATOR_MODEL_FILE_NAME):
    print("The estimator file doesn't exist: {}".format(MZ_ESTIMATOR_MODEL_FILE_NAME))
    sys.exit(1)
else:
    with open(MZ_ESTIMATOR_MODEL_FILE_NAME, 'rb') as file:
        mz_estimator = pickle.load(file)

SCAN_ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, args.run_name, 'scan')
if not os.path.isfile(SCAN_ESTIMATOR_MODEL_FILE_NAME):
    print("The estimator file doesn't exist: {}".format(SCAN_ESTIMATOR_MODEL_FILE_NAME))
    sys.exit(1)
else:
    with open(SCAN_ESTIMATOR_MODEL_FILE_NAME, 'rb') as file:
        scan_estimator = pickle.load(file)

RT_ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, args.run_name, 'rt')
if not os.path.isfile(RT_ESTIMATOR_MODEL_FILE_NAME):
    print("The estimator file doesn't exist: {}".format(RT_ESTIMATOR_MODEL_FILE_NAME))
    sys.exit(1)
else:
    with open(RT_ESTIMATOR_MODEL_FILE_NAME, 'rb') as file:
        rt_estimator = pickle.load(file)

MAPPING_FILE_NAME = "{}/recalibrated-percolator-output/percolator-idx-mapping.csv".format(EXPERIMENT_DIR)
if not os.path.isfile(MAPPING_FILE_NAME):
    print("The mapping file doesn't exist: {}".format(MAPPING_FILE_NAME))
    sys.exit(1)
else:
    file_idx = file_idx_for_run(args.run_name)

# set up the training base directories
TRAINING_SET_BASE_DIR = '{}/training-sets/{}'.format(EXPERIMENT_DIR, args.training_set_name)
PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TRAINING_SET_BASE_DIR)
OVERLAY_FILES_DIR = '{}/overlays'.format(TRAINING_SET_BASE_DIR)

if os.path.exists(TRAINING_SET_BASE_DIR):
    shutil.rmtree(TRAINING_SET_BASE_DIR)
os.makedirs(TRAINING_SET_BASE_DIR)

if os.path.exists(PRE_ASSIGNED_FILES_DIR):
    shutil.rmtree(PRE_ASSIGNED_FILES_DIR)
os.makedirs(PRE_ASSIGNED_FILES_DIR)

if os.path.exists(OVERLAY_FILES_DIR):
    shutil.rmtree(OVERLAY_FILES_DIR)
os.makedirs(OVERLAY_FILES_DIR)

# check if the features file exists - if not, create it
FEATURES_FILE_NAME = '{}/features-for-labelling.pkl'.format(TRAINING_SET_BASE_DIR)
if (not os.path.isfile(FEATURES_FILE_NAME)) or args.create_new_features:
    # calculate the target coordinates
    print("calculating the target coordinates for each sequence-charge")
    library_sequences_df['target_coords'] = library_sequences_df.apply(lambda row: estimate_target_coordinates(row, mz_estimator, scan_estimator, rt_estimator), axis=1)

    if not ray.is_initialized():
        ray.init(num_cpus=args.number_of_processors)

    # for each library sequence, estimate where in this run it will be
    sequence_features_l = ray.get([calculate_feature_extents.remote(row) for row in library_sequences_df.itertuples()])

    # save the features file
    features_df = pd.DataFrame(sequence_features_l, columns=['sequence','charge','mz_lower','mz_upper','scan_lower','scan_upper','rt_lower','rt_upper','number_of_isotopes'])
    features_df.to_pickle(FEATURES_FILE_NAME)
else:
    features_df = pd.read_pickle(FEATURES_FILE_NAME)
    print("loaded {} features from {}".format(len(features_df), FEATURES_FILE_NAME))

# check the raw tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}/{}'.format(EXPERIMENT_DIR, args.run_name, args.tile_set_name)
if not os.path.exists(TILES_BASE_DIR):
    print("The raw tiles base directory is required but does not exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

# check the tiles directory exists for each tile index we need
for tile_idx in args.tile_idx_list:
    tile_dir = "{}/tile-{}".format(TILES_BASE_DIR, tile_idx)
    if os.path.exists(tile_dir):
        # copy the raw tiles to the pre-assigned tiles directory
        file_list = sorted(glob.glob("{}/frame-*-tile-*.png".format(tile_dir)))
        print("copying {} tiles from {} to {}".format(len(file_list), tile_dir, PRE_ASSIGNED_FILES_DIR))
        for file in file_list:
            base_name = os.path.basename(file)
            destination_name = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, base_name)
            shutil.copyfile(file, destination_name)
    else:
        print("The tiles directory is required but does not exist: {}".format(tile_dir))
        sys.exit(1)

# get the frame properties so we can map frame ID to RT
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select Id,Time from frame_properties where Time >= {} and Time <= {} and MsMsType == {}".format(args.rt_lower, args.rt_upper, FRAME_TYPE_MS1), db_conn)
db_conn.close()

# get all the tiles that have been generated from the raw data
tile_filename_list = sorted(glob.glob("{}/frame-*-tile-*.png".format(PRE_ASSIGNED_FILES_DIR)))

if args.small_set_mode:
    tile_filename_list = tile_filename_list[:20]

classes_d = {}
small_objects = 0
total_objects = 0
tile_list = []

# for each raw tile, create its overlay and label text file
for idx,tile_filename in enumerate(tile_filename_list):
    if idx % 100 == 0:
        print("processing {} of {} tiles".format(idx+1, len(tile_filename_list)))

    base_name = os.path.basename(tile_filename)
    frame_id = int(base_name.split('-')[1])
    tile_id = int(base_name.split('-')[3].split('.')[0])

    # get the m/z range for this tile
    (tile_mz_lower,tile_mz_upper) = mz_range_for_tile(tile_id)

    annotations_filename = '{}.txt'.format(os.path.splitext(base_name)[0])
    annotations_path = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, annotations_filename)
    tile_list.append((base_name, annotations_filename))

    # get the retention time for this frame
    frame_rt = ms1_frame_properties_df[ms1_frame_properties_df.Id == frame_id].iloc[0].Time
    # find the features intersecting with this frame
    intersecting_features_df = features_df[(features_df.rt_lower <= frame_rt) & (features_df.rt_upper >= frame_rt) & (features_df.mz_lower >= tile_mz_lower) & (features_df.mz_upper <= tile_mz_upper)]
    # remember the coordinates so we can write them to the annotations file
    feature_coordinates = []
    # draw the labels on the raw tile
    img = Image.open(tile_filename)
    draw = ImageDraw.Draw(img)
    for idx,feature in intersecting_features_df.iterrows():
        (t,x0_buffer) = tile_pixel_x_from_mz(feature.mz_lower - 0.25)
        if t < tile_id:
            x0_buffer = 1
        (t,x1_buffer) = tile_pixel_x_from_mz(feature.mz_upper + 0.25)
        if t > tile_id:
            x1_buffer = PIXELS_X
        y0 = feature.scan_lower
        y1 = feature.scan_upper
        w = x1_buffer - x0_buffer
        h = y1 - y0
        charge = feature.charge
        # calculate the annotation coordinates for the text file
        yolo_x = (x0_buffer + (w / 2)) / PIXELS_X
        yolo_y = (y0 + (h / 2)) / PIXELS_Y
        yolo_w = w / PIXELS_X
        yolo_h = h / PIXELS_Y
        # label this object if it meets the criteria
        if label_this_object(frame_id, feature):
            # we are only interested in charge 2 and higher
            if charge >= 2:
                feature_class = charge - 2
                if feature_class in classes_d.keys():
                    classes_d[feature_class] += 1
                else:
                    classes_d[feature_class] = 1
                # add it to the list
                feature_coordinates.append(("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(feature_class, yolo_x, yolo_y, yolo_w, yolo_h)))
                # draw the rectangle
                draw.rectangle(xy=[(x0_buffer, y0), (x1_buffer, y1)], fill=None, outline='red')
                # keep record of the 'small' objects
                total_objects += 1
                if (w <= SMALL_OBJECT_W) and (h <= SMALL_OBJECT_H):
                    small_objects += 1

    # write the overlay tile
    img.save('{}/{}'.format(OVERLAY_FILES_DIR, base_name))

    # write the annotations text file
    with open(annotations_path, 'w') as f:
        for item in feature_coordinates:
            f.write("%s\n" % item)

for c in sorted(classes_d.keys()):
    print("charge {} objects: {}".format(c+2, classes_d[c]))
print("{} out of {} objects are small.".format(small_objects, total_objects))

# assign the tiles to the training sets

train_proportion = 0.8
val_proportion = 0.15
train_n = round(len(tile_list) * train_proportion)
val_n = round(len(tile_list) * val_proportion)

train_set = random.sample(tile_list, train_n)
val_test_set = list(set(tile_list) - set(train_set))
val_set = random.sample(val_test_set, val_n)
test_set = list(set(val_test_set) - set(val_set))

print("tile set counts - train {}, validation {}, test {}".format(len(train_set), len(val_set), len(test_set)))

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

for file_pair in train_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(TRAIN_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(TRAIN_SET_DIR, file_pair[1]))

for file_pair in val_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(VAL_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(VAL_SET_DIR, file_pair[1]))

for file_pair in test_set:
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[0]), '{}/{}'.format(TEST_SET_DIR, file_pair[0]))
    shutil.copyfile('{}/{}'.format(PRE_ASSIGNED_FILES_DIR, file_pair[1]), '{}/{}'.format(TEST_SET_DIR, file_pair[1]))

# create obj.names, for copying to ./darknet/data, with the object names, each one on a new line
LOCAL_NAMES_FILENAME = "{}/peptides-obj.names".format(TRAINING_SET_BASE_DIR)
NUMBER_OF_CLASSES = len(classes_d.keys())

with open(LOCAL_NAMES_FILENAME, 'w') as f:
    for object_class in range(NUMBER_OF_CLASSES):
        f.write("charge-{}\n".format(object_class + 2))

# create obj.data, for copying to ./darknet/data
LOCAL_DATA_FILENAME = "{}/peptides-obj.data".format(TRAINING_SET_BASE_DIR)

with open(LOCAL_DATA_FILENAME, 'w') as f:
    f.write("classes={}\n".format(NUMBER_OF_CLASSES))
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

# take a copy of the training set because we'll be augmenting it later
backup_training_set_dir = "{}-backup".format(TRAIN_SET_DIR)
if os.path.exists(backup_training_set_dir):
    shutil.rmtree(backup_training_set_dir)
shutil.copytree(TRAIN_SET_DIR, backup_training_set_dir)
