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

# in YOLO a small object is smaller than 16x16 @ 416x416 image size.
SMALL_OBJECT_W = SMALL_OBJECT_H = 16/416

# python ./otf-peak-detect/yolo-feature-detection/training/create-training-set-from-tfd.py -eb ~/Downloads/experiments -en dwm-test -rn 190719_Hela_Ecoli_1to1_01 -tidx 34 -np 4

parser = argparse.ArgumentParser(description='Set up a training set from raw tiles.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-tn','--training_set_name', type=str, default='yolo', help='Name of the training set.', required=False)
parser.add_argument('-rtl','--rt_lower', type=int, default=200, help='Lower bound of the RT range.', required=False)
parser.add_argument('-rtu','--rt_upper', type=int, default=800, help='Upper bound of the RT range.', required=False)
parser.add_argument('-tidx','--tile_idx_list', nargs='+', type=int, help='Space-separated indexes of the tiles to use for the training set.', required=True)
parser.add_argument('-np','--number_of_processors', type=int, default=8, help='The number of processors to use.', required=False)
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
RUN_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

# check the converted database exists
CONVERTED_DATABASE_NAME = "{}/exp-{}-run-{}-converted.sqlite".format(RUN_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

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

# check the raw tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.run_name)
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

# check the consolidated features file exists - create it if it doesn't
FEATURES_NAME = '{}/features/{}/exp-{}-run-{}-features-all.pkl'.format(EXPERIMENT_DIR, args.run_name, args.experiment_name, args.run_name)
if not os.path.isfile(FEATURES_NAME):
    print("the consolidated features file does not exist - creating {}".format(FEATURES_NAME))
    # consolidate the features into a single file
    features_file_list = sorted(glob.glob('{}/features/{}/exp-{}-run-{}-features-precursor-*.pkl'.format(EXPERIMENT_DIR, args.run_name, args.experiment_name, args.run_name)))
    df_l = []
    for features_file in features_file_list:
        df = pd.read_pickle(features_file)
        df_l.append(df)
    features_df = pd.concat(df_l, axis=0, sort=False)
    features_df.to_pickle(FEATURES_NAME)
    print("created {}".format(FEATURES_NAME))

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


# get the frame properties so we can map frame ID to RT
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select Id,Time from frame_properties where Time >= {} and Time <= {} and MsMsType == {}".format(args.rt_lower, args.rt_upper, FRAME_TYPE_MS1), db_conn)
db_conn.close()

# load the features detected by TFD
features_df = pd.read_pickle(FEATURES_NAME)

# add the m/z extent of the features
features_df['mz_lower'] = features_df.apply(lambda row: row.envelope[0][0], axis=1)
features_df['mz_upper'] = features_df.apply(lambda row: row.envelope[-1][0], axis=1)

# add the RT extent of the features
features_df['rt_lower_frame'] = features_df.apply(lambda row: row.rt_lower if row.rt_curve_fit else row.rt_apex-2 , axis=1)
features_df['rt_upper_frame'] = features_df.apply(lambda row: row.rt_upper if row.rt_curve_fit else row.rt_apex+2 , axis=1)

# get all the tiles that have been generated from the raw data
tile_filename_list = sorted(glob.glob("{}/frame-*-tile-*.png".format(PRE_ASSIGNED_FILES_DIR)))

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
    intersecting_features_df = features_df[(features_df.rt_lower_frame <= frame_rt) & (features_df.rt_upper_frame >= frame_rt) & (features_df.mz_lower >= tile_mz_lower) & (features_df.mz_upper <= tile_mz_upper)]
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
    print(class {} objects: {}".format(c, classes_d[c]))
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
NUMBER_OF_CLASSES = 2

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
