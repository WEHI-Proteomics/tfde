# This application creates everything YOLO needs in the training set. The output base directory should be copied to ~/darket/data/peptides on the training machine with scp -rp. Prior to this step, the raw tiles must be created with create-raw-data-tiles.py and either put on the spectra server or on the local filesystem.
import json
from urllib import parse
from urllib.request import urlopen
from PIL import Image, ImageDraw
import os, shutil
import random

PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines

BASE_DIR = '/data/yolo-training-sets'
TRAINING_SET_BASE_DIR = '{}/sets'.format(BASE_DIR)
PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TRAINING_SET_BASE_DIR)
OVERLAY_FILES_DIR = '{}/overlays'.format(TRAINING_SET_BASE_DIR)

LOCAL_TILE_SOURCE_DIR = '/data/experiments/dwm-test/tiles/190719_Hela_Ecoli_1to1_01/tile-33/'

if os.path.exists(PRE_ASSIGNED_FILES_DIR):
    shutil.rmtree(PRE_ASSIGNED_FILES_DIR)
os.makedirs(PRE_ASSIGNED_FILES_DIR)

if os.path.exists(OVERLAY_FILES_DIR):
    shutil.rmtree(OVERLAY_FILES_DIR)
os.makedirs(OVERLAY_FILES_DIR)

annotations_file_name = '{}/annotated-tiles/via_export_json_aw.json'.format(BASE_DIR)
with open(annotations_file_name) as annotations_file:
    annotations = json.load(annotations_file)

url = list(annotations.items())[0][1]['filename']

# in YOLO a small object is smaller than 16x16 @ 416x416 image size.
SMALL_OBJECT_W = SMALL_OBJECT_H = 16/416

# TILE_SOURCE = 'server'
TILE_SOURCE = 'local'

tile_list = []
classes_d = {}
small_objects = 0
total_objects = 0
digits = '0123456789'
for tile in list(annotations.items()):
    tile_d = tile[1]
    tile_regions = tile_d['regions']
    # process this tile if there are annotations for it
    if len(tile_regions) > 0:
        # load the tile
        tile_url = tile_d['filename']  # this is the URL so we need to download it

        # determine the frame_id and tile_id
        path_split = parse.urlsplit(tile_url).path.split('/')
        tile_idx = int(path_split[2])
        frame_id = int(path_split[4])

        if TILE_SOURCE == 'server':
            tile = Image.open(urlopen(tile_url))
        elif TILE_SOURCE == 'local':
            file_name = '{}/frame-{}-tile-33-mz-694-712.png'.format(LOCAL_TILE_SOURCE_DIR, frame_id)
            tile = Image.open(file_name)

        # set the file names
        tile_filename = 'frame-{}-tile-{}.png'.format(frame_id, tile_idx)
        tile_path = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, tile_filename)
        annotations_filename = 'frame-{}-tile-{}.txt'.format(frame_id, tile_idx)
        annotations_path = '{}/{}'.format(PRE_ASSIGNED_FILES_DIR, annotations_filename)
        overlay_filename = '{}/frame-{}-tile-{}.png'.format(OVERLAY_FILES_DIR, frame_id, tile_idx)
        tile_list.append((tile_filename, annotations_filename))

        # save this tile
        tile.save(tile_path)

        # get a drawing context for the tile
        draw = ImageDraw.Draw(tile)

        # render the annotations
        feature_coordinates = []
        total_objects += len(tile_regions)
        for region in tile_regions:
            shape_attributes = region['shape_attributes']
            x = shape_attributes['x']
            y = shape_attributes['y']
            w = shape_attributes['width']
            h = shape_attributes['height']
            # calculate the annotation coordinates for the text file
            yolo_x = (x + (w / 2)) / PIXELS_X
            yolo_y = (y + (h / 2)) / PIXELS_Y
            yolo_w = w / PIXELS_X
            yolo_h = h / PIXELS_Y
            # keep record of the small objects
            if (yolo_w <= SMALL_OBJECT_W) and (yolo_h <= SMALL_OBJECT_H):
                small_objects += 1
            # determine the class of this annotation
            region_attributes = region['region_attributes']
            charge = int(''.join(c for c in region_attributes['charge'] if c in digits))
            # we are only interested in charge 2 and higher
            if charge >= 2:
                feature_class = charge - 2
                if feature_class in classes_d.keys():
                    classes_d[feature_class] += 1
                else:
                    classes_d[feature_class] = 1
                # add it to the list
                feature_coordinates.append(("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(feature_class, yolo_x, yolo_y, yolo_w, yolo_h)))
                # draw the overlay
                draw.rectangle(xy=[(x, y), (x+w, y+h)], fill=None, outline='red')

        # save the overlay tile
        tile.save(overlay_filename)

        # write the annotations text file
        with open(annotations_path, 'w') as f:
            for item in feature_coordinates:
                f.write("%s\n" % item)
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

print("train {}, validation {}, test {}".format(len(train_set), len(val_set), len(test_set)))

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
