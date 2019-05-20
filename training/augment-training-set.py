import glob
import os
import pandas as pd
import numpy as np
import sqlite3
import shutil
from random import randint
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import argparse
import ray
import time

parser = argparse.ArgumentParser(description='Augment with training set.')
parser.add_argument('-tb','--tile_base', type=str, help='Path to the base directory of the training set.', required=True)
parser.add_argument('-pa','--proportion_to_augment', type=float, default=1.0, help='Proportion of the training set to augment.', required=False)
parser.add_argument('-at','--augmentations_per_tile', type=int, default=2, help='Number of augmentations for each tile.', required=False)
parser.add_argument('-mx','--max_translation_x', type=int, default=300, help='Maximum number of pixels to translate in the x dimension.', required=False)
parser.add_argument('-my','--max_translation_y', type=int, default=50, help='Maximum number of pixels to translate in the y dimension.', required=False)
parser.add_argument('-pd','--proportion_to_delete', type=float, default=0.1, help='Proportion of the original training set to delete after augmentation.', required=False)
args = parser.parse_args()

# initialise Ray
if not ray.is_initialized():
    ray.init()

# load the tiles and their labels
TILE_BASE = args.tile_base
TRAINING_SET_FILES_DIR = '{}/train'.format(TILE_BASE)
TRAINING_SET_BACKUP_FILES_DIR = '{}/train-backup'.format(TILE_BASE)
AUGMENTED_FILES_DIR = '{}/augmented'.format(TILE_BASE)
AUGMENTED_OVERLAY_FILES_DIR = '{}/overlay'.format(AUGMENTED_FILES_DIR)

PIXELS = 910 # length of each tile edge in pixels

UBUNTU_FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
MACOS_FONT_PATH = '/Library/Fonts/Arial.ttf'

start_run = time.time()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))

# initialise the directories required for the data set creation
if os.path.exists(AUGMENTED_FILES_DIR):
    shutil.rmtree(AUGMENTED_FILES_DIR)
os.makedirs(AUGMENTED_FILES_DIR)
os.makedirs(AUGMENTED_OVERLAY_FILES_DIR)

# delete the augmented tiles already in the training set
augmented_files = glob.glob("{}/*-aug-*.png".format(TRAINING_SET_FILES_DIR))
for fname in augmented_files:
    if os.path.isfile(fname):
        os.remove(fname)

# load the file names into a dataframe
filenames = []
for file in glob.glob("{}/*.png".format(TRAINING_SET_FILES_DIR)):
    filenames.append((os.path.basename(os.path.splitext(file)[0])))

filenames_df = pd.DataFrame(filenames, columns=['filename'])
training_set_original_size = len(filenames_df)

number_to_select = int(training_set_original_size * args.proportion_to_augment)
filenames_to_augment_df = filenames_df.sample(n=number_to_select)

number_to_delete = int(training_set_original_size * args.proportion_to_delete)
filenames_to_delete_df = filenames_df.sample(n=number_to_delete)

print("generating {} augmentations of {} tiles".format(args.augmentations_per_tile, number_to_select))

@ray.remote
def augment_tile(filename, filename_idx):
    print("augmenting {} ({} of {})".format(filename, filename_idx+1, len(filenames_to_augment_df)))

    # load the font to use for labelling the overlays
    if os.path.isfile(UBUNTU_FONT_PATH):
        feature_label = ImageFont.truetype(UBUNTU_FONT_PATH, 10)
    else:
        feature_label = ImageFont.truetype(MACOS_FONT_PATH, 10)

    # load the tile
    img = Image.open('{}/{}.png'.format(TRAINING_SET_FILES_DIR, filename))
    # generate the augmented tiles
    for augmentation_idx in range(args.augmentations_per_tile):
        # apply the transformation
        x_offset = randint(-args.max_translation_x, args.max_translation_x+1)
        y_offset = randint(-args.max_translation_y, args.max_translation_y+1)
        # transformation matrix uses the inverse transformation
        # mapping (x,y) in the destination image to (ax + by + c, dx + ey + f) in the source image
        a = 1
        b = 0
        c = -x_offset # +ve is to the left, -ve is to the right
        d = 0
        e = 1
        f = -y_offset # +ve is up, -ve is down
        aug_img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        # now need to apply the same transformation to each label
        augment_label_l = []
        overlay_img = aug_img.copy() # copy the original so we can draw on it
        draw_context = ImageDraw.Draw(overlay_img)
        label_df = pd.read_csv('{}/{}.txt'.format(TRAINING_SET_FILES_DIR, filename), sep=' ', header=None, names=['class_id','x','y','w','h'])
        for label_idx in range(len(label_df)):
            class_id = int(label_df.iloc[label_idx].class_id)
            label_x = label_df.iloc[label_idx].x
            label_y = label_df.iloc[label_idx].y
            label_width = label_df.iloc[label_idx].w
            label_height = label_df.iloc[label_idx].h
            # get the current pixel coordinates
            pixel_x = int(label_x * PIXELS)
            pixel_y = int(label_y * PIXELS)
            pixel_width = int(label_width * PIXELS)
            pixel_height = int(label_height * PIXELS)
            # calculate the new label centre x,y in pixel coordinates
            augment_pixel_x = pixel_x + x_offset # new centre in pixel coordinates
            augment_pixel_y = pixel_y + y_offset
            # calculate the new label centre x,y in label coordinates
            augment_label_x = augment_pixel_x / PIXELS # new centre in label coordinates
            augment_label_y = augment_pixel_y / PIXELS
            # label the object if its centre is still within the tile
            if ((augment_label_x > 0) and (augment_label_x < 1) and (augment_label_y > 0) and (augment_label_y < 1)):
                # add it to the list of new labels
                augment_label_l.append(("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_id, augment_label_x, augment_label_y, label_width, label_height)))
                # calculate the overlay rectangle
                x0 = int(augment_pixel_x - (pixel_width / 2))
                x1 = int(augment_pixel_x + (pixel_width / 2))
                y0 = int(augment_pixel_y - (pixel_height / 2))
                y1 = int(augment_pixel_y + (pixel_height / 2))
                # draw the box annotation
                draw_context.rectangle(xy=[(x0, y0), (x1, y1)], fill=None, outline='red')
                draw_context.text((x0, y0-12), "feature class {}".format(class_id), font=feature_label, fill='red')
        # write out the augmented tile
        augmented_base_filename = '{}-aug-{}-{}'.format(filename, x_offset, y_offset)
        tile_filename = '{}/{}.png'.format(AUGMENTED_FILES_DIR, augmented_base_filename)
        aug_img.save(tile_filename)
        # save the label file
        label_filename = '{}/{}.txt'.format(AUGMENTED_FILES_DIR, augmented_base_filename)
        with open(label_filename, 'w') as f:
            for item in augment_label_l:
                f.write("%s\n" % item)
        # save the overlay image
        overlay_img.save('{}/{}.png'.format(AUGMENTED_OVERLAY_FILES_DIR, augmented_base_filename))

ray.get([augment_tile.remote(filename=filename, filename_idx=idx) for idx,filename in enumerate(filenames_to_augment_df.filename)])

print("shutting down ray")
ray.shutdown()

# keep a copy of the original training set
if os.path.exists(TRAINING_SET_BACKUP_FILES_DIR):
    shutil.rmtree(TRAINING_SET_BACKUP_FILES_DIR)
shutil.copytree(TRAINING_SET_FILES_DIR, TRAINING_SET_BACKUP_FILES_DIR)

# delete the specified proportion of the original, un-augmented training set
print("deleting {}% of the original training set".format(int(args.proportion_to_delete*100)))
for filename in filenames_to_delete_df.filename:
    image_file = "{}/{}.png".format(TRAINING_SET_FILES_DIR, filename)
    label_file = "{}/{}.txt".format(TRAINING_SET_FILES_DIR, filename)
    if os.path.isfile(image_file):
        os.remove(image_file)
    if os.path.isfile(label_file):
        os.remove(label_file)

# copy the augmented tiles to the training set directory
augmented_files = glob.glob("{}/*.*".format(AUGMENTED_FILES_DIR))
print("copying augmented tiles to the training set.")
for fname in augmented_files:
    if os.path.isfile(fname):
        basename = os.path.basename(fname)
        shutil.copyfile('{}/{}'.format(AUGMENTED_FILES_DIR, basename), '{}/{}'.format(TRAINING_SET_FILES_DIR, basename))

# regenerate the training file list
training_set_files = glob.glob("{}/*.png".format(TRAINING_SET_FILES_DIR))
training_set_augmented_size = len(training_set_files)
print("increased the training set from {} to {} tiles ({} augmented).".format(training_set_original_size, training_set_augmented_size, training_set_augmented_size-training_set_original_size))
print("refreshing the train-list file")
training_set_l = []
for fname in training_set_files:
    if os.path.isfile(fname):
        basename = os.path.basename(fname)
        training_set_l.append('data/peptides/train/{}'.format(basename))
df = pd.DataFrame(training_set_l, columns=['filename'])
df.to_csv("{}/data-files/train-list.txt".format(TILE_BASE), index=False, header=False)

stop_run = time.time()
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))
print("\n{} info: {}".format(parser.prog, info))
