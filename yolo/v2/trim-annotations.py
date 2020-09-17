import glob
import os
import argparse
import shutil
import sys
import json
import pandas as pd



###########################
parser = argparse.ArgumentParser(description='Trim the annotations.')
parser.add_argument('-af','--annotation_file_name', type=str, help='The file name of the annotations file to trim.', required=True)
parser.add_argument('-fid','--frame_id_list_to_keep', type=str, help='Frame IDs of the frames to keep in the annotations; all others are discarded. Can specify several ranges (e.g. 10-20,21-30,31-40), a single range (e.g. 10-24), individual frames (e.g. 34,56,32), or a single frame (e.g. 54).', required=True)

args = parser.parse_args()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))

# find the path and the base name of the annotation file name
annotations_base_name = os.path.basename(args.annotation_file_name)
annotations_dir = os.path.dirname(args.annotation_file_name)

# parse the tile indexes
indexes_l = []
for item in args.frame_id_list_to_keep.replace(" ", "").split(','):
    index_range = item.split('-')
    if all([i.isnumeric() for i in index_range]):  # only use the range if it's valid
        index_range = [int(i) for i in index_range]
        if len(index_range) == 2:
            index_lower = min(index_range)
            index_upper = max(index_range)
            indexes_l.append([i for i in range(index_lower, index_upper+1)])
        else:
            indexes_l.append(index_range)
indexes_l = [item for sublist in indexes_l for item in sublist]
if len(indexes_l) == 0:
    print("Need to specify at least one frame ID to include: {}".format(args.frame_id_list))
    sys.exit(1)

# load the annotations file
print('processing {}'.format(args.annotation_file_name))
with open(args.annotation_file_name) as file:
    annotations = json.load(file)

# go through the annotations file and find the keys to be removed
keys_to_remove = []
for tile_key in list(annotations.keys()):
    tile_d = annotations[tile_key]
    tile_base_name = tile_d['file_attributes']['source']['tile']['base_name']
    splits = tile_base_name.split('-')
    frame_id = int(splits[3])
    if not frame_id in indexes_l:
        keys_to_remove.append(tile_key)

# now remove them
for key_to_remove in keys_to_remove:
    del annotations[key_to_remove]

# write out the trimmed annotations file
trimmed_annotations_file_name = '{}/{}-trimmed.json'.format(annotations_dir, annotations_base_name.split('.')[0])
with open(trimmed_annotations_file_name, 'w') as outfile:
    json.dump(annotations, outfile)

print("wrote out {} keys to {}".format(len(annotations.keys()), trimmed_annotations_file_name))
