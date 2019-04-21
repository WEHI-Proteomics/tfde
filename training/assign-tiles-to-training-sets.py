# Split the tiles into training, test, validation sets

# load the file names into a dataframe
import glob, os
import shutil
import sqlite3
import pandas as pd
import numpy as np
import argparse

MS1_CE = 10

SET_GAP = 10  # number of frames between training sets to avoid features appearing in more than one set

TRAINING_SET_PROPORTION = 0.8
VALIDATION_SET_PROPORTION = 0.1
TEST_SET_PROPORTION = 0.1

parser = argparse.ArgumentParser(description='Assign the tiles to a training set.')
parser.add_argument('-rtl','--rt_lower', type=int, help='Lower bound of the RT range.', required=True)
parser.add_argument('-rtu','--rt_upper', type=int, help='Upper bound of the RT range.', required=True)
parser.add_argument('-tb','--tile_base', type=str, help='Path to the base directory of the training set.', required=True)
parser.add_argument('-dbb','--database_base', type=str, help='Path to the base directory of the raw database.', required=True)
args = parser.parse_args()

BASE_NAME = args.database_base
CONVERTED_DATABASE_NAME = '{}/HeLa_20KInt.sqlite'.format(BASE_NAME)

TILE_BASE = args.tile_base
PRE_ASSIGNED_FILES_DIR = '{}/pre-assigned'.format(TILE_BASE)
OVERLAY_FILES_DIR = '{}/overlay'.format(TILE_BASE)

NUMBER_OF_CLASSES = 4

if not input("This will erase the training set directories in {}. Are you sure? (y/n): ".format(TILE_BASE)).lower().strip()[:1] == "y": sys.exit(1)

db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(args.rt_lower, args.rt_upper, MS1_CE), db_conn)
db_conn.close()

filenames = []
for file in glob.glob("{}/*.png".format(PRE_ASSIGNED_FILES_DIR)):
    filenames.append((os.path.basename(os.path.splitext(file)[0])))

fn_df = pd.DataFrame(filenames, columns=['name'])

# allocate the training, validation, and test sets from separate periods of RT, as frames are a time series - with an N-frame gap between sets
def train_validate_test_split_v4(df, train_percent=TRAINING_SET_PROPORTION, validate_percent=VALIDATION_SET_PROPORTION, seed=None):
    test_percent = 1.0 - (train_percent + validate_percent)

    # split the ms1 frame ids into three sections according to their proportions
    split_1 = int(train_percent*len(ms1_frame_properties_df))
    split_1_plus_gap = split_1 + SET_GAP
    split_2 = int((train_percent+validate_percent)*len(ms1_frame_properties_df))
    split_2_plus_gap = split_2 + SET_GAP
    train_ids_df, gap_1_df, valid_ids_df, gap_2_df, test_ids_df = np.split(ms1_frame_properties_df, [split_1, split_1_plus_gap, split_2, split_2_plus_gap])
    print("training set: {:.1f} to {:.1f} secs ({} frames)".format(train_ids_df.retention_time_secs.min(), train_ids_df.retention_time_secs.max(), len(train_ids_df)))
    print("validation set: {:.1f} to {:.1f} secs ({} frames)".format(valid_ids_df.retention_time_secs.min(), valid_ids_df.retention_time_secs.max(), len(valid_ids_df)))
    print("test set: {:.1f} to {:.1f} secs ({} frames)".format(test_ids_df.retention_time_secs.min(), test_ids_df.retention_time_secs.max(), len(test_ids_df)))
    
    train_terms = ['frame-' + str(s) for s in train_ids_df.frame_id]
    train_df = df[df['name'].str.contains('|'.join(train_terms))].copy()

    valid_terms = ['frame-' + str(s) for s in valid_ids_df.frame_id]
    valid_df = df[df['name'].str.contains('|'.join(valid_terms))].copy()

    test_terms = ['frame-' + str(s) for s in test_ids_df.frame_id]
    test_df = df[df['name'].str.contains('|'.join(test_terms))].copy()

    return train_df, valid_df, test_df

train_df,validate_df,test_df = train_validate_test_split_v4(fn_df)

print("number of images in train set: {}, validation set: {}, test set {}".format(len(train_df),len(validate_df),len(test_df)))

data_dirs = ['train', 'validation', 'test']
data_dfs = [train_df, validate_df, test_df]

DESTINATION_DATASET_BASE = 'data/peptides'
LOCAL_DATA_FILES_DIR = '{}/data-files'.format(TILE_BASE)
DESTINATION_DATA_FILES_DIR = '{}/data-files'.format(DESTINATION_DATASET_BASE)
FILE_LIST_SUFFIX = 'list'

# initialise the directories required for the data set organisation
if os.path.exists(LOCAL_DATA_FILES_DIR):
    shutil.rmtree(LOCAL_DATA_FILES_DIR)
os.makedirs(LOCAL_DATA_FILES_DIR)

for idx,dd in enumerate(data_dirs):
    print("processing {}".format(dd))
    
    # initialise the directory for each set
    data_dir = "{}/{}".format(TILE_BASE, dd)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    
    # create the prefix column
    data_dfs[idx]['text_entry'] = '{}/{}/'.format(DESTINATION_DATASET_BASE, dd) + data_dfs[idx].name + '.png'

    # copy the files into their respective directories
    for r in zip(data_dfs[idx].name):
        shutil.copyfile("{}/{}.png".format(PRE_ASSIGNED_FILES_DIR,r[0]), "{}/{}/{}.png".format(TILE_BASE, dd, r[0]))
        shutil.copyfile("{}/{}.txt".format(PRE_ASSIGNED_FILES_DIR,r[0]), "{}/{}/{}.txt".format(TILE_BASE, dd, r[0]))

    # generate the text files for each set
    data_dfs[idx].text_entry.to_csv("{}/{}-{}.txt".format(LOCAL_DATA_FILES_DIR, dd, FILE_LIST_SUFFIX), index=False, header=False)

# create the names and data files for Darknet
LOCAL_NAMES_FILENAME = "{}/peptides-obj.names".format(TILE_BASE)
DESTINATION_NAMES_FILENAME = "{}/peptides-obj.names".format(DESTINATION_DATASET_BASE)
LOCAL_DATA_FILENAME = "{}/peptides-obj.data".format(TILE_BASE)

with open(LOCAL_NAMES_FILENAME, 'w') as f:
    for charge in range(1,NUMBER_OF_CLASSES+1):
        f.write("charge-{}\n".format(charge))

print("finished writing {}".format(LOCAL_NAMES_FILENAME))

with open(LOCAL_DATA_FILENAME, 'w') as f:
    f.write("classes={}\n".format(NUMBER_OF_CLASSES))
    f.write("train={}\n".format("{}/{}-{}.txt".format(DESTINATION_DATA_FILES_DIR, data_dirs[0], FILE_LIST_SUFFIX)))
    f.write("valid={}\n".format("{}/{}-{}.txt".format(DESTINATION_DATA_FILES_DIR, data_dirs[1], FILE_LIST_SUFFIX)))
    f.write("names={}\n".format(DESTINATION_NAMES_FILENAME))
    f.write("backup=backup/\n")

print("finished writing {}".format(LOCAL_DATA_FILENAME))
