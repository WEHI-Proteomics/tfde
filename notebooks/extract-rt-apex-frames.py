import argparse
import sqlite3
import pandas as pd
import sys
import pickle
import os


parser = argparse.ArgumentParser(description='Extract the specified frames for analysis of a sequence\'s raw data.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-fid','--frame_id_list', nargs='+', type=int, help='Space-separated frame ids.', required=False)
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

if args.frame_id_list is not None:
    if len(args.frame_id_list) == 1:
        frame_ids = '({})'.format(args.frame_id_list[0])
    else:
        frame_ids = str(tuple(args.frame_id_list))

    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    frames_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_id in {}".format(frame_ids), db_conn)
    db_conn.close()

    frames_file_name = './{}-frames-subset-df.pkl'.format(args.run_name)
    frames_df.to_pickle(frames_file_name)
    print("wrote {} entries to {}".format(len(frames_df), frames_file_name))

db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
frame_properties_df = pd.read_sql_query("select * from frame_properties", db_conn)
db_conn.close()

frame_properties_file_name = './{}-frame-properties-df.pkl'.format(args.run_name)
frame_properties_df.to_pickle(frame_properties_file_name)
print("wrote {} entries to {}".format(len(frame_properties_df), frame_properties_file_name))
