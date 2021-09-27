import glob
import os
import shutil
import time
import argparse
import sys
import pandas as pd
import sqlite3
import json
from multiprocessing import Pool

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

# nohup python -u ./open-path/pda/bulk-cuboid-extract.py -en dwm-test > bulk-cuboid-extract.log 2>&1 &

parser = argparse.ArgumentParser(description='Orchestrate the feature extraction of sequence library features from all runs.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_names', type=str, help='Comma-separated names of runs to process.', required=True)
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.', required=False)
parser.add_argument('-ssms','--small_set_mode_size', type=int, default='100', help='The number of identifications to sample for small set mode.', required=False)
parser.add_argument('-mpwrt','--max_peak_width_rt', type=int, default=10, help='Maximum peak width tolerance for the extraction from the estimated coordinate in RT.', required=False)
parser.add_argument('-mpwccs','--max_peak_width_ccs', type=int, default=20, help='Maximum peak width tolerance for the extraction from the estimated coordinate in CCS.', required=False)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.8, help='Proportion of the machine\'s cores to use for this program.', required=False)
args = parser.parse_args()

# print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

CONVERTED_DB_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(CONVERTED_DB_DIR):
    print("The converted databases directory is required but doesn't exist: {}".format(CONVERTED_DB_DIR))
    sys.exit(1)

# check the log directory exists
LOG_DIR = "{}/logs".format(EXPERIMENT_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# set up the target decoy classifier directory
TARGET_DECOY_MODEL_DIR = "{}/target-decoy-models".format(EXPERIMENT_DIR)
if os.path.exists(TARGET_DECOY_MODEL_DIR):
    shutil.rmtree(TARGET_DECOY_MODEL_DIR)
os.makedirs(TARGET_DECOY_MODEL_DIR)
print("The target-decoy classifier directory was deleted and re-created: {}".format(TARGET_DECOY_MODEL_DIR))

# the experiment metrics file
METRICS_DB_NAME = "{}/experiment-metrics-for-library-sequences.sqlite".format(TARGET_DECOY_MODEL_DIR)
if os.path.isfile(METRICS_DB_NAME):
    os.remove(METRICS_DB_NAME)

if args.small_set_mode:
    small_set_flags = "-ssm -ssms {}".format(args.small_set_mode_size)
else:
    small_set_flags = ""

# set up the processing pool
pool = Pool(processes=6)

run_names_l = args.run_names.split(',')
print("{} runs to process: {}".format(len(run_names_l), run_names_l))
extract_cmd_l = []
for run_name in run_names_l:
    print("processing {}".format(run_name))
    LOG_FILE_NAME = "{}/extract-library-sequence-features-for-run-{}.log".format(LOG_DIR, run_name)
    current_directory = os.path.abspath(os.path.dirname(__file__))
    cmd = "python -u {}/extract-library-sequence-features-for-run.py -eb {} -en {} -rn {} -mpwrt {} -mpwccs {} {} > {} 2>&1".format(current_directory, args.experiment_base_dir, args.experiment_name, run_name, args.max_peak_width_rt, args.max_peak_width_ccs, small_set_flags, LOG_FILE_NAME)
    extract_cmd_l.append(cmd)
pool.map(run_process, extract_cmd_l)

# load the run-based metrics into a single experiment-based dataframe
run_sequence_files = glob.glob('{}/library-sequences-in-run-*.feather'.format(TARGET_DECOY_MODEL_DIR))
print("found {} sequence files to consolidate into an experiment set and stored in {}.".format(len(run_sequence_files), METRICS_DB_NAME))
# load the run-lavel metrics into a database
db_conn = sqlite3.connect(METRICS_DB_NAME)
for file in run_sequence_files:
    df = pd.read_feather(file)
    # convert the lists and dictionaries to strings
    df.target_coords = df.apply(lambda row: json.dumps(row.target_coords), axis=1)
    df.decoy_coords = df.apply(lambda row: json.dumps(row.decoy_coords), axis=1)
    df.target_metrics = df.apply(lambda row: json.dumps(row.target_metrics), axis=1)
    df.decoy_metrics = df.apply(lambda row: json.dumps(row.decoy_metrics), axis=1)
    df.attributes = df.apply(lambda row: json.dumps(row.attributes), axis=1)
    # count the sequence peak instances
    peak_counts_l = []
    for group_name,group_df in df.groupby(['sequence','charge','run_name'], as_index=False):
        peak_counts_l.append(tuple(group_name) + (len(group_df),))
    peak_counts_df = pd.DataFrame(peak_counts_l, columns=['sequence','charge','run_name','peak_count'])
    df = pd.merge(df, peak_counts_df, how='left', left_on=['sequence','charge','run_name'], right_on=['sequence','charge','run_name'])
    # store the metrics in the database
    df.to_sql(name='extracted_metrics', con=db_conn, if_exists='append', index=False)
db_conn.close()

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
