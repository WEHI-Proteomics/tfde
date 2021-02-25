import pandas as pd
import numpy as np
import sys
import os.path
import argparse
import time
import configparser
from configparser import ExtendedInterpolation
import json
import multiprocessing as mp
sys.path.append('/home/ubuntu/open-path/pda/packaged/')
from process_precursor_cuboid_ms1 import find_features, check_monoisotopic_peak
from argparse import Namespace
import ray
import sqlite3
import shutil

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# returns a dataframe with the frame properties
def load_frame_properties(converted_db_name):
    # get all the isolation windows
    db_conn = sqlite3.connect(converted_db_name)
    frames_properties_df = pd.read_sql_query("select * from frame_properties order by Id ASC;", db_conn)
    db_conn.close()

    print("loaded {} frame_properties from {}".format(len(frames_properties_df), converted_db_name))
    return frames_properties_df

# find the closest lower ms1 frame_id, and the closest upper ms1 frame_id
def find_closest_ms1_frame_to_rt(frames_properties_df, retention_time_secs):
    # find the frame ids within this range of RT
    df = frames_properties_df[(frames_properties_df.Time > retention_time_secs) & (frames_properties_df.MsMsType == FRAME_TYPE_MS1)]
    if len(df) > 0:
        closest_ms1_frame_above_rt = df.Id.min()
    else:
        # couldn't find an ms1 frame above this RT, so just use the last one
        closest_ms1_frame_above_rt = frames_properties_df[(frames_properties_df.MsMsType == FRAME_TYPE_MS1)].Id.max()
    df = frames_properties_df[(frames_properties_df.Time < retention_time_secs) & (frames_properties_df.MsMsType == FRAME_TYPE_MS1)]
    if len(df) > 0:
        closest_ms1_frame_below_rt = df.Id.max()
    else:
        # couldn't find an ms1 frame below this RT, so just use the first one
        closest_ms1_frame_below_rt = frames_properties_df[(frames_properties_df.MsMsType == FRAME_TYPE_MS1)].Id.min()
    result = {}
    result['below'] = closest_ms1_frame_below_rt
    result['above'] = closest_ms1_frame_above_rt
    return result

# process a precursor cuboid to detect ms1 features
def ms1(precursor_metadata, ms1_points_df, args):
    # find features in the cuboid
    checked_features_l = []
    features_df = find_features(precursor_metadata, ms1_points_df, args)
    if features_df is not None:
        features_df.reset_index(drop=True, inplace=True)
        for idx,feature in features_df.iterrows():
            feature_d = check_monoisotopic_peak(feature=feature, raw_points=ms1_points_df, idx=idx, total=len(features_df), args=args)
            checked_features_l.append(feature_d)

    checked_features_df = pd.DataFrame(checked_features_l)
    if len(checked_features_df) > 0:
        checked_features_df['monoisotopic_mass'] = (checked_features_df.monoisotopic_mz * checked_features_df.charge) - (args.PROTON_MASS * checked_features_df.charge)
    print("found {} features for precursor {}".format(len(checked_features_df), precursor_metadata['precursor_id']))
    return checked_features_df

# prepare the metadata and raw points for the feature detection
@ray.remote
def detect_ms1_features(precursor_cuboid_row, converted_db_name):

    # use the ms1 function to perform the feature detection step
    ms1_args = Namespace()
    # ms1_args.experiment_name = args.experiment_name
    # ms1_args.run_name = args.run_name
    ms1_args.MS1_PEAK_DELTA = config.getfloat('ms1', 'MS1_PEAK_DELTA')
    ms1_args.SATURATION_INTENSITY = config.getfloat('common', 'SATURATION_INTENSITY')
    ms1_args.MAX_MS1_PEAK_HEIGHT_RATIO_ERROR = config.getfloat('ms1', 'MAX_MS1_PEAK_HEIGHT_RATIO_ERROR')
    ms1_args.PROTON_MASS = config.getfloat('common', 'PROTON_MASS')
    ms1_args.INSTRUMENT_RESOLUTION = config.getfloat('common', 'INSTRUMENT_RESOLUTION')
    ms1_args.NUMBER_OF_STD_DEV_MZ = config.getfloat('ms1', 'NUMBER_OF_STD_DEV_MZ')
    ms1_args.CARBON_MASS_DIFFERENCE = config.getfloat('common', 'CARBON_MASS_DIFFERENCE')

    # create the metadata record
    cuboid_metadata = {}
    cuboid_metadata['precursor_id'] = precursor_cuboid_row.precursor_cuboid_id
    cuboid_metadata['window_mz_lower'] = precursor_cuboid_row.mz_lower - (ms1_args.CARBON_MASS_DIFFERENCE / 1) # get more points in case we need to look for a missed monoisotopic peak - assume charge 1+ to allow for maximum distance to the left
    cuboid_metadata['window_mz_upper'] = precursor_cuboid_row.mz_upper
    cuboid_metadata['wide_mz_lower'] = precursor_cuboid_row.mz_lower - (ms1_args.CARBON_MASS_DIFFERENCE / 1) # get more points in case we need to look for a missed monoisotopic peak - assume charge 1+ to allow for maximum distance to the left
    cuboid_metadata['wide_mz_upper'] = precursor_cuboid_row.mz_upper
    cuboid_metadata['window_scan_width'] = precursor_cuboid_row.scan_upper - precursor_cuboid_row.scan_lower
    cuboid_metadata['fe_scan_lower'] = precursor_cuboid_row.scan_lower
    cuboid_metadata['fe_scan_upper'] = precursor_cuboid_row.scan_upper
    cuboid_metadata['wide_scan_lower'] = precursor_cuboid_row.scan_lower
    cuboid_metadata['wide_scan_upper'] = precursor_cuboid_row.scan_upper
    cuboid_metadata['wide_rt_lower'] = precursor_cuboid_row.rt_lower
    cuboid_metadata['wide_rt_upper'] = precursor_cuboid_row.rt_upper
    cuboid_metadata['fe_ms1_frame_lower'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=precursor_cuboid_row.rt_lower)['below']
    cuboid_metadata['fe_ms1_frame_upper'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=precursor_cuboid_row.rt_upper)['above']
    cuboid_metadata['fe_ms2_frame_lower'] = None
    cuboid_metadata['fe_ms2_frame_upper'] = None
    cuboid_metadata['wide_frame_lower'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=precursor_cuboid_row.rt_lower)['below']
    cuboid_metadata['wide_frame_upper'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=precursor_cuboid_row.rt_upper)['above']
    cuboid_metadata['number_of_windows'] = 1

    # load the raw points for this cuboid
    db_conn = sqlite3.connect(converted_db_name)
    ms1_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {}".format(FRAME_TYPE_MS1, cuboid_metadata['wide_mz_lower'], precursor_cuboid_row.mz_upper, precursor_cuboid_row.scan_lower, precursor_cuboid_row.scan_upper, precursor_cuboid_row.rt_lower, precursor_cuboid_row.rt_upper), db_conn)
    db_conn.close()

    # adjust the args
    ms1_args.precursor_id = precursor_cuboid_row.precursor_cuboid_id

    # detect the features
    df = ms1(precursor_metadata=cuboid_metadata, ms1_points_df=ms1_points_df, args=ms1_args)
    return df

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers

###################################
parser = argparse.ArgumentParser(description='Detect the features precursor cuboids found in a run with 3D intensity descent.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-ml','--mz_lower', type=int, default='100', help='Lower limit for m/z.', required=False)
parser.add_argument('-mu','--mz_upper', type=int, default='1700', help='Upper limit for m/z.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./open-path/pda/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-pid', '--precursor_id', type=int, help='Only process this precursor ID.', required=False)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], help='The Ray mode to use.', required=True)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.9, help='Proportion of the machine\'s cores to use for this program.', required=False)
args = parser.parse_args()

# Print the arguments for the log
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

# check the converted databases directory exists
CONVERTED_DATABASE_NAME = "{}/converted-databases/exp-{}-run-{}-converted.sqlite".format(EXPERIMENT_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

CUBOIDS_DIR = "{}/precursor-cuboids-3did".format(EXPERIMENT_DIR)
CUBOIDS_FILE = '{}/exp-{}-run-{}-mz-{}-{}-precursor-cuboids.pkl'.format(CUBOIDS_DIR, args.experiment_name, args.run_name, args.mz_lower, args.mz_upper)

# check the cuboids file
if not os.path.isfile(CUBOIDS_FILE):
    print("The cuboids file is required but doesn't exist: {}".format(CUBOIDS_FILE))
    sys.exit(1)

# load the precursor cuboids
precursor_cuboids_df = pd.read_pickle(CUBOIDS_FILE)
print('loaded {} precursor cuboids from {}'.format(len(precursor_cuboids_df), CUBOIDS_FILE))

# limit the cuboids to just the selected one
if args.precursor_id is not None:
    precursor_cuboids_df = precursor_cuboids_df[(precursor_cuboids_df.precursor_cuboid_id == args.precursor_id)]

# parse the config file
config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.ini_file)

# load the frame properties
frames_properties_df = load_frame_properties(CONVERTED_DATABASE_NAME)

FEATURES_DIR = "{}/features-3did".format(EXPERIMENT_DIR)
FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name)

# set up the output directory
if os.path.exists(FEATURES_DIR):
    shutil.rmtree(FEATURES_DIR)
os.makedirs(FEATURES_DIR)

# set up Ray
print("setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

# find the features in each precursor cuboid
features_l = ray.get([detect_ms1_features.remote(precursor_cuboid_row=row, converted_db_name=CONVERTED_DATABASE_NAME) for row in precursor_cuboids_df.itertuples()])
# join the list of dataframes into a single dataframe
features_df = pd.concat(features_l, axis=0, sort=False)

print("writing {} features to {}".format(len(features_df), FEATURES_FILE))
features_df.to_pickle(FEATURES_FILE)
