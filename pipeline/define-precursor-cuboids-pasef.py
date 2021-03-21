import os
import numpy as np
import pandas as pd
import sqlite3
import argparse
import time
import shutil
import json
import configparser
from configparser import ExtendedInterpolation
import multiprocessing as mp
import ray

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers

# returns a dataframe with the prepared isolation windows
def load_isolation_windows(cfg):
    # get all the isolation windows
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    isolation_window_df = pd.read_sql_query("select * from isolation_windows order by Precursor", db_conn)
    db_conn.close()

    isolation_window_df['mz_lower'] = isolation_window_df.IsolationMz - (isolation_window_df.IsolationWidth / 2) - cfg.getfloat('ms2', 'MS2_MZ_ISOLATION_WINDOW_EXTENSION')
    isolation_window_df['mz_upper'] = isolation_window_df.IsolationMz + (isolation_window_df.IsolationWidth / 2) + cfg.getfloat('ms2', 'MS2_MZ_ISOLATION_WINDOW_EXTENSION')

    if args.small_set_mode:
        # select a subset around the middle
        start_idx = int(len(isolation_window_df) / 2)
        stop_idx = start_idx + 20
        isolation_window_df = isolation_window_df[start_idx:stop_idx]

    print("loaded {} isolation windows from {}".format(len(isolation_window_df), CONVERTED_DATABASE_NAME))
    return isolation_window_df

# returns a dataframe with the frame properties
def load_frame_properties():
    # get all the isolation windows
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    frames_properties_df = pd.read_sql_query("select * from frame_properties order by Id ASC;", db_conn)
    db_conn.close()

    print("loaded {} frame_properties from {}".format(len(frames_properties_df), CONVERTED_DATABASE_NAME))
    return frames_properties_df

def metadata_for_frame(frame_properties_df, frame_id):
    r = frame_properties_df[(frame_properties_df.Id == frame_id)].iloc[0]
    result = {}
    result['retention_time_secs'] = r.Time
    result['frame_type'] = r.MsMsType
    return result

# find the closest lower ms1 frame_id, and the closest upper ms1 frame_id
# number_of_ms1_frames_padding is the number of ms1 frames to extend
def find_closest_ms1_frame_to_rt(cfg, frames_properties_df, retention_time_secs, number_of_ms1_frames_padding=0):
    ms1_frame_props_df = frames_properties_df[(frames_properties_df.MsMsType == cfg.getint('common','FRAME_TYPE_MS1'))]
    # find the closest ms1 frame above
    df = ms1_frame_props_df[(ms1_frame_props_df.Time > retention_time_secs)]
    if len(df) > (number_of_ms1_frames_padding+1):
        closest_ms1_frame_above_rt = df.iloc[number_of_ms1_frames_padding].Id
    else:
        # couldn't find an ms1 frame above this RT, so just use the last one
        closest_ms1_frame_above_rt = ms1_frame_props_df.Id.max()
    # find the closest ms1 frame below
    df = ms1_frame_props_df[(ms1_frame_props_df.Time < retention_time_secs)]
    if len(df) > (number_of_ms1_frames_padding+1):
        closest_ms1_frame_below_rt = df.iloc[-(number_of_ms1_frames_padding+1)].Id
    else:
        # couldn't find an ms1 frame below this RT, so just use the first one
        closest_ms1_frame_below_rt = ms1_frame_props_df.Id.min()
    return (closest_ms1_frame_below_rt, closest_ms1_frame_above_rt)

# find the closest lower ms1 frame_id, and the closest upper ms1 frame_id
def find_closest_ms1_frame_to_ms2_frame(cfg, frames_properties_df, ms2_frame_id):
    # get the ms2 frame's RT
    ms2_frame_rt = frames_properties_df[frames_properties_df.Id == ms2_frame_id].iloc[0].Time
    # get the closest ms1 frames
    return find_closest_ms1_frame_to_rt(cfg, frames_properties_df, ms2_frame_rt, cfg.getint('ms1','RT_FRAGMENT_EVENT_DELTA_FRAMES'))

@ray.remote
def process_precursor(cfg, frame_properties_df, precursor_id, precursor_group_df):
    # calculate the coordinates
    window = precursor_group_df.iloc[0]
    window_mz_lower = window.mz_lower                                             # the isolation window's m/z range
    window_mz_upper = window.mz_upper
    wide_mz_lower = window_mz_lower - (cfg.getfloat('common','CARBON_MASS_DIFFERENCE') / 1)            # get more points in case we need to look for a missed monoisotopic peak - assume charge 1+ to allow for maximum distance to the left
    wide_mz_upper = window_mz_upper
    scan_width = int(window.ScanNumEnd - window.ScanNumBegin)                     # the isolation window's scan range
    fe_scan_lower = int(window.ScanNumBegin)                                      # fragmentation event scan range
    fe_scan_upper = int(window.ScanNumEnd)
    wide_scan_lower = int(window.ScanNumBegin - scan_width)                       # get more points to make sure we get the apex of the peak in drift
    wide_scan_upper = int(window.ScanNumEnd + scan_width)
    fe_ms2_frame_lower = precursor_group_df.Frame.astype(int).min()               # only the ms2 frames associated with the precursor
    fe_ms2_frame_upper = precursor_group_df.Frame.astype(int).max()
    fe_ms1_frame_lower,_ = find_closest_ms1_frame_to_ms2_frame(cfg, frame_properties_df,fe_ms2_frame_lower)
    _,fe_ms1_frame_upper = find_closest_ms1_frame_to_ms2_frame(cfg, frame_properties_df,fe_ms2_frame_upper)
    wide_rt_lower = metadata_for_frame(frame_properties_df, precursor_group_df.Frame.astype(int).min())['retention_time_secs'] - cfg.getfloat('ms1','RT_BASE_PEAK_WIDTH_SECS')  # get more points to make sure we get the apex of the peak in retention time
    wide_rt_upper = metadata_for_frame(frame_properties_df, precursor_group_df.Frame.astype(int).max())['retention_time_secs'] + cfg.getfloat('ms1','RT_BASE_PEAK_WIDTH_SECS')
    wide_frame_lower,_ = find_closest_ms1_frame_to_rt(cfg, frame_properties_df, wide_rt_lower)
    _,wide_frame_upper = find_closest_ms1_frame_to_rt(cfg, frame_properties_df, wide_rt_upper)

    # collect the coordinates for the precursor cuboid
    precursor_coordinates_columns = ['precursor_id', 'window_mz_lower', 'window_mz_upper', 'wide_mz_lower', 'wide_mz_upper', 'window_scan_width', 'fe_scan_lower', 'fe_scan_upper', 'wide_scan_lower', 'wide_scan_upper', 'wide_rt_lower', 'wide_rt_upper', 'fe_ms1_frame_lower', 'fe_ms1_frame_upper', 'fe_ms2_frame_lower', 'fe_ms2_frame_upper', 'wide_frame_lower', 'wide_frame_upper', 'number_of_windows']
    precursor_coordinates_values = [int(precursor_id), window_mz_lower, window_mz_upper, wide_mz_lower, wide_mz_upper, int(scan_width), int(fe_scan_lower), int(fe_scan_upper), int(wide_scan_lower), int(wide_scan_upper), wide_rt_lower, wide_rt_upper, int(fe_ms1_frame_lower), int(fe_ms1_frame_upper), int(fe_ms2_frame_lower), int(fe_ms2_frame_upper), int(wide_frame_lower), int(wide_frame_upper), len(precursor_group_df)]
    precursor_coordinates_d = {}
    for idx,c in enumerate(precursor_coordinates_columns):
        precursor_coordinates_d[c] = precursor_coordinates_values[idx]
    return precursor_coordinates_d

##############################################
parser = argparse.ArgumentParser(description='Extract the precursor cuboids from the Bruker instrument database to work units based on the precursors.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.')
parser.add_argument('-rm','--ray_mode', type=str, choices=['cluster','join','local'], help='The Ray mode to use.', required=True)
parser.add_argument('-ra','--redis_address', type=str, help='Address of the cluster to join.', required=False)
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

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.ini_file)

# set up the precursor cuboids
CUBOIDS_DIR = '{}/precursor-cuboids'.format(EXPERIMENT_DIR)
if not os.path.exists(CUBOIDS_DIR):
    os.makedirs(CUBOIDS_DIR)

CUBOIDS_COORDS_FILE = '{}/exp-{}-run-{}-precursor-cuboid-coords-pasef.pkl'.format(CUBOIDS_DIR, args.experiment_name, args.run_name)

# get the frame metadata
print("loading the frames information")
frame_properties_df = load_frame_properties()

# load the isolation windows table
print("loading the isolation windows")
isolation_window_df = load_isolation_windows(config)

print("Setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(object_store_memory=20000000000,
                    redis_max_memory=25000000000,
                    num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

# determine the coordinates of each precursor's cuboid
print("extracting the raw points for each precursor cuboid for {} precursors".format(len(isolation_window_df.Precursor.unique())))
coords_l = ray.get([process_precursor.remote(cfg=config, frame_properties_df=frame_properties_df, precursor_id=group_name, precursor_group_df=group_df) for group_name,group_df in isolation_window_df.groupby('Precursor')])
coords_df = pd.DataFrame(coords_l)
coords_df.to_pickle(CUBOIDS_COORDS_FILE)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))