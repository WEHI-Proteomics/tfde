import os
import sys
if os.path.exists('/home/ubuntu/open-path/timstof-sdk/'):
    sys.path.insert(1, '/home/ubuntu/otf-peak-detect/timstof-sdk/')
else:
    sys.path.insert(1, '/home/daryl/otf-peak-detect/timstof-sdk/')
import timsdata
import numpy as np
import pandas as pd
import sqlite3
import argparse
import time
import os.path
import configparser
from configparser import ExtendedInterpolation

# Convert a raw instrument database to an interim schema so we can set up database indexes that make the next step super-speedy
# python ./open-path/pda/convert-raw-database.py -en dwm-test -rn 190719_Hela_Ecoli_1to1_01 -rdd ./experiments/dwm-test/raw-databases/190719_Hela_Ecoli_1to1_01_Slot1-1_1_1926.d -ini ./open-path/pda/pasef-process-short-gradient.ini

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# returns a dataframe with the prepared isolation windows
def load_isolation_windows(database_name, small_set_mode):
    # get all the isolation windows
    db_conn = sqlite3.connect(database_name)
    isolation_window_df = pd.read_sql_query("select * from PasefFrameMsMsInfo order by Precursor", db_conn)
    db_conn.close()

    isolation_window_df['mz_lower'] = isolation_window_df.IsolationMz - (isolation_window_df.IsolationWidth / 2) - MS2_MZ_ISOLATION_WINDOW_EXTENSION
    isolation_window_df['mz_upper'] = isolation_window_df.IsolationMz + (isolation_window_df.IsolationWidth / 2) + MS2_MZ_ISOLATION_WINDOW_EXTENSION

    if small_set_mode:
        # select a subset around the middle
        start_idx = int(len(isolation_window_df) / 2)
        stop_idx = start_idx + 20
        isolation_window_df = isolation_window_df[start_idx:stop_idx]

    print("loaded {} isolation windows from {}".format(len(isolation_window_df), database_name))
    return isolation_window_df

# returns a dataframe with the frame properties
def load_frame_properties(database_name):
    # get all the isolation windows
    db_conn = sqlite3.connect(database_name)
    frames_properties_df = pd.read_sql_query("select * from Frames order by Id ASC;", db_conn)
    db_conn.close()

    print("loaded {} frame_properties from {}".format(len(frames_properties_df), database_name))
    return frames_properties_df

# load the raw points within the given frame and scan range
def load_raw_points(frame_lower, frame_upper):
    # check parameter ranges
    min_frames = frames_properties_df.Id.min()
    max_frames = frames_properties_df.Id.max()-1
    if frame_lower < min_frames:
        frame_lower = min_frames
    if frame_upper > max_frames:
        frame_upper = max_frames

    # connect to the database with the timsTOF SDK
    td = timsdata.TimsData(args.raw_database_directory)

    # read the raw points in the specified frame range
    frame_points = []
    for frame_id in range(frame_lower, frame_upper+1):
        # find the metadata for this frame
        frame_info = frames_properties_df[(frames_properties_df.Id == frame_id)].iloc[0]
        retention_time_secs = frame_info['Time']
        frame_type = int(frame_info['MsMsType'])
        number_of_scans = int(frame_info['NumScans'])

        # set up one_over_k0 and voltage values for each scan
        scan_number_arr = np.arange(number_of_scans, dtype=np.float64)
        one_over_k0_arr = td.scanNumToOneOverK0(frame_id, scan_number_arr)
        voltage_arr = td.scanNumToVoltage(frame_id, scan_number_arr)

        # read the points from the scan lines
        for scan_idx,scan in enumerate(td.readScans(frame_id=frame_id, scan_begin=0, scan_end=number_of_scans)):
            index = np.array(scan[0], dtype=np.float64)
            mz_values = td.indexToMz(frame_id, index)
            intensity_values = scan[1]
            scan_number = scan_idx
            one_over_k0 = one_over_k0_arr[scan_idx]
            voltage = voltage_arr[scan_idx]
            number_of_points_on_scan = len(mz_values)
            for i in range(0, number_of_points_on_scan):   # step through the readings (i.e. points) on this scan line
                mz_value = float(mz_values[i])
                intensity = int(intensity_values[i])
                frame_points.append({'frame_id':frame_id, 'frame_type':frame_type, 'mz':mz_value, 'scan':scan_number, 'intensity':intensity, 'retention_time_secs':retention_time_secs, 'one_over_k0':one_over_k0, 'voltage':voltage})
    points_df = pd.DataFrame(frame_points)
    return points_df


##############################################
parser = argparse.ArgumentParser(description='Transform the raw database to a processing schema.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-rdd','--raw_database_directory', type=str, help='The full path to the directory (i.e. the \'.d\' path) of the raw database.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-nfb','--number_of_frames_in_batch', type=int, default=2000, help='The number of frames in a batch.', required=False)
parser.add_argument('-ssm', '--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.')
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

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.ini_file)

RT_BASE_PEAK_WIDTH_SECS = config.getfloat('common', 'RT_BASE_PEAK_WIDTH_SECS')
CARBON_MASS_DIFFERENCE = config.getfloat('common', 'CARBON_MASS_DIFFERENCE')
MS2_MZ_ISOLATION_WINDOW_EXTENSION = config.getfloat('ms2', 'MS2_MZ_ISOLATION_WINDOW_EXTENSION')

# check the path to the raw database exists
if not os.path.exists(args.raw_database_directory):
    print("The path to the raw database doesn't exist: {}".format(args.raw_database_directory))
    sys.exit(1)

# check the run directory exists
RUN_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

# check the analysis.tdf exists
RAW_DATABASE_NAME = "{}/analysis.tdf".format(args.raw_database_directory)
if not os.path.isfile(RAW_DATABASE_NAME):
    print("The raw database analysis.tdf doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)

RUN_DB_NAME = "{}/exp-{}-run-{}-converted.sqlite".format(RUN_DIR, args.experiment_name, args.run_name)

# remove it if there's one there from last time
if os.path.isfile(RUN_DB_NAME):
    os.remove(RUN_DB_NAME)

# get the frame metadata
print("loading the frames information")
frames_properties_df = load_frame_properties(RAW_DATABASE_NAME)

# save it in our converted database
db_conn = sqlite3.connect(RUN_DB_NAME)
frames_properties_df.to_sql(name='frame_properties', con=db_conn, if_exists='replace', index=False)
db_conn.close()

# load the isolation windows table
print("loading the isolation windows")
isolation_windows_df = load_isolation_windows(RAW_DATABASE_NAME, args.small_set_mode)
db_conn = sqlite3.connect(RUN_DB_NAME)
isolation_windows_df.to_sql(name='isolation_windows', con=db_conn, if_exists='replace', index=False)
db_conn.close()

# copy the frames
print("loading the raw points")
max_frame_id = frames_properties_df.Id.max()

if args.small_set_mode:
    max_frame_id = 10

print("max_frame_id: {}".format(max_frame_id))

for frame_lower in range(1, max_frame_id, args.number_of_frames_in_batch):
    frame_upper = frame_lower + args.number_of_frames_in_batch - 1
    print("processing frames {} to {}".format(frame_lower, frame_upper))
    points_df = load_raw_points(frame_lower=frame_lower, frame_upper=frame_upper)
    db_conn = sqlite3.connect(RUN_DB_NAME)
    points_df.to_sql(name='frames', con=db_conn, if_exists='append', index=False)
    db_conn.close()

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
