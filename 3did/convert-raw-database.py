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

# Convert a raw instrument database to an interim schema so we can set up database indexes that make the next step super-speedy
# python ./open-path/pda/convert-raw-database.py -en dwm-test -rn 190719_Hela_Ecoli_1to1_01 -rdd ./experiments/dwm-test/raw-databases/190719_Hela_Ecoli_1to1_01_Slot1-1_1_1926.d -ini ./open-path/pda/pasef-process-short-gradient.ini

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

SEGMENT_EXTENSION = 2.0

# returns a dataframe with the frame properties
def load_frame_properties(database_name):
    # get all the isolation windows
    db_conn = sqlite3.connect(database_name)
    frames_properties_df = pd.read_sql_query("select Id,Time,NumScans,NumPeaks from Frames where MsMsType==0 order by Id ASC;", db_conn)
    db_conn.close()

    print("loaded {} frame_properties from {}".format(len(frames_properties_df), database_name))
    return frames_properties_df


##############################################
parser = argparse.ArgumentParser(description='Transform the raw database to a processing schema.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-rdd','--raw_database_directory', type=str, help='The full path to the directory (i.e. the \'.d\' path) of the raw database.', required=True)
parser.add_argument('-ml','--mz_lower', type=int, default='100', help='Lower limit for m/z.', required=False)
parser.add_argument('-mu','--mz_upper', type=int, default='1700', help='Upper limit for m/z.', required=False)
parser.add_argument('-mw','--mz_width_per_segment', type=int, default=20, help='Width in Da of the m/z processing window per segment.', required=False)
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

# check the path to the raw database exists
if not os.path.exists(args.raw_database_directory):
    print("The path to the raw database doesn't exist: {}".format(args.raw_database_directory))
    sys.exit(1)

# check the run directory exists
RUN_DIR = '{}/converted-databases-3did'.format(EXPERIMENT_DIR)
if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

# check the analysis.tdf exists
RAW_DATABASE_NAME = "{}/analysis.tdf".format(args.raw_database_directory)
if not os.path.isfile(RAW_DATABASE_NAME):
    print("The raw database analysis.tdf doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)

# get the frame metadata
print("loading the frames information")
frames_properties_df = load_frame_properties(RAW_DATABASE_NAME)
number_of_rows = frames_properties_df.NumPeaks.sum()
print('there are {} total points'.format(number_of_rows))

data = np.zeros((number_of_rows,), dtype=[('frame_id', np.uint16), ('mz', np.float32), ('scan', np.uint16), ('intensity', np.uint16), ('retention_time_secs', np.float32)])

# copy the frames
print("loading the raw points")
# connect to the database with the timsTOF SDK
td = timsdata.TimsData(args.raw_database_directory)

# read the raw points in the specified frame range
base_idx = 0
for row in frames_properties_df.itertuples():
    # read the points from the scan lines
    for scan_idx,scan in enumerate(td.readScans(frame_id=row.Id, scan_begin=0, scan_end=row.NumScans)):
        index = np.array(scan[0], dtype=np.float64)
        mz_values = td.indexToMz(row.Id, index)
        intensity_values = scan[1]
        scan_number = scan_idx
        number_of_points_on_scan = len(mz_values)
        for i in range(0, number_of_points_on_scan):   # step through the readings (i.e. points) on this scan line
            mz_value = float(mz_values[i])
            intensity = int(intensity_values[i])
            data[base_idx] = (row.Id, mz_value, scan_number, intensity, row.Time)
            base_idx += 1


# # calculate the segments
# print('segmenting')
# mz_range = args.mz_upper - args.mz_lower
# NUMBER_OF_MZ_SEGMENTS = (mz_range // args.mz_width_per_segment) + (mz_range % args.mz_width_per_segment > 0)  # thanks to https://stackoverflow.com/a/23590097/1184799
# for i in range(NUMBER_OF_MZ_SEGMENTS):
#     segment_mz_lower=args.mz_lower+(i*args.mz_width_per_segment)
#     segment_mz_upper=args.mz_lower+(i*args.mz_width_per_segment)+args.mz_width_per_segment
#     segment_id=i+1
#     segment_filename = '{}/exp-{}-run-{}-segment-{}-{}-{}.pkl'.format(RUN_DIR, args.experiment_name, args.run_name, segment_id, segment_mz_lower, segment_mz_upper)
#     segment_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where mz >= {} and mz <= {}".format(segment_mz_lower, segment_mz_upper+SEGMENT_EXTENSION), con, dtype={'frame_id':np.uint16,'mz':np.float32,'scan':np.uint16,'intensity':np.uint16,'retention_time_secs':np.float32})
#     print('writing {} points to {}'.format(len(segment_df), segment_filename))
#     segment_df.to_pickle(segment_filename)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
