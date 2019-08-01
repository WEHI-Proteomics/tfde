import sys
import sqlite3
import pandas as pd
import argparse
import numpy as np
import peakutils
import time
import json
import os.path

# frames array indexes
FRAME_ID_IDX = 0
FRAME_MZ_IDX = 1
FRAME_SCAN_IDX = 2
FRAME_INTENSITY_IDX = 3
FRAME_POINT_ID_IDX = 4
FRAME_RT_IDX = 5

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482


parser = argparse.ArgumentParser(description='An intensity descent method for summing frames.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
parser.add_argument('-fts','--frames_to_sum', type=int, help='The number of MS1 source frames to sum.', required=True)
parser.add_argument('-fso','--frame_summing_offset', type=int, help='The number of MS1 source frames to shift for each summation.', required=True)
parser.add_argument('-mf','--noise_threshold', type=int, default=2, help='Minimum number of frames a point must appear in to be processed.', required=False)
parser.add_argument('-ce','--collision_energy', type=int, help='Collision energy for ms1, in eV.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number.', required=False)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number.', required=False)
parser.add_argument('-sl','--scan_lower', type=int, help='The lower scan number.', required=True)
parser.add_argument('-su','--scan_upper', type=int, help='The upper scan number.', required=True)
parser.add_argument('-bs','--batch_size', type=int, default=10000, help='The size of the frames to be written to the database.', required=False)
args = parser.parse_args()

# Connect to the databases
source_conn = sqlite3.connect(args.source_database_name)
src_c = source_conn.cursor()

# remove the destination database if it remains from a previous run - it's faster to recreate it
if os.path.isfile(args.destination_database_name):
    os.remove(args.destination_database_name)

dest_conn = sqlite3.connect(args.destination_database_name)
dest_c = dest_conn.cursor()

print("Setting up tables...")

dest_c.execute("DROP TABLE IF EXISTS summed_frames")
dest_c.execute("DROP TABLE IF EXISTS summing_info")
dest_c.execute("DROP TABLE IF EXISTS elution_profile")

dest_c.execute("CREATE TABLE summed_frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, retention_time_secs REAL, peak_id INTEGER)")
dest_c.execute("CREATE TABLE summing_info (item TEXT, value TEXT)")
dest_c.execute("CREATE TABLE elution_profile (retention_time_secs REAL, intensity INTEGER)")

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

# Find the complete set of frame ids to be processed
frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.collision_energy), source_conn)
frame_ids = tuple(frame_ids_df.values[:,0])
number_of_summed_frames = 1 + int(((len(frame_ids) - args.frames_to_sum) / args.frame_summing_offset))
print("summing {} source frames with collision energy {} to create {} summed frames".format(len(frame_ids), args.collision_energy, number_of_summed_frames))

if args.frame_lower is None:
    args.frame_lower = 1
    print("lower frame_id set to {} from the data".format(args.frame_lower))

if args.frame_upper is None:
    args.frame_upper = number_of_summed_frames
    print("upper frame_id set to {} from the data".format(args.frame_upper))

start_run = time.time()

# Step through the source frames and sum them
elution_profile = []
points = []
frame_count = 0
raw_summed_join = []
for summedFrameId in range(args.frame_lower,args.frame_upper+1):
    baseFrameIdsIndex = (summedFrameId-1)*args.frame_summing_offset
    frameIdsToSum = frame_ids[baseFrameIdsIndex:baseFrameIdsIndex+args.frames_to_sum]
    numberOfFramesToSum = len(frameIdsToSum)
    if numberOfFramesToSum == 1:
        frameIdsToSum = "({})".format(frameIdsToSum[0])
    print("Processing {} frames ({}) to create summed frame {}".format(numberOfFramesToSum, frameIdsToSum, summedFrameId))
    frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity,point_id,retention_time_secs from frames where frame_id in {} order by frame_id, mz, scan asc;".format(frameIdsToSum), source_conn)
    frame_v = frame_df.values

    frame_start = time.time()
    pointId = 1
    frame_points = []
    for scan in range(args.scan_lower, args.scan_upper+1):
        points_v = frame_v[np.where(frame_v[:,FRAME_SCAN_IDX] == scan)]
        points_to_process = len(points_v)
        while len(points_v) > 0:
            max_intensity_index = np.argmax(points_v[:,FRAME_INTENSITY_IDX])
            point_mz = points_v[max_intensity_index, FRAME_MZ_IDX]
            delta_mz = standard_deviation(point_mz) * 4.0
            # Find all the points in this point's std dev window
            nearby_point_indices = np.where((points_v[:,FRAME_MZ_IDX] >= point_mz-delta_mz) & (points_v[:,FRAME_MZ_IDX] <= point_mz+delta_mz))[0]
            nearby_points = points_v[nearby_point_indices]
            # How many distinct frames do the points come from?
            unique_frames = np.unique(nearby_points[:,FRAME_ID_IDX])
            if len(unique_frames) >= args.noise_threshold:
                # find the total intensity and centroid m/z
                centroid_intensity = nearby_points[:,FRAME_INTENSITY_IDX].sum()
                centroid_mz = peakutils.centroid(nearby_points[:,FRAME_MZ_IDX], nearby_points[:,FRAME_INTENSITY_IDX])
                centroid_rt = peakutils.centroid(nearby_points[:,FRAME_RT_IDX], nearby_points[:,FRAME_INTENSITY_IDX])
                frame_points.append((int(summedFrameId), int(pointId), float(centroid_mz), int(scan), int(round(centroid_intensity)), centroid_rt, 0))
                for p in nearby_points:
                    raw_summed_join.append((int(summedFrameId), int(pointId), int(p[FRAME_ID_IDX]), int(p[FRAME_POINT_ID_IDX])))
                pointId += 1

            # remove the points we've processed
            points_v = np.delete(points_v, nearby_point_indices, 0)

    summed_frame_rt = peakutils.centroid(frame_df.retention_time_secs, frame_df.intensity)
    if len(frame_points) > 0:
        elution_profile.append((summed_frame_rt, sum(zip(*frame_points)[4])))
    else:
        elution_profile.append((summed_frame_rt, 0))

    # add the frame's points to the set
    points += frame_points
    frame_end = time.time()
    print("{} sec for frame {} ({} points)".format(frame_end-frame_start, summedFrameId, len(frame_points)))
    del frame_points[:]

    frame_count += 1

    # check if we've processed a batch number of frames - store in database if so
    if (frame_count % args.batch_size == 0):
        print("frame count {} - writing summed frames to the database...".format(frame_count))
        dest_c.executemany("INSERT INTO summed_frames VALUES (?, ?, ?, ?, ?, ?, ?)", points)
        dest_c.executemany("INSERT INTO elution_profile VALUES (?, ?)", elution_profile)
        dest_conn.commit()
        del points[:]
        del elution_profile[:]

if len(points) > 0:
    dest_c.executemany("INSERT INTO summed_frames VALUES (?, ?, ?, ?, ?, ?, ?)", points)

if len(elution_profile) > 0:
    dest_c.executemany("INSERT INTO elution_profile VALUES (?, ?)", elution_profile)

# write out the raw-to-summed mapping
print("writing out the raw-to-summed mapping")
raw_summed_join_columns = ['summed_frame_id', 'summed_point_id', 'raw_frame_id', 'raw_point_id']
raw_summed_join_df = pd.DataFrame(raw_summed_join, columns=raw_summed_join_columns)
raw_summed_join_df['summed_frame_point'] = raw_summed_join_df['summed_frame_id'].map(str) + '|' + raw_summed_join_df['summed_point_id'].map(str)
raw_summed_join_df['raw_frame_point'] = raw_summed_join_df['raw_frame_id'].map(str) + '|' + raw_summed_join_df['raw_point_id'].map(str)
raw_summed_join_df.to_sql(name='raw_summed_join', con=dest_conn, if_exists='replace', index=False)

stop_run = time.time()

info.append(("scan_lower", args.scan_lower))
info.append(("scan_upper", args.scan_upper))
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))

print("{} info: {}".format(parser.prog, info))

info_entry = []
info_entry.append(("summed frames {}-{}".format(args.frame_lower, args.frame_upper), json.dumps(info)))

dest_c.executemany("INSERT INTO summing_info VALUES (?, ?)", info_entry)

source_conn.close()

dest_conn.commit()
dest_conn.close()
