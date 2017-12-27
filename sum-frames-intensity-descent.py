import sys
import sqlite3
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import time

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482


parser = argparse.ArgumentParser(description='An intensity descent method for summing frames.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
parser.add_argument('-n','--number_of_summed_frames_required', type=int, help='The number of summed frames required.', required=True)
parser.add_argument('-sf','--base_source_frame_index', default=0, type=int, help='The frame index to start summing for this collision energy (note: this is an index, not the source frame ID).', required=False)
parser.add_argument('-fts','--frames_to_sum', type=int, default=5, help='The number of source frames to sum.', required=False)
parser.add_argument('-mf','--noise_threshold', type=int, default=2, help='Minimum number of frames a point must appear in to be processed.', required=False)
parser.add_argument('-bf','--base_summed_frame_id', type=int, default=1, help='The base frame ID of the summed frames.', required=False)
parser.add_argument('-ce','--collision_energy', type=int, default=10, help='Collision energy, in eV. Use 10 for MS1, 35 for MS2', required=False)
parser.add_argument('-sl','--scan_lower', type=int, help='The lower scan number.', required=False)
parser.add_argument('-su','--scan_upper', type=int, help='The upper scan number.', required=False)
args = parser.parse_args()

source_conn = sqlite3.connect(args.source_database_name)
dest_conn = sqlite3.connect(args.destination_database_name)
src_c = source_conn.cursor()
dest_c = dest_conn.cursor()

print("Setting up tables and indexes")

dest_c.execute('''CREATE TABLE IF NOT EXISTS frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)''')
dest_c.execute('''CREATE INDEX IF NOT EXISTS idx_frames ON frames (frame_id)''')
dest_c.execute('''DROP TABLE IF EXISTS elution_profile''')
dest_c.execute('''CREATE TABLE elution_profile (frame_id INTEGER, intensity INTEGER)''')
dest_c.execute('''DROP INDEX IF EXISTS idx_elution_profile''')
dest_c.execute('''CREATE INDEX idx_elution_profile ON elution_profile (frame_id)''')
dest_c.execute('''DROP TABLE IF EXISTS summing_info''')
dest_c.execute('''CREATE TABLE summing_info (item TEXT, value TEXT)''')

if args.scan_lower is None:
    q = src_c.execute("SELECT value FROM convert_info WHERE item=\"scan_lower\"")
    row = q.fetchone()
    args.scan_lower = int(row[0])
    print("lower scan set to {} from the data".format(args.scan_lower))

if args.scan_upper is None:
    q = src_c.execute("SELECT value FROM convert_info WHERE item=\"scan_upper\"")
    row = q.fetchone()
    args.scan_upper = int(row[0])
    print("upper scan set to {} from the data".format(args.scan_upper))

q = src_c.execute("SELECT value FROM convert_info WHERE item=\"mz_lower\"")
row = q.fetchone()
mz_lower = float(row[0])
q = src_c.execute("SELECT value FROM convert_info WHERE item=\"mz_upper\"")
row = q.fetchone()
mz_upper = float(row[0])

# Store the arguments as metadata in the database for later reference
summing_info = []
for arg in vars(args):
    summing_info.append((arg, getattr(args, arg)))

# Find the complete set of frame ids to be processed
number_of_source_frames = args.number_of_summed_frames_required*args.frames_to_sum
frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.collision_energy), source_conn)
frame_ids = tuple(frame_ids_df.values[args.base_source_frame_index:args.base_source_frame_index+number_of_source_frames,0])
print("summing {} source frames with collision energy {}".format(len(frame_ids), args.collision_energy))

start_run = time.time()
summedFrameId = args.base_summed_frame_id
# Step through the source frames and sum them
for summedFrameId in range(args.base_summed_frame_id,args.base_summed_frame_id+args.number_of_summed_frames_required):
    baseFrameIdsIndex = (summedFrameId-args.base_summed_frame_id) * args.frames_to_sum
    frameIdsToSum = frame_ids[baseFrameIdsIndex:baseFrameIdsIndex+args.frames_to_sum]
    print("Processing frames {} to create summed frame {}".format(frameIdsToSum, summedFrameId))
    frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} order by frame_id, mz, scan asc;".format(frameIdsToSum), source_conn)
    frame_v = frame_df.values

    frame_start = time.time()
    pointId = 1
    points = []
    for scan in range(args.scan_lower, args.scan_upper+1):
        scan_start_time = time.time()
        points_v = frame_v[np.where(frame_v[:,2] == scan)]
        points_to_process = len(points_v)
        while len(points_v) > 0:
            max_intensity_index = np.argmax(points_v[:,3])
            point_mz = points_v[max_intensity_index, 1]
            # print("m/z {}, intensity {}".format(point_mz, points_v[max_intensity_index, 3]))
            delta_mz = standard_deviation(point_mz) * 4.0
            # Find all the points in this point's std dev window
            nearby_point_indices = np.where((points_v[:,1] >= point_mz-delta_mz) & (points_v[:,1] <= point_mz+delta_mz))[0]
            nearby_points = points_v[nearby_point_indices]
            # How many distinct frames do the points come from?
            unique_frames = np.unique(nearby_points[:,0])
            if len(unique_frames) >= args.noise_threshold:
                # find the total intensity and centroid m/z
                centroid_intensity = nearby_points[:,3].sum()
                centroid_mz = peakutils.centroid(nearby_points[:,1], nearby_points[:,3])
                points.append((summedFrameId, pointId, centroid_mz, scan, int(round(centroid_intensity)), 0))
                pointId += 1

                # mz.append(centroid_mz)
                # intensity.append(centroid_intensity)

                # Plot a scan
                # f = plt.figure()
                # ax1 = f.add_subplot(111)
                # plt.xlim(point_mz-delta_mz, point_mz+delta_mz)
                # ax1.plot(points_v[:,1], points_v[:,3], 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)
                # ax1.plot(points_v[max_intensity_index,1], points_v[max_intensity_index,3], 'x', markerfacecolor='red', markeredgecolor='red', markeredgewidth=1.0, markersize=10)
                # ax1.plot(points_v[nearby_point_indices,1], points_v[nearby_point_indices,3], '+', markerfacecolor='red', markeredgecolor='green', markeredgewidth=1.0, markersize=10)
                # ax1.plot(mz, intensity, 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
                # plt.suptitle("Points along a scan")
                # plt.title("database {}, base frame {}, scan {}, points={}".format(args.source_database_name, base_frame, scan, len(points_v)))
                # plt.xlabel('m/z')
                # plt.ylabel('intensity')
                # plt.margins(0.02)
                # plt.show()

                # remove the points we've processed
                points_v = np.delete(points_v, nearby_point_indices, 0)
            else:
                # remove this point because it doesn't have enough neighbours
                points_v = np.delete(points_v, max_intensity_index, 0)
        scan_stop_time = time.time()

    dest_c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", points)
    dest_conn.commit()
    # add the elution profile
    dest_c.executemany("INSERT INTO elution_profile VALUES (?, ?)", [(summedFrameId, sum(zip(*points)[4]))])
    dest_conn.commit()
    frame_end = time.time()
    print("{} sec for frame {}".format(frame_end-frame_start, summedFrameId))
stop_run = time.time()
print("{} seconds to process run".format(stop_run-start_run))

summing_info.append(("summed_frame_lower", args.base_summed_frame_id))
summing_info.append(("summed_frame_upper", args.base_summed_frame_id+args.number_of_summed_frames_required))
# summing_info.append(("mz_lower", mz_lower))
# summing_info.append(("mz_upper", mz_upper))

summing_info.append(("run processing time (sec)", stop_run-start_run))
summing_info.append(("processed", time.ctime()))
dest_c.executemany("INSERT INTO summing_info VALUES (?, ?)", summing_info)
dest_conn.commit()
