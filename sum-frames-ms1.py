import sys
import pymysql
import pandas as pd
import argparse
import numpy as np
import peakutils
import time

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482


parser = argparse.ArgumentParser(description='An intensity descent method for summing frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fts','--frames_to_sum', type=int, default=5, help='The number of source frames to sum.', required=False)
parser.add_argument('-mf','--noise_threshold', type=int, default=2, help='Minimum number of frames a point must appear in to be processed.', required=False)
parser.add_argument('-ce','--collision_energy', type=int, help='Collision energy, in eV.', required=True)
args = parser.parse_args()

# Connect to the database
source_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database="{}".format(args.database_name))
src_c = source_conn.cursor()

print("Setting up tables and indexes")

src_c.execute("CREATE OR REPLACE TABLE summed_frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)")
src_c.execute("CREATE OR REPLACE TABLE summing_info (item TEXT, value TEXT)")
src_c.execute("CREATE OR REPLACE TABLE elution_profile (frame_id INTEGER, intensity INTEGER)")

src_c.execute("SELECT value from convert_info where item=\"num_scans\"")
row = src_c.fetchone()
scan_lower = 0
scan_upper = int(row[0])
print("scan range {}-{}".format(scan_lower, scan_upper))

# Store the arguments as metadata in the database for later reference
summing_info = []
for arg in vars(args):
    summing_info.append((arg, getattr(args, arg)))

# Find the complete set of frame ids to be processed
frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.collision_energy), source_conn)
frame_ids = tuple(frame_ids_df.values[:,0])
print("summing {} source frames with collision energy {}".format(len(frame_ids), args.collision_energy))

start_run = time.time()
number_of_summed_frames_required = len(frame_ids)/args.frames_to_sum
# Step through the source frames and sum them
for summedFrameId in range(1,number_of_summed_frames_required+1):
    baseFrameIdsIndex = (summedFrameId-1) * args.frames_to_sum
    frameIdsToSum = frame_ids[baseFrameIdsIndex:baseFrameIdsIndex+args.frames_to_sum]
    print("Processing frames {} to create summed frame {}".format(frameIdsToSum, summedFrameId))
    frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} order by frame_id, mz, scan asc;".format(frameIdsToSum), source_conn)
    frame_v = frame_df.values

    frame_start = time.time()
    pointId = 1
    points = []
    for scan in range(scan_lower, scan_upper+1):
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
                points.append((int(summedFrameId), int(pointId), float(centroid_mz), int(scan), int(round(centroid_intensity)), 0))
                pointId += 1

            # remove the points we've processed
            points_v = np.delete(points_v, nearby_point_indices, 0)

    src_c.executemany("INSERT INTO summed_frames VALUES (%s, %s, %s, %s, %s, %s)", points)
    source_conn.commit()

    # add the elution profile
    src_c.executemany("INSERT INTO elution_profile VALUES (%s, %s)", [(summedFrameId, sum(zip(*points)[4]))])
    source_conn.commit()

    frame_end = time.time()
    print("{} sec for frame {}".format(frame_end-frame_start, summedFrameId))

stop_run = time.time()
print("{} seconds to process run".format(stop_run-start_run))

summing_info.append(("run processing time (sec)", stop_run-start_run))
summing_info.append(("processed", time.ctime()))
src_c.executemany("INSERT INTO summing_info VALUES (%s, %s)", summing_info)
source_conn.commit()
source_conn.close()
