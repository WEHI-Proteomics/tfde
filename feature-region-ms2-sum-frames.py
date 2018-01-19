from __future__ import print_function
import sys
import sqlite3
import argparse
import numpy as np
import time
import pandas as pd
import peakutils
from operator import itemgetter
import pymysql

# feature array indices
FEATURE_ID_IDX = 0
FEATURE_START_FRAME_IDX = 1
FEATURE_END_FRAME_IDX = 2
FEATURE_SCAN_LOWER_IDX = 3
FEATURE_SCAN_UPPER_IDX = 4
FEATURE_MZ_LOWER_IDX = 5
FEATURE_MZ_UPPER_IDX = 6

# frame array indices
FRAME_ID_IDX = 0
FRAME_MZ_IDX = 1
FRAME_SCAN_IDX = 2
FRAME_INTENSITY_IDX = 3

MS1_COLLISION_ENERGY = 10
MS2_COLLISION_ENERGY = 35

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

# Find the source MS2 frame IDs corresponding to the specified summed MS1 frame ID
def ms2_frame_ids_from_ms1_frame_id(ms1_frame_id):
    # find the set of frames summed to make this MS1 frame
    lower_source_frame_index = (ms1_frame_id-1) * frames_to_sum
    upper_source_frame_index = lower_source_frame_index + frames_to_sum
    return tuple(ms2_frame_ids_v[lower_source_frame_index:upper_source_frame_index,0])


parser = argparse.ArgumentParser(description='Sum MS2 frames in the region of the MS1 feature\'s drift and retention time.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the (converted but not summed) source database, for reading MS2 frames.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
parser.add_argument('-mf','--noise_threshold', type=int, default=150, help='Minimum number of frames a point must appear in to be processed.', required=False)
parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
args = parser.parse_args()

source_conn = sqlite3.connect(args.source_database_name)
src_c = source_conn.cursor()
dest_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database='timsTOF')
dest_c = dest_conn.cursor()

# Store the arguments as metadata in the database for later reference
ms2_feature_info = []
for arg in vars(args):
    ms2_feature_info.append((arg, getattr(args, arg)))

start_run = time.time()

print("Loading the MS2 frame IDs")
ms2_frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(MS2_COLLISION_ENERGY), source_conn)
ms2_frame_ids_v = ms2_frame_ids_df.values

print("Getting some metadata about how the frames were summed")
dest_c.execute("SELECT value FROM summing_info WHERE item=\"frames_to_sum\"")
row = dest_c.fetchone()
frames_to_sum = int(row[0])
print("Number of source frames that were summed: {}".format(frames_to_sum))

print("Loading the MS1 features")
features_df = pd.read_sql_query("select feature_id,start_frame,end_frame,scan_lower,scan_upper,mz_lower,mz_upper from features where feature_id >= {} and feature_id <= {} and charge_state >= {} order by feature_id ASC;".format(args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state), dest_conn)
features_v = features_df.values

# Close the connections
dest_conn.close()

points = []
for feature in features_v:
    feature_id = int(feature[FEATURE_ID_IDX])
    feature_start_frame = int(feature[FEATURE_START_FRAME_IDX])
    feature_end_frame = int(feature[FEATURE_END_FRAME_IDX])
    feature_scan_lower = int(feature[FEATURE_SCAN_LOWER_IDX])
    feature_scan_upper = int(feature[FEATURE_SCAN_UPPER_IDX])
    feature_mz_lower = feature[FEATURE_MZ_LOWER_IDX]
    feature_mz_upper = feature[FEATURE_MZ_UPPER_IDX]

    # Load the MS2 frame points for the feature's region
    ms2_frame_ids = ()
    for frame_id in range(feature_start_frame, feature_end_frame+1):
        ms2_frame_ids += ms2_frame_ids_from_ms1_frame_id(frame_id)
    print("feature ID {}, MS1 frame IDs {}-{}, {} MS2 frames, scans {}-{}".format(feature_id, feature_start_frame, feature_end_frame, len(ms2_frame_ids), feature_scan_lower, feature_scan_upper))
    frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} and scan <= {} and scan >= {} order by frame_id, mz, scan asc;".format(ms2_frame_ids, feature_scan_upper, feature_scan_lower), source_conn)
    frame_v = frame_df.values
    print("frame occupies {} bytes".format(frame_v.nbytes))

    # Sum the points in the feature's region, just as we did for MS1 frames
    pointId = 1
    for scan in range(feature_scan_lower, feature_scan_upper+1):
        print("{},".format(scan), end="")
        points_v = frame_v[np.where(frame_v[:,FRAME_SCAN_IDX] == scan)]
        while len(points_v) > 0:
            max_intensity_index = np.argmax(points_v[:,FRAME_INTENSITY_IDX])
            point_mz = points_v[max_intensity_index, FRAME_MZ_IDX]
            # print("m/z {}, intensity {}".format(point_mz, points_v[max_intensity_index, 3]))
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
                points.append((feature_id, pointId, float(centroid_mz), scan, int(round(centroid_intensity)), len(unique_frames), 0))
                pointId += 1
            # remove the points we've processed
            points_v = np.delete(points_v, nearby_point_indices, 0)
    print("")

source_conn.close()

# Connect to the database
dest_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database='timsTOF')
dest_c = dest_conn.cursor()

# Set up the tables if they don't exist already
dest_c.execute("CREATE TABLE IF NOT EXISTS summed_ms2_regions (feature_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, number_frames INTEGER, peak_id INTEGER)")  # number_frames = number of source frames the point was found in
dest_c.execute("CREATE TABLE IF NOT EXISTS summed_ms2_regions_info (item TEXT, value TEXT)")

# Remove any existing entries for this feature range
dest_c.execute("DELETE FROM summed_ms2_regions WHERE feature_id >= {} and feature_id <= {}".format(args.feature_id_lower, args.feature_id_upper))
dest_c.execute("DELETE FROM summed_ms2_regions_info WHERE item=\"features {}-{}\"".format(args.feature_id_lower, args.feature_id_upper))

# Store the points in the database
dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, point_id, mz, scan, intensity, number_frames, peak_id) VALUES (%s, %s, %s, %s, %s, %s, %s)", points)

stop_run = time.time()
print("{} seconds to process run".format(stop_run-start_run))

ms2_feature_info.append(("ms1_feature_id_lower", min(points, key=itemgetter(0))[0]))
ms2_feature_info.append(("ms1_feature_id_upper", max(points, key=itemgetter(0))[0]))

ms2_feature_info.append(("run processing time (sec)", stop_run-start_run))
ms2_feature_info.append(("processed", time.ctime()))

ms2_feature_info_entry = []
ms2_feature_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in ms2_feature_info)))

dest_c.executemany("INSERT INTO summed_ms2_regions_info VALUES (%s, %s)", ms2_feature_info_entry)

dest_conn.commit()
dest_conn.close()
