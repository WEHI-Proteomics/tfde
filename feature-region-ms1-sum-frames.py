from __future__ import print_function
import sys
import sqlite3
import argparse
import numpy as np
import time
import pandas as pd
import peakutils
from operator import itemgetter

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

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482


parser = argparse.ArgumentParser(description='Sum all MS1 frames in the region of a MS1 feature\'s m/z, drift, and retention time.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=True)
parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=True)
parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
parser.add_argument('-sd','--standard_deviations', type=int, default=8, help='Number of standard deviations in m/z to look either side of a point.', required=False)
args = parser.parse_args()

src_conn = sqlite3.connect(args.source_database_name)
src_c = src_conn.cursor()

dest_conn = sqlite3.connect(args.destination_database_name)
dest_c = dest_conn.cursor()

# Store the arguments as metadata in the database for later reference
ms1_feature_region_summing_info = []
for arg in vars(args):
    ms1_feature_region_summing_info.append((arg, getattr(args, arg)))

# Set up the tables if they don't exist already
print("Setting up tables and indexes")

dest_c.execute("DROP TABLE IF EXISTS summed_ms1_regions")
dest_c.execute("DROP TABLE IF EXISTS summed_ms1_regions_info")
dest_c.execute("CREATE TABLE summed_ms1_regions (feature_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, number_frames INTEGER, peak_id INTEGER)")  # number_frames = number of source frames the point was found in
dest_c.execute("CREATE TABLE summed_ms1_regions_info (item TEXT, value TEXT)")

src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_1 on features (feature_id, charge_state, mz_lower, mz_upper)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_2 on summed_frames (frame_id, peak_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_3 on clusters (feature_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_4 on peaks (frame_id, cluster_id)")

start_run = time.time()

# Take the ms1 features within the m/z band of interest, and sum the ms1 frames within the feature's mz/ and scan range

print("Loading the MS1 features")
features_df = pd.read_sql_query("""select feature_id,start_frame,end_frame,scan_lower,scan_upper,mz_lower,mz_upper from features where feature_id >= {} and 
    feature_id <= {} and charge_state >= {} and mz_lower <= {} and mz_upper >= {} order by feature_id ASC;""".format(args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state, args.mz_upper, args.mz_lower), src_conn)
if args.random_features_file is not None:
    # read the file of feature indexes
    random_feature_indexes_file = open(args.random_features_file, 'r')
    random_feature_indexes = list(map(int, random_feature_indexes_file.read().splitlines()))
    random_feature_indexes_file.close()
    features_df = features_df.iloc[random_feature_indexes]
features_v = features_df.values

points = []
for feature in features_v:
    feature_id = int(feature[FEATURE_ID_IDX])
    feature_start_frame = int(feature[FEATURE_START_FRAME_IDX])
    feature_end_frame = int(feature[FEATURE_END_FRAME_IDX])
    feature_scan_lower = int(feature[FEATURE_SCAN_LOWER_IDX])
    feature_scan_upper = int(feature[FEATURE_SCAN_UPPER_IDX])
    feature_mz_lower = feature[FEATURE_MZ_LOWER_IDX]
    feature_mz_upper = feature[FEATURE_MZ_UPPER_IDX]
    print("Processing feature ID {} ({} frames)".format(feature_id, feature_end_frame-feature_start_frame))

    # Load the MS1 frame (summed) points for the feature's peaks
    frame_df = pd.read_sql_query("""select frame_id,mz,scan,intensity from summed_frames where (frame_id,peak_id) in (select frame_id,peak_id from peaks where (frame_id,cluster_id) in 
        (select frame_id,cluster_id from clusters where feature_id={}));""".format(feature_id), src_conn)
    frame_v = frame_df.values
    print("frame occupies {} bytes".format(frame_v.nbytes))

    # Sum the points, just as we did for MS1 frames
    pointId = 1
    for scan in range(feature_scan_lower, feature_scan_upper+1):
        points_v = frame_v[np.where(frame_v[:,FRAME_SCAN_IDX] == scan)]
        while len(points_v) > 0:
            max_intensity_index = np.argmax(points_v[:,FRAME_INTENSITY_IDX])
            point_mz = points_v[max_intensity_index, FRAME_MZ_IDX]
            # print("m/z {}, intensity {}".format(point_mz, points_v[max_intensity_index, 3]))
            delta_mz = standard_deviation(point_mz) * args.standard_deviations
            # Find all the points in this point's std dev window
            nearby_point_indices = np.where(abs(points_v[:,FRAME_MZ_IDX] - point_mz) <= delta_mz)[0]
            nearby_points = points_v[nearby_point_indices]
            # How many distinct frames do the points come from?
            unique_frames = np.unique(nearby_points[:,FRAME_ID_IDX])
            # find the total intensity and centroid m/z
            centroid_intensity = nearby_points[:,FRAME_INTENSITY_IDX].sum()
            centroid_mz = peakutils.centroid(nearby_points[:,FRAME_MZ_IDX], nearby_points[:,FRAME_INTENSITY_IDX])
            points.append((feature_id, pointId, float(centroid_mz), scan, int(round(centroid_intensity)), len(unique_frames), 0))
            pointId += 1
            # remove the points we've processed
            points_v = np.delete(points_v, nearby_point_indices, 0)

dest_c.executemany("INSERT INTO summed_ms1_regions VALUES (?, ?, ?, ?, ?, ?, ?)", points)

stop_run = time.time()
print("{} seconds to process run".format(stop_run-start_run))

ms1_feature_region_summing_info.append(("run processing time (sec)", stop_run-start_run))
ms1_feature_region_summing_info.append(("processed", time.ctime()))

ms1_feature_region_summing_info_entry = []
ms1_feature_region_summing_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in ms1_feature_region_summing_info)))

dest_c.executemany("INSERT INTO summed_ms1_regions_info VALUES (?, ?)", ms1_feature_region_summing_info_entry)
dest_conn.commit()
dest_conn.close()
