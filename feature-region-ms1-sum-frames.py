from __future__ import print_function
import sys
import pymysql
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
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=True)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=True)
parser.add_argument('-mf','--noise_threshold', type=int, default=2, help='Minimum number of frames a point must appear in to be processed.', required=False)
parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
args = parser.parse_args()

source_conn = pymysql.connect(host='localhost', user='root', passwd='password', database='timsTOF')
src_c = source_conn.cursor()

print("Setting up tables and indexes")

src_c.execute("CREATE TABLE IF NOT EXISTS summed_ms1_regions (feature_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, number_frames INTEGER, peak_id INTEGER)")  # number_frames = number of source frames the point was found in
src_c.execute("CREATE TABLE IF NOT EXISTS ms1_feature_region_summing_info (item TEXT, value TEXT)")

# Remove any existing entries for this feature range
src_c.execute("DELETE FROM summed_ms1_regions WHERE feature_id >= {} and feature_id <= {}".format(args.feature_id_lower, args.feature_id_upper))
src_c.execute("DELETE FROM ms1_feature_region_summing_info WHERE item=\"features {}-{}\"".format(args.feature_id_lower, args.feature_id_upper))

# Store the arguments as metadata in the database for later reference
ms1_feature_region_summing_info = []
for arg in vars(args):
    ms1_feature_region_summing_info.append((arg, getattr(args, arg)))

start_run = time.time()

print("Loading the MS1 features")
features_df = pd.read_sql_query("select feature_id,start_frame,end_frame,scan_lower,scan_upper,mz_lower,mz_upper from features where feature_id >= {} and feature_id <= {} and charge_state >= {} order by feature_id ASC;".format(args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state), source_conn)
features_v = features_df.values

for feature in features_v:
    feature_id = int(feature[FEATURE_ID_IDX])
    feature_start_frame = int(feature[FEATURE_START_FRAME_IDX])
    feature_end_frame = int(feature[FEATURE_END_FRAME_IDX])
    feature_scan_lower = int(feature[FEATURE_SCAN_LOWER_IDX])
    feature_scan_upper = int(feature[FEATURE_SCAN_UPPER_IDX])
    feature_mz_lower = feature[FEATURE_MZ_LOWER_IDX]
    feature_mz_upper = feature[FEATURE_MZ_UPPER_IDX]
    print("Processing feature ID {} ({} frames)".format(feature_id, feature_end_frame-feature_start_frame))

    # Load the MS1 frame points for the feature's region
    frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id >= {} and frame_id <= {} and mz <= {} and mz >= {} and scan <= {} and scan >= {} order by frame_id, mz, scan asc;".format(feature_start_frame, feature_end_frame, feature_mz_upper, feature_mz_lower, feature_scan_upper, feature_scan_lower), source_conn)
    frame_v = frame_df.values
    print("frame occupies {} bytes".format(frame_v.nbytes))

    # Sum the points in the feature's region, just as we did for MS1 frames
    pointId = 1
    points = []
    for scan in range(feature_scan_lower, feature_scan_upper+1):
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
    src_c.executemany("INSERT INTO summed_ms1_regions VALUES (%s, %s, %s, %s, %s, %s, %s)", points)

print("Creating index on summed_ms1_regions")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions ON summed_ms1_regions (feature_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_2 ON summed_ms1_regions (feature_id,point_id)")

stop_run = time.time()
print("{} seconds to process run".format(stop_run-start_run))

# Keep a record of the features we actually processed
ms1_feature_region_summing_info.append(("feature_id_lower", args.feature_id_lower))
ms1_feature_region_summing_info.append(("feature_id_upper", args.feature_id_upper))

ms1_feature_region_summing_info.append(("run processing time (sec)", stop_run-start_run))
ms1_feature_region_summing_info.append(("processed", time.ctime()))

ms1_feature_region_summing_info_entry = []
ms1_feature_region_summing_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in ms1_feature_region_summing_info)))

src_c.executemany("INSERT INTO ms1_feature_region_summing_info VALUES (%s, %s)", ms1_feature_region_summing_info_entry)
source_conn.commit()
