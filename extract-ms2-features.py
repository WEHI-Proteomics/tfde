import sys
import sqlite3
import argparse
import numpy as np
import time

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

def ms2_frame_ids_from_ms1_frame_id(ms1_frame_id):
    # find the set of frames summed to make this MS1 frame
    lower_source_frame_index = ms1_frame_id / frames_to_sum
    upper_source_frame_index = lower_source_frame_index + frames_to_sum -1
    print("ms1 source frames for frame ID {}: {}".format(ms1_frame_ids_v[lower_source_frame_index:upper_source_frame_index+1,0], ms1_frame_id))
    print("corresponding ms2 frames: {}".format(ms2_frame_ids_v[lower_source_frame_index:upper_source_frame_index+1,0]))


parser = argparse.ArgumentParser(description='Extract MS2 features from MS1 features.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database (for reading MS2 frames).', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database (for reading MS1 features and writing MS2 features).', required=True)
parser.add_argument('-nf','--number_of_ms1_features', type=int, help='Maximum number of MS1 features to process.', required=False)
parser.add_argument('-mf','--noise_threshold', type=int, default=2, help='Minimum number of frames a point must appear in to be processed.', required=False)
args = parser.parse_args()

source_conn = sqlite3.connect(args.source_database_name)
dest_conn = sqlite3.connect(args.destination_database_name)
src_c = source_conn.cursor()
dest_c = dest_conn.cursor()

print("Setting up tables and indexes")

dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions")
dest_c.execute("CREATE TABLE summed_ms2_regions (ms1_feature_id INTEGER PRIMARY KEY, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)")

dest_c.execute("DROP INDEX IF EXISTS idx_summed_ms2_regions")
dest_c.execute("CREATE INDEX idx_summed_ms2_regions ON summed_ms2_regions (ms1_feature_id)")

dest_c.execute("DROP TABLE IF EXISTS ms2_feature_info")
dest_c.execute("CREATE TABLE ms2_feature_info (item TEXT, value TEXT)")

# Store the arguments as metadata in the database for later reference
ms2_feature_info = []
for arg in vars(args):
    ms2_feature_info.append((arg, getattr(args, arg)))

start_run = time.time()

print("Loading the MS1 frame IDs")
ms1_frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(MS1_COLLISION_ENERGY), source_conn)
ms1_frame_ids_v = ms1_frame_ids_df.values

print("Loading the MS2 frame IDs")
ms2_frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(MS2_COLLISION_ENERGY), source_conn)
ms2_frame_ids_v = ms2_frame_ids_df.values

print("Getting some metadata about how the frames were summed")

q = src_c.execute("SELECT value FROM summing_info WHERE item=\"frames_to_sum\"")
row = q.fetchone()
frames_to_sum = int(row[0])
print("frames to sum {}".format(frames_to_sum))


for feature in features_v:
    feature_id = feature[FEATURE_ID_IDX]
    ms2_frame_ids = ()
    for frame_id in range(feature[FEATURE_START_FRAME_IDX], feature[FEATURE_END_FRAME_IDX]+1):
        ms2_frame_ids += ms2_frame_ids_from_ms1_frame_id(frame_id)
    frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} order by frame_id, mz, scan asc;".format(ms2_frame_ids), source_conn)
    frame_v = frame_df.values

    pointId = 1
    points = []
    for scan in range(args.scan_lower, args.scan_upper+1):
        points_v = frame_v[np.where(frame_v[:,FRAME_SCAN_IDX] == scan)]
        points_to_process = len(points_v)
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
                points.append((feature_id, pointId, centroid_mz, scan, int(round(centroid_intensity)), 0))
                pointId += 1
                # remove the points we've processed
                points_v = np.delete(points_v, nearby_point_indices, 0)
            else:
                # remove this point because it doesn't have enough neighbours
                points_v = np.delete(points_v, max_intensity_index, 0)
    dest_c.executemany("INSERT INTO summed_ms2_regions VALUES (?, ?, ?, ?, ?, ?)", points)

    # check whether we have finished
    if ((args.number_of_ms1_features is not None) and (feature_id > args.number_of_features)):
        print("Reached the maximum number of features")
        break


stop_run = time.time()
print("{} seconds to process run".format(stop_run-start_run))

summing_info.append(("run processing time (sec)", stop_run-start_run))
summing_info.append(("processed", time.ctime()))
dest_c.executemany("INSERT INTO ms2_feature_info VALUES (?, ?)", ms2_feature_info)
dest_conn.commit()
