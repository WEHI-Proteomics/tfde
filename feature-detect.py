import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import copy
import argparse
import os.path


def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

def IOU(bbox1, bbox2):
    # Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I
    U = w1 * h1 + w2 * h2 - I
    return I / U


parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number to process.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number to process.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=2, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-mf','--minimum_frames', type=int, default=2, help='Minimum number of frames a feature must appear in.', required=False)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
feature_info = []
for arg in vars(args):
    feature_info.append((arg, getattr(args, arg)))

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)
c = source_conn.cursor()

print("Setting up tables and indexes")
c.execute('''DROP TABLE IF EXISTS features''')
c.execute('''CREATE TABLE features (feature_id INTEGER, charge_state INTEGER, scan_lower INTEGER, scan_upper INTEGER, intensity_sum INTEGER, base_peak_mz REAL, start_frame INTEGER, end_frame INTEGER)''')
c.execute('''DROP INDEX IF EXISTS idx_features''')
c.execute('''CREATE INDEX idx_features ON features (feature_id)''')
c.execute('''DROP TABLE IF EXISTS feature_info''')
c.execute('''CREATE TABLE feature_info (item TEXT, value TEXT)''')

features = np.empty((0,8), float)
feature_id = 1
cluster_updates = []
start_run = time.time()
for frame_id in range(args.frame_lower, args.frame_upper+1):
    print "Processing frame {}".format(frame_id)
    # Get all the clusters for this frame
    clusters_df = pd.read_sql_query("select cluster_id,charge_state,scan_upper,scan_lower,mz_upper,mz_lower,centroid_mz,centroid_scan,intensity_sum,base_peak_mz,monoisotopic_mz from clusters where frame_id={} order by cluster_id asc;".format(frame_id), source_conn)
    clusters_v = clusters_df.values
    # cluster array indices
    CLUSTER_ID_IDX = 0
    CLUSTER_CHARGE_STATE_IDX = 1
    CLUSTER_SCAN_UPPER_IDX = 2
    CLUSTER_SCAN_LOWER_IDX = 3
    CLUSTER_MZ_UPPER_IDX = 4
    CLUSTER_MZ_LOWER_IDX = 5
    CLUSTER_CENTROID_MZ_IDX = 6
    CLUSTER_CENTROID_SCAN_IDX = 7
    CLUSTER_INTENSITY_SUM_IDX = 8
    CLUSTER_BASE_PEAK_MZ_IDX = 9
    CLUSTER_MONOISOTOPIC_MZ_IDX = 10
    # feature array indices
    FEATURE_ID_IDX = 0
    FEATURE_CHARGE_STATE_IDX = 1
    FEATURE_SCAN_LOWER_IDX = 2
    FEATURE_SCAN_UPPER_IDX = 3
    FEATURE_INTENSITY_SUM_IDX = 4
    FEATURE_BASE_PEAK_MZ_IDX = 5
    FEATURE_FIRST_FRAME_IDX = 6
    FEATURE_LAST_FRAME_IDX = 7
    # go through each cluster and see whether it belongs to an existing feature
    for cluster in clusters_v:
        cluster_id = cluster[CLUSTER_ID_IDX]
        # is this a new feature?
        std_dev_offset = standard_deviation(cluster[CLUSTER_BASE_PEAK_MZ_IDX]) * args.mz_std_dev
        feature_indices = np.where((cluster[CLUSTER_CHARGE_STATE_IDX] == features[:,FEATURE_CHARGE_STATE_IDX]) & 
                                    (cluster[CLUSTER_CENTROID_SCAN_IDX] >= features[:,FEATURE_SCAN_LOWER_IDX]) & 
                                    (cluster[CLUSTER_CENTROID_SCAN_IDX] <= features[:,FEATURE_SCAN_UPPER_IDX]) & 
                                    (abs(features[:,FEATURE_BASE_PEAK_MZ_IDX]-cluster[CLUSTER_BASE_PEAK_MZ_IDX]) <= std_dev_offset) &
                                    (abs(features[:,FEATURE_INTENSITY_SUM_IDX]-cluster[CLUSTER_INTENSITY_SUM_IDX])/features[:,FEATURE_INTENSITY_SUM_IDX] < 0.1) & 
                                    (features[:,FEATURE_LAST_FRAME_IDX] == frame_id-1))[0]
        if (len(feature_indices) == 0): 
            # add this as a new feature
            feature = np.array([[feature_id,
                                    cluster[CLUSTER_CHARGE_STATE_IDX],
                                    cluster[CLUSTER_SCAN_LOWER_IDX],
                                    cluster[CLUSTER_SCAN_UPPER_IDX],
                                    cluster[CLUSTER_INTENSITY_SUM_IDX],
                                    cluster[CLUSTER_BASE_PEAK_MZ_IDX],
                                    frame_id,
                                    frame_id]])
            features = np.append(features, feature, axis=0)
            cluster_updates.append((feature_id, frame_id, cluster_id))
            print "Added feature {} based on cluster {}".format(int(feature_id), int(cluster_id))
            feature_id += 1
        elif (len(feature_indices) == 1):
            feature_index = feature_indices[0]
            f_id = features[feature_index][FEATURE_ID_IDX]
            print "Matched cluster {} with feature {}".format(int(cluster_id), int(f_id))
            # update the feature with this cluster's characteristics
            features[feature_index][FEATURE_SCAN_LOWER_IDX] = cluster[CLUSTER_SCAN_LOWER_IDX]            
            features[feature_index][FEATURE_SCAN_UPPER_IDX] = cluster[CLUSTER_SCAN_UPPER_IDX]            
            features[feature_index][FEATURE_INTENSITY_SUM_IDX] = cluster[CLUSTER_INTENSITY_SUM_IDX]            
            features[feature_index][FEATURE_BASE_PEAK_MZ_IDX] = cluster[CLUSTER_BASE_PEAK_MZ_IDX]
            features[feature_index][FEATURE_LAST_FRAME_IDX] = frame_id
            cluster_updates.append((feature_id, frame_id, cluster_id))
        else:
            print "Found {} feature matches".format(len(feature_indices))
            # take the best match
            # for i in feature_indices:

# Remove all the features that were only found in X consecutive frames
print "total number of features {}".format(len(features))
feature_indices_to_remove = np.where((features[:,FEATURE_LAST_FRAME_IDX]-features[:,FEATURE_FIRST_FRAME_IDX]) < args.minimum_frames)
features = np.delete(features, feature_indices_to_remove, 0)
print "number of features with more than {} frames: {}".format(args.minimum_frames, len(features))

stop_run = time.time()
feature_info.append(("run processing time (sec)", stop_run-start_run))
feature_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO feature_info VALUES (?, ?)", feature_info)

# Write out the features to the database
features_l = features.tolist()
c.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?)", features_l)

updates = []
for cluster_update in cluster_updates:
    if cluster_update[0] in features[:,FEATURE_ID_IDX]:
        updates.append(cluster_update)
c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", updates)

source_conn.commit()
source_conn.close()
