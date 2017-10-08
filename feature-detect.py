import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import copy
import argparse
import os.path

# cluster array indices
CLUSTER_ID_IDX = 0
CLUSTER_CHARGE_STATE_IDX = 1
CLUSTER_BASE_MZ_CENTROID_IDX = 2
CLUSTER_BASE_MZ_STD_DEV_IDX = 3
CLUSTER_BASE_SCAN_CENTROID_IDX = 4
CLUSTER_BASE_SCAN_STD_DEV_IDX = 5
CLUSTER_MONO_MZ_CENTROID_IDX = 6
CLUSTER_MONO_MZ_STD_DEV_IDX = 7
CLUSTER_MONO_SCAN_CENTROID_IDX = 8
CLUSTER_MONO_SCAN_STD_DEV_IDX = 9
CLUSTER_INTENSITY_SUM_IDX = 10

# feature array indices
FEATURE_ID_IDX = 0
FEATURE_CHARGE_STATE_IDX = 1
FEATURE_BASE_MZ_CENTROID_IDX = 2
FEATURE_BASE_MZ_STD_DEV_IDX = 3
FEATURE_BASE_SCAN_CENTROID_IDX = 4
FEATURE_BASE_SCAN_STD_DEV_IDX = 5
FEATURE_MONO_MZ_CENTROID_IDX = 6
FEATURE_MONO_MZ_STD_DEV_IDX = 7
FEATURE_MONO_SCAN_CENTROID_IDX = 8
FEATURE_MONO_SCAN_STD_DEV_IDX = 9
FEATURE_LAST_FRAME_ID_IDX = 10


def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482


parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number to process.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number to process.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=2, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=2, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
feature_info = []
for arg in vars(args):
    feature_info.append((arg, getattr(args, arg)))

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)
c = source_conn.cursor()

print("Setting up tables and indexes")
c.execute('''DROP TABLE IF EXISTS feature_info''')
c.execute('''CREATE TABLE feature_info (item TEXT, value TEXT)''')

features = np.empty((0,11), float)
feature_id = 1
feature_cluster_mapping = []
start_run = time.time()
for frame_id in range(args.frame_lower, args.frame_upper+1):
    print "Processing frame {}".format(frame_id)
    # Get all the clusters for this frame
    clusters_df = pd.read_sql_query("select cluster_id,charge_state, base_peak_mz_centroid, base_peak_mz_std_dev, base_peak_scan_centroid, base_peak_scan_std_dev, mono_peak_mz_centroid, mono_peak_mz_std_dev, mono_peak_scan_centroid, mono_peak_scan_std_dev, intensity_sum from clusters where frame_id={} order by cluster_id asc;".format(frame_id), source_conn)
    clusters_v = clusters_df.values
    # go through each cluster and see whether it belongs to an existing feature
    while len(clusters_v) > 0:
        # find the most intense cluster
        cluster_max_index = clusters_v.argmax(axis=0)[CLUSTER_INTENSITY_SUM_IDX]
        cluster = clusters_v[cluster_max_index]

        cluster_id = cluster[CLUSTER_ID_IDX]
        # is this a new feature?
        mz_std_dev_offset = cluster[CLUSTER_BASE_MZ_STD_DEV_IDX] * args.mz_std_dev
        scan_std_dev_offset = cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX] * args.scan_std_dev
        feature_indices = np.where((cluster[CLUSTER_CHARGE_STATE_IDX] == features[:,FEATURE_CHARGE_STATE_IDX]) & 
                                    (abs(features[:,FEATURE_MONO_MZ_CENTROID_IDX]-cluster[CLUSTER_MONO_MZ_CENTROID_IDX]) <= mz_std_dev_offset) &
                                    (abs(features[:,FEATURE_MONO_SCAN_CENTROID_IDX]-cluster[CLUSTER_MONO_SCAN_CENTROID_IDX]) <= scan_std_dev_offset) &
                                    (features[:,FEATURE_LAST_FRAME_ID_IDX] == frame_id-1))[0]
        if (len(feature_indices) == 0):
            # add this as a new feature
            feature = np.array([[feature_id,
                                    cluster[CLUSTER_CHARGE_STATE_IDX],
                                    cluster[CLUSTER_BASE_MZ_CENTROID_IDX],
                                    cluster[CLUSTER_BASE_MZ_STD_DEV_IDX],
                                    cluster[CLUSTER_BASE_SCAN_CENTROID_IDX],
                                    cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX],
                                    cluster[CLUSTER_MONO_MZ_CENTROID_IDX],
                                    cluster[CLUSTER_MONO_MZ_STD_DEV_IDX],
                                    cluster[CLUSTER_MONO_SCAN_CENTROID_IDX],
                                    cluster[CLUSTER_MONO_SCAN_STD_DEV_IDX],
                                    frame_id]])
            features = np.append(features, feature, axis=0)
            feature_cluster_mapping.append((feature_id, frame_id, cluster_id))
            print "Added feature {} based on cluster {}:{}".format(int(feature_id), int(frame_id), int(cluster_id))
            feature_id += 1
        else:
            # always take the lowest cluster ID (i.e. the most intense match)
            f_index = feature_indices[0]
            print "Matched cluster {}:{} with feature {}".format(int(frame_id), int(cluster_id), int(features[f_index][FEATURE_ID_IDX]))
            # update the feature with this cluster's characteristics
            features[f_index][FEATURE_BASE_MZ_CENTROID_IDX] = cluster[CLUSTER_BASE_MZ_CENTROID_IDX]
            features[f_index][FEATURE_BASE_MZ_STD_DEV_IDX] = cluster[CLUSTER_BASE_MZ_STD_DEV_IDX]
            features[f_index][FEATURE_BASE_SCAN_CENTROID_IDX] = cluster[CLUSTER_BASE_SCAN_CENTROID_IDX]
            features[f_index][FEATURE_BASE_SCAN_STD_DEV_IDX] = cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX]
            features[f_index][FEATURE_MONO_MZ_CENTROID_IDX] = cluster[CLUSTER_MONO_MZ_CENTROID_IDX]
            features[f_index][FEATURE_MONO_MZ_STD_DEV_IDX] = cluster[CLUSTER_MONO_MZ_STD_DEV_IDX]
            features[f_index][FEATURE_MONO_SCAN_CENTROID_IDX] = cluster[CLUSTER_MONO_SCAN_CENTROID_IDX]
            features[f_index][FEATURE_MONO_SCAN_STD_DEV_IDX] = cluster[CLUSTER_MONO_SCAN_STD_DEV_IDX]
            features[f_index][FEATURE_LAST_FRAME_ID_IDX] = frame_id
            feature_cluster_mapping.append((features[f_index][FEATURE_ID_IDX], frame_id, cluster_id))

        # remove the cluster from the frame
        clusters_v = np.delete(clusters_v, cluster_max_index, 0)

c.execute("UPDATE clusters SET feature_id=0")
for f in feature_cluster_mapping:
    values = (f[0], f[1], f[2])
    c.execute("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", values)

stop_run = time.time()
feature_info.append(("run processing time (sec)", stop_run-start_run))
feature_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO feature_info VALUES (?, ?)", feature_info)

source_conn.commit()
source_conn.close()
