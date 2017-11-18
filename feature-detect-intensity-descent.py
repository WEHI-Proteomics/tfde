import sys
import numpy as np
import pandas as pd
import math
import time
import sqlite3
import copy
import argparse
import os.path

# cluster array indices
CLUSTER_FRAME_ID_IDX = 0
CLUSTER_ID_IDX = 1
CLUSTER_CHARGE_STATE_IDX = 2
CLUSTER_BASE_SCAN_STD_DEV_IDX = 3
CLUSTER_BASE_MAX_POINT_MZ_IDX = 4
CLUSTER_BASE_MAX_POINT_SCAN_IDX = 5
CLUSTER_INTENSITY_SUM_IDX = 6

NUMBER_OF_SECONDS_PER_FRAME = 0.1

feature_id = 1
feature_updates = []
cluster_updates = []


def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

def find_feature(base_index):
    global clusters_v

    cluster = clusters_v[base_index]

    frame_id = int(cluster[CLUSTER_FRAME_ID_IDX])
    cluster_id = int(cluster[CLUSTER_ID_IDX])
    charge_state = int(cluster[CLUSTER_CHARGE_STATE_IDX])

    search_start_frame = max(frame_id-NUMBER_OF_FRAMES_TO_LOOK, int(np.min(clusters_v[:,CLUSTER_FRAME_ID_IDX])))
    search_end_frame = min(frame_id+NUMBER_OF_FRAMES_TO_LOOK, int(np.max(clusters_v[:,CLUSTER_FRAME_ID_IDX])))

    # Seed the search bounds by the properties of the base peaks
    base_max_point_mz = cluster[CLUSTER_BASE_MAX_POINT_MZ_IDX]
    base_max_point_scan = cluster[CLUSTER_BASE_MAX_POINT_SCAN_IDX]
    base_mz_std_dev_offset = standard_deviation(base_max_point_mz) * args.mz_std_dev
    base_scan_std_dev_offset = cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX] * args.scan_std_dev

    start_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == search_start_frame)[0]
    end_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == search_end_frame)[0]
    first_start_frame_index = start_frame_indices[0]  # first index of the start frame
    last_end_frame_index = end_frame_indices[len(end_frame_indices)-1]  # last index of the end frame

    # look for other clusters that belong to this feature
    clusters_v_subset = clusters_v[first_start_frame_index:last_end_frame_index]
    nearby_indices = np.where(
        (clusters_v_subset[:, CLUSTER_INTENSITY_SUM_IDX] >= 0) &
        (clusters_v_subset[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (abs(clusters_v_subset[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - base_max_point_mz) <= base_mz_std_dev_offset) &
        (abs(clusters_v_subset[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]
    nearby_indices_adjusted = nearby_indices+first_start_frame_index

    results = {}
    results['base_index'] = base_index
    results['base_cluster_frame_id'] = frame_id
    results['base_cluster_id'] = cluster_id
    results['search_start_frame'] = search_start_frame
    results['search_end_frame'] = search_end_frame
    results['cluster_indices'] = nearby_indices_adjusted
    results['charge_state'] = charge_state
    return results

def find_features():
    global feature_id
    global clusters_v
    global feature_updates
    global cluster_updates

    # go through each cluster and see whether it belongs to an existing feature
    while feature_id <= args.number_of_features:
        # find the most intense cluster
        cluster_max_index = np.argmax(clusters_v[:,CLUSTER_INTENSITY_SUM_IDX])
        cluster = clusters_v[cluster_max_index]

        feature = find_feature(base_index=cluster_max_index)
        base_cluster_frame_id = feature['base_cluster_frame_id']
        base_cluster_id = feature['base_cluster_id']
        search_start_frame = feature['search_start_frame']
        search_end_frame = feature['search_end_frame']
        cluster_indices = feature['cluster_indices']
        charge_state = feature['charge_state']

        if len(cluster_indices) > 10:
            print("feature {}, search frames {}-{}, intensity {}".format(feature_id, search_start_frame, search_end_frame, int(cluster[CLUSTER_INTENSITY_SUM_IDX])))
            # Assign this feature ID to all the clusters in the feature
            for cluster_idx in cluster_indices:
                values = (feature_id, int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), int(clusters_v[cluster_idx][CLUSTER_ID_IDX]))
                cluster_updates.append(values)

            # Add the feature's details to the collection
            values = (feature_id, base_cluster_frame_id, base_cluster_id, charge_state, search_start_frame, search_end_frame, 1.0)
            feature_updates.append(values)

            feature_id += 1

        # remove the features we've processed from the run
        clusters_v[cluster_indices, CLUSTER_INTENSITY_SUM_IDX] = -1

parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
parser.add_argument('-ef','--empty_frames', type=int, default=10, help='Maximum number of empty frames to tolerate.', required=False)
parser.add_argument('-nf','--number_of_features', type=int, default=50, help='Maximum number of features to find.', required=False)
parser.add_argument('-ns','--number_of_seconds', type=int, default=6, help='Number of seconds to look either side of the maximum cluster.', required=False)
args = parser.parse_args()

NUMBER_OF_FRAMES_TO_LOOK = int(args.number_of_seconds / NUMBER_OF_SECONDS_PER_FRAME)

# Store the arguments as metadata in the database for later reference
feature_info = []
for arg in vars(args):
    feature_info.append((arg, getattr(args, arg)))

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)
c = source_conn.cursor()

print("Setting up tables and indexes")

c.execute('''DROP TABLE IF EXISTS features''')
c.execute('''CREATE TABLE `features` ( `feature_id` INTEGER, 
                                        `base_frame_id` INTEGER, 
                                        `base_cluster_id` INTEGER, 
                                        `charge_state` INTEGER, 
                                        `start_frame` INTEGER, 
                                        `end_frame` INTEGER, 
                                        `quality_score` REAL, 
                                        PRIMARY KEY(`feature_id`) )''')
c.execute('''DROP INDEX IF EXISTS idx_features''')
c.execute('''CREATE INDEX idx_features ON features (feature_id)''')

c.execute('''DROP TABLE IF EXISTS feature_info''')
c.execute('''CREATE TABLE feature_info (item TEXT, value TEXT)''')

print("Loading the clusters information")

c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum from clusters order by frame_id, cluster_id asc;")
clusters_v = np.array(c.fetchall(), dtype=np.float32)

print("clusters array occupies {} bytes".format(clusters_v.nbytes))
print("Finding features")

start_run = time.time()
find_features()
stop_run = time.time()

print("updating the clusters table")
c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", cluster_updates)
print("updating the features table")
c.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?)", feature_updates)

feature_info.append(("run processing time (sec)", stop_run-start_run))
feature_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO feature_info VALUES (?, ?)", feature_info)

source_conn.commit()
source_conn.close()
