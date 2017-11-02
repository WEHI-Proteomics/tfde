import sys
import numpy as np
import pandas as pd
import math
import time
import sqlite3
import copy
import argparse
import os.path

# Source: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy/2415343
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482



parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number to process.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number to process.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=2, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=2, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
parser.add_argument('-ef','--empty_frames', type=int, default=2, help='Maximum number of empty frames to tolerate.', required=False)
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
c.execute('''CREATE TABLE `features` ( `feature_id` INTEGER, 
                                        `charge_state` INTEGER, 
                                        `start_frame` INTEGER, 
                                        `end_frame` INTEGER, 
                                        PRIMARY KEY(`feature_id`) )''')
c.execute('''DROP INDEX IF EXISTS idx_features''')
c.execute('''CREATE INDEX idx_features ON features (feature_id)''')

c.execute("update clusters set feature_id=0")

c.execute('''DROP TABLE IF EXISTS feature_info''')
c.execute('''CREATE TABLE feature_info (item TEXT, value TEXT)''')

features = np.empty((0,11), float)
feature_id = 1
feature_cluster_mapping = []
start_run = time.time()

print("Finding features")

# cluster array indices
CLUSTER_FRAME_ID_IDX = 0
CLUSTER_ID_IDX = 1
CLUSTER_CHARGE_STATE_IDX = 2
CLUSTER_BASE_MZ_CENTROID_IDX = 3
CLUSTER_BASE_MZ_STD_DEV_IDX = 4
CLUSTER_BASE_SCAN_CENTROID_IDX = 5
CLUSTER_BASE_SCAN_STD_DEV_IDX = 6
CLUSTER_BASE_MAX_POINT_MZ_IDX = 7
CLUSTER_BASE_MAX_POINT_SCAN_IDX = 8
CLUSTER_MONO_MZ_CENTROID_IDX = 9
CLUSTER_MONO_MZ_STD_DEV_IDX = 10
CLUSTER_MONO_SCAN_CENTROID_IDX = 11
CLUSTER_MONO_SCAN_STD_DEV_IDX = 12
CLUSTER_MONO_MAX_POINT_MZ_IDX = 13
CLUSTER_MONO_MAX_POINT_SCAN_IDX = 14
CLUSTER_INTENSITY_SUM_IDX = 15

# Get all the clusters
#                                          0          1           2              3                       4                       5                         6                      7                        8                          9                      10                    11                      12                     13                       14                    15
clusters_df = pd.read_sql_query("select frame_id, cluster_id, charge_state, base_peak_mz_centroid, base_peak_mz_std_dev, base_peak_scan_centroid, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, mono_peak_mz_centroid, mono_peak_mz_std_dev, mono_peak_scan_centroid, mono_peak_scan_std_dev, mono_peak_max_point_mz, mono_peak_max_point_scan, intensity_sum from clusters order by frame_id, cluster_id asc;", source_conn)
clusters_v = clusters_df.values
print("clusters array occupies {} bytes".format(clusters_v.nbytes))

# go through each cluster and see whether it belongs to an existing feature
while len(np.where((clusters_v[:,CLUSTER_INTENSITY_SUM_IDX] > -1))[0]) > 0:
    cluster_indices = np.empty(0, dtype=int)

    # find the most intense cluster
    cluster_max_index = clusters_v.argmax(axis=0)[CLUSTER_INTENSITY_SUM_IDX]
    cluster = clusters_v[cluster_max_index]
    cluster_indices = np.append(cluster_indices, cluster_max_index)

    frame_id = int(cluster[CLUSTER_FRAME_ID_IDX])
    cluster_id = int(cluster[CLUSTER_ID_IDX])
    charge_state = int(cluster[CLUSTER_CHARGE_STATE_IDX])

    feature_start_frame = frame_id
    feature_end_frame = frame_id

    # Seed the search bounds by the properties of the base peaks
    mono_mz_centroid = cluster[CLUSTER_MONO_MZ_CENTROID_IDX]
    mono_scan_centroid = cluster[CLUSTER_MONO_SCAN_CENTROID_IDX]
    mono_max_point_mz = cluster[CLUSTER_MONO_MAX_POINT_MZ_IDX]
    mono_max_point_scan = cluster[CLUSTER_MONO_MAX_POINT_SCAN_IDX]
    mono_mz_std_dev_offset = standard_deviation(mono_max_point_mz) * args.mz_std_dev
    mono_scan_std_dev_offset = cluster[CLUSTER_MONO_SCAN_STD_DEV_IDX] * args.scan_std_dev

    base_mz_centroid = cluster[CLUSTER_BASE_MZ_CENTROID_IDX]
    base_scan_centroid = cluster[CLUSTER_BASE_SCAN_CENTROID_IDX]
    base_max_point_mz = cluster[CLUSTER_BASE_MAX_POINT_MZ_IDX]
    base_max_point_scan = cluster[CLUSTER_BASE_MAX_POINT_SCAN_IDX]
    base_mz_std_dev_offset = standard_deviation(base_max_point_mz) * args.mz_std_dev
    base_scan_std_dev_offset = cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX] * args.scan_std_dev

    # look for other clusters that belong to this feature
    # Look in the 'forward' direction
    frame_offset = 1
    missed_frames = 0
    while (missed_frames < args.empty_frames) and (frame_id+frame_offset <= args.frame_upper):
        mono_matches = np.logical_and((abs(clusters_v[:,CLUSTER_MONO_MAX_POINT_MZ_IDX] - mono_max_point_mz) <= mono_mz_std_dev_offset), 
            (abs(clusters_v[:,CLUSTER_MONO_MAX_POINT_SCAN_IDX] - mono_max_point_scan) <= mono_scan_std_dev_offset))
        base_matches = np.logical_and((abs(clusters_v[:,CLUSTER_BASE_MAX_POINT_MZ_IDX] - base_max_point_mz) <= base_mz_std_dev_offset), 
            (abs(clusters_v[:,CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))
        # cluster_matches = np.logical_or(mono_matches, base_matches)
        cluster_matches = base_matches
        charge_state_matches = (clusters_v[:,CLUSTER_CHARGE_STATE_IDX] == charge_state)
        next_frame_matches = (clusters_v[:,CLUSTER_FRAME_ID_IDX] == (frame_id+frame_offset))
        nearby_indices_forward = np.where(np.logical_and(np.logical_and(next_frame_matches, charge_state_matches), cluster_matches))[0]
        nearby_clusters_forward = clusters_v[nearby_indices_forward]
        if len(nearby_indices_forward) == 0:
            missed_frames += 1
        else:
            if len(nearby_indices_forward) > 1:
                # take the most intense cluster
                clusters_v_index_to_use = nearby_indices_forward[np.argmax(nearby_clusters_forward[:,CLUSTER_INTENSITY_SUM_IDX])]
            else:
                clusters_v_index_to_use = nearby_indices_forward[0]
            # Update the m/z window with the weighted average and standard deviation of the clusters found so far
            # mono_mz_centroid, mono_mz_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_MONO_MZ_CENTROID_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])
            # mono_scan_centroid, mono_scan_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_MONO_SCAN_CENTROID_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])

            # base_mz_centroid, base_mz_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_BASE_MZ_CENTROID_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])
            # base_scan_centroid, base_scan_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_BASE_SCAN_CENTROID_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])

            # mono_mz_std_dev_offset = mono_mz_std_dev * args.mz_std_dev
            # mono_scan_std_dev_offset = mono_scan_std_dev * args.scan_std_dev
            # base_mz_std_dev_offset = base_mz_std_dev * args.mz_std_dev
            # base_scan_std_dev_offset = base_scan_std_dev * args.scan_std_dev

            mono_max_point_mz = clusters_v[clusters_v_index_to_use,CLUSTER_MONO_MAX_POINT_MZ_IDX]
            mono_max_point_scan = clusters_v[clusters_v_index_to_use,CLUSTER_MONO_MAX_POINT_SCAN_IDX]
            base_max_point_mz = clusters_v[clusters_v_index_to_use,CLUSTER_BASE_MAX_POINT_MZ_IDX]
            base_max_point_scan = clusters_v[clusters_v_index_to_use,CLUSTER_BASE_MAX_POINT_SCAN_IDX]

            missed_frames = 0
            feature_end_frame = frame_id+frame_offset
            cluster_indices = np.append(cluster_indices, clusters_v_index_to_use)
        frame_offset += 1

    # Look in the 'backward' direction
    frame_offset = 1
    missed_frames = 0

    while (missed_frames < args.empty_frames) and (frame_id-frame_offset >= args.frame_lower):
        mono_matches = np.logical_and((abs(clusters_v[:,CLUSTER_MONO_MAX_POINT_MZ_IDX] - mono_max_point_mz) <= mono_mz_std_dev_offset), 
            (abs(clusters_v[:,CLUSTER_MONO_MAX_POINT_SCAN_IDX] - mono_max_point_scan) <= mono_scan_std_dev_offset))
        base_matches = np.logical_and((abs(clusters_v[:,CLUSTER_BASE_MAX_POINT_MZ_IDX] - base_max_point_mz) <= base_mz_std_dev_offset), 
            (abs(clusters_v[:,CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))
        # cluster_matches = np.logical_or(mono_matches, base_matches)
        cluster_matches = base_matches
        charge_state_matches = (clusters_v[:,CLUSTER_CHARGE_STATE_IDX] == charge_state)
        previous_frame_matches = (clusters_v[:,CLUSTER_FRAME_ID_IDX] == (frame_id-frame_offset))
        nearby_indices_backward = np.where(np.logical_and(np.logical_and(previous_frame_matches, charge_state_matches), cluster_matches))[0]
        nearby_clusters_backward = clusters_v[nearby_indices_backward]
        if len(nearby_indices_backward) == 0:
            missed_frames += 1
        else:
            if len(nearby_indices_backward) > 1:
                # take the most intense cluster
                clusters_v_index_to_use = nearby_indices_backward[np.argmax(nearby_clusters_backward[:,CLUSTER_INTENSITY_SUM_IDX])]
            else:
                clusters_v_index_to_use = nearby_indices_backward[0]
            # Update the m/z window with the weighted average and standard deviation of the clusters found so far
            # mono_mz_centroid, mono_mz_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_MONO_MAX_POINT_MZ_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])
            # mono_scan_centroid, mono_scan_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_MONO_MAX_POINT_SCAN_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])

            # base_mz_centroid, base_mz_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_BASE_MAX_POINT_MZ_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])
            # base_scan_centroid, base_scan_std_dev = weighted_avg_and_std(clusters_v[cluster_indices,CLUSTER_BASE_MAX_POINT_SCAN_IDX], clusters_v[cluster_indices,CLUSTER_INTENSITY_SUM_IDX])

            # mono_mz_std_dev_offset = mono_mz_std_dev * args.mz_std_dev
            # mono_scan_std_dev_offset = mono_scan_std_dev * args.scan_std_dev
            # base_mz_std_dev_offset = base_mz_std_dev * args.mz_std_dev
            # base_scan_std_dev_offset = base_scan_std_dev * args.scan_std_dev

            mono_max_point_mz = clusters_v[clusters_v_index_to_use,CLUSTER_MONO_MAX_POINT_MZ_IDX]
            mono_max_point_scan = clusters_v[clusters_v_index_to_use,CLUSTER_MONO_MAX_POINT_SCAN_IDX]
            base_max_point_mz = clusters_v[clusters_v_index_to_use,CLUSTER_BASE_MAX_POINT_MZ_IDX]
            base_max_point_scan = clusters_v[clusters_v_index_to_use,CLUSTER_BASE_MAX_POINT_SCAN_IDX]

            missed_frames = 0
            feature_start_frame = frame_id-frame_offset
            cluster_indices = np.append(cluster_indices, clusters_v_index_to_use)
        frame_offset += 1

    if feature_end_frame-feature_start_frame > 1:
        # Assign this feature ID to all the clusters in the feature
        cluster_updates = []
        for cluster_idx in cluster_indices:
            values = (feature_id, int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), int(clusters_v[cluster_idx][CLUSTER_ID_IDX]))
            cluster_updates.append(values)
        c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", cluster_updates)

        # Add the feature's details to the collection
        values = (feature_id, charge_state, feature_start_frame, feature_end_frame)
        c.execute("INSERT INTO features VALUES (?, ?, ?, ?)", values)

        feature_id += 1

    # remove the features we've processed from the run
    # clusters_v = np.delete(clusters_v, cluster_indices, 0)
    clusters_v[cluster_indices, CLUSTER_INTENSITY_SUM_IDX] = -1

stop_run = time.time()
feature_info.append(("run processing time (sec)", stop_run-start_run))
feature_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO feature_info VALUES (?, ?)", feature_info)

source_conn.commit()
source_conn.close()
