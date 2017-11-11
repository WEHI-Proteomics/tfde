import sys
import numpy as np
import pandas as pd
import math
import time
import sqlite3
import copy
import argparse
import os.path
import csv
# import numba
# from numba import jit

# @jit
def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

# @jit
def find_feature(base_index):

    cluster = clusters_v[base_index]

    frame_id = int(cluster[CLUSTER_FRAME_ID_IDX])
    charge_state = int(cluster[CLUSTER_CHARGE_STATE_IDX])

    search_start_frame = max(frame_id-NUMBER_OF_FRAMES_TO_LOOK, args.frame_lower)
    search_end_frame = min(frame_id+NUMBER_OF_FRAMES_TO_LOOK, args.frame_upper)

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
    nearby_clusters = clusters_v[nearby_indices_adjusted]
    # print("\nbase index {}, nearby_indices_adjusted {}".format(base_index, nearby_indices_adjusted.astype(int)))

    # make sure there is not more than one cluster from each frame
    frame_ids = nearby_clusters[:, CLUSTER_FRAME_ID_IDX]
    if len(np.unique(frame_ids)) > 1:
        frame_change_indices = np.where(np.roll(frame_ids,1)!=frame_ids)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
        cluster_indices = nearby_indices_adjusted[frame_change_indices]
    else:
        cluster_indices = nearby_indices_adjusted[0]

    # check the candidate feature's quality
    # tolerate up to 10% frames missing +/- 3 seconds
    # tolerate up to 20% frames missing +/- 6 seconds
    # tolerate up to 50% frames missing +/- 10 seconds
    unique_frame_ids = clusters_v[cluster_indices, CLUSTER_FRAME_ID_IDX]
    # print("clusters {}, unique frames {}".format(len(cluster_indices), len(unique_frame_ids)))
    # count the number of frames the clusters represent
    three_second_clusters_count = len(np.where(abs(unique_frame_ids-frame_id) < 3/NUMBER_OF_SECONDS_PER_FRAME)[0])
    six_second_clusters_count = len(np.where(abs(unique_frame_ids-frame_id) < 6/NUMBER_OF_SECONDS_PER_FRAME)[0]) - three_second_clusters_count
    ten_second_clusters_count = len(np.where(abs(unique_frame_ids-frame_id) < 10/NUMBER_OF_SECONDS_PER_FRAME)[0]) - three_second_clusters_count - six_second_clusters_count
    # calculate the total number of frames
    three_second_frame_count = int(2*3/NUMBER_OF_SECONDS_PER_FRAME)
    six_second_frame_count = int(2*6/NUMBER_OF_SECONDS_PER_FRAME) - three_second_frame_count
    ten_second_frame_count = int(2*10/NUMBER_OF_SECONDS_PER_FRAME) - three_second_frame_count - six_second_frame_count
    # determine the quality of the feature
    band_3s = float(three_second_clusters_count) / three_second_frame_count
    band_6s = float(six_second_clusters_count) / six_second_frame_count
    band_10s = float(ten_second_clusters_count) / ten_second_frame_count
    # print("{}/{}, {}/{}, {}/{}".format(three_second_clusters_count, three_second_frame_count, six_second_clusters_count, six_second_frame_count, ten_second_clusters_count, ten_second_frame_count))

    good_quality_feature = ((band_3s >= 0.9) and (band_6s >= 0.8) and (band_10s >= 0.5))

    results = {}
    results['base_cluster_frame_id'] = int(cluster[CLUSTER_FRAME_ID_IDX])
    results['base_cluster_id'] = int(cluster[CLUSTER_ID_IDX])
    results['search_start_frame'] = search_start_frame
    results['search_end_frame'] = search_end_frame
    results['cluster_indices'] = cluster_indices
    results['charge_state'] = charge_state
    results['quality'] = good_quality_feature
    results['band_3s'] = band_3s
    results['band_6s'] = band_6s
    results['band_10s'] = band_10s
    return results


parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number to process.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number to process.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
parser.add_argument('-ef','--empty_frames', type=int, default=10, help='Maximum number of empty frames to tolerate.', required=False)
parser.add_argument('-nf','--number_of_features', type=int, help='Maximum number of features to find.', required=False)
parser.add_argument('-ns','--number_of_seconds', type=int, default=10, help='Number of seconds to look either side of the maximum cluster.', required=False)
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
                                        `base_cluster_id` INTEGER, 
                                        `charge_state` INTEGER, 
                                        `start_frame` INTEGER, 
                                        `end_frame` INTEGER, 
                                        `band_3s` REAL, 
                                        `band_6s` REAL, 
                                        `band_10s` REAL, 
                                        PRIMARY KEY(`feature_id`) )''')
c.execute('''DROP INDEX IF EXISTS idx_features''')
c.execute('''CREATE INDEX idx_features ON features (feature_id)''')

c.execute('''DROP TABLE IF EXISTS feature_info''')
c.execute('''CREATE TABLE feature_info (item TEXT, value TEXT)''')

feature_id = 1
start_run = time.time()

# cluster array indices
CLUSTER_FRAME_ID_IDX = 0
CLUSTER_ID_IDX = 1
CLUSTER_CHARGE_STATE_IDX = 2
CLUSTER_BASE_SCAN_STD_DEV_IDX = 3
CLUSTER_BASE_MAX_POINT_MZ_IDX = 4
CLUSTER_BASE_MAX_POINT_SCAN_IDX = 5
CLUSTER_INTENSITY_SUM_IDX = 6

NUMBER_OF_SECONDS_PER_FRAME = 0.1
NUMBER_OF_FRAMES_TO_LOOK = int(args.number_of_seconds / NUMBER_OF_SECONDS_PER_FRAME)

print("Loading the clusters information")

c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum from clusters order by frame_id, cluster_id asc;")
clusters_v = np.array(c.fetchall(), dtype=np.float32)

print("clusters array occupies {} bytes".format(clusters_v.nbytes))
print("Finding features")

clusters_searched = 0

with open('discovery_rate.csv','wb') as discovery_rate_file:
    writer=csv.writer(discovery_rate_file, delimiter=',')

    # go through each cluster and see whether it belongs to an existing feature
    while len(np.where((clusters_v[:,CLUSTER_INTENSITY_SUM_IDX] > -1))[0]) > 0:
        # find the most intense cluster
        cluster_max_index = clusters_v.argmax(axis=0)[CLUSTER_INTENSITY_SUM_IDX]
        cluster = clusters_v[cluster_max_index]

        feature = find_feature(base_index=cluster_max_index)
        base_cluster_frame_id = feature['base_cluster_frame_id']
        base_cluster_id = feature['base_cluster_id']
        search_start_frame = feature['search_start_frame']
        search_end_frame = feature['search_end_frame']
        cluster_indices = feature['cluster_indices']
        charge_state = feature['charge_state']
        feature_quality = feature['quality']
        band_3s = feature['band_3s']
        band_6s = feature['band_6s']
        band_10s = feature['band_10s']
        # print("clusters remaining {}, max intensity {}, feature {}".format(len(np.where((clusters_v[:,CLUSTER_INTENSITY_SUM_IDX] > -1))[0]), int(cluster[CLUSTER_INTENSITY_SUM_IDX]), feature))

        if feature_quality:
            print("feature {}, frames searched {}-{}, 3s {:.2f}, 6s {:.2f}, 10s {:.2f}".format(feature_id, search_start_frame, search_end_frame, band_3s, band_6s, band_10s))
            # Assign this feature ID to all the clusters in the feature
            cluster_updates = []
            for cluster_idx in cluster_indices:
                values = (feature_id, int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), int(clusters_v[cluster_idx][CLUSTER_ID_IDX]))
                cluster_updates.append(values)
                # print("\tframe {}, {} m/z, {} scan".format(int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), clusters_v[cluster_idx][CLUSTER_BASE_MAX_POINT_MZ_IDX], int(clusters_v[cluster_idx][CLUSTER_BASE_MAX_POINT_SCAN_IDX])))
            c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", cluster_updates)

            # Add the feature's details to the collection
            values = (feature_id, base_cluster_id, charge_state, search_start_frame, search_end_frame, band_3s, band_6s, band_10s)
            c.execute("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)

            if args.number_of_features is not None:
                if feature_id == args.number_of_features:
                    break

            feature_id += 1

        clusters_searched += 1
        writer.writerow([clusters_searched, feature_id-1, int(cluster[CLUSTER_INTENSITY_SUM_IDX])])
        discovery_rate_file.flush()
        # remove the features we've processed from the run
        clusters_v[cluster_indices, CLUSTER_INTENSITY_SUM_IDX] = -1
        if feature_id == 263:   # temporary
            break

    stop_run = time.time()
    feature_info.append(("run processing time (sec)", stop_run-start_run))
    feature_info.append(("processed", time.ctime()))
    c.executemany("INSERT INTO feature_info VALUES (?, ?)", feature_info)

    source_conn.commit()
    source_conn.close()
