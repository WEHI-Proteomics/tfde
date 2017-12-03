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

DELTA_MZ = 1.003355     # mass difference between Carbon-12 and Carbon-13 isotopes, in Da

NOISE_ASSESSMENT_WIDTH = 1      # length of time in seconds to average the noise level
NOISE_ASSESSMENT_OFFSET = 1     # offset in seconds from the end of the search frames
NOISE_THRESHOLD = 20000

feature_id = 1
feature_updates = []
cluster_updates = []
noise_level_readings = []

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

def find_nearest_low_index_below_threshold(base_index, threshold, indices):
    global clusters_v

    input_values_below_threshold = (clusters_v[indices,CLUSTER_INTENSITY_SUM_IDX] - threshold) < 0
    change_indices = np.where(np.roll(input_values_below_threshold,1) != input_values_below_threshold)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
    if len(change_indices) > 0:
        distance_from_base = indices[change_indices] - base_index
        distance_from_base[distance_from_base>0] = -sys.maxint
        idx = distance_from_base.argmax()
    else:
        idx = None
    return idx

# returns the index of the indices
def find_nearest_high_index_below_threshold(base_index, threshold, indices):
    global clusters_v

    print("find_nearest_high_index_below_threshold - base index {}".format(base_index))
    input_values_below_threshold = (clusters_v[indices,CLUSTER_INTENSITY_SUM_IDX] - threshold) < 0
    print("input_values_below_threshold {}".format(input_values_below_threshold))
    change_indices = np.where(np.roll(input_values_below_threshold,1) != input_values_below_threshold)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
    print("change_indices {}".format(change_indices))
    if len(change_indices) > 0:
        distance_from_base = indices[change_indices] - base_index
        distance_from_base[distance_from_base<0] = sys.maxint
        print("distance_from_base {}".format(distance_from_base))
        idx = distance_from_base.argmin()
        print("idx {}".format(idx))
    else:
        idx = None
    return idx

# find the corresponding indices in clusters_v for a given frame_id range
def find_frame_indices(start_frame_id, end_frame_id):
    start_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == start_frame_id)[0]
    end_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == end_frame_id)[0]
    first_start_frame_index = start_frame_indices[0]  # first index of the start frame
    last_end_frame_index = end_frame_indices[len(end_frame_indices)-1]  # last index of the end frame
    return (first_start_frame_index, last_end_frame_index)

def find_feature(base_index):
    global clusters_v
    global noise_level_readings

    noise_level_1 = None
    noise_level_2 = None

    cluster = clusters_v[base_index]

    frame_id = int(cluster[CLUSTER_FRAME_ID_IDX])
    cluster_id = int(cluster[CLUSTER_ID_IDX])
    charge_state = int(cluster[CLUSTER_CHARGE_STATE_IDX])

    search_start_frame = max(frame_id-NUMBER_OF_FRAMES_TO_LOOK, int(np.min(clusters_v[:,CLUSTER_FRAME_ID_IDX])))    # make sure they are valid frame_ids
    search_end_frame = min(frame_id+NUMBER_OF_FRAMES_TO_LOOK, int(np.max(clusters_v[:,CLUSTER_FRAME_ID_IDX])))

    lower_noise_eval_frame_1 = search_start_frame - int((NOISE_ASSESSMENT_OFFSET+NOISE_ASSESSMENT_WIDTH)/NUMBER_OF_SECONDS_PER_FRAME)
    upper_noise_eval_frame_1 = search_start_frame - int(NOISE_ASSESSMENT_OFFSET/NUMBER_OF_SECONDS_PER_FRAME)
    if (lower_noise_eval_frame_1 >= int(np.min(clusters_v[:,CLUSTER_FRAME_ID_IDX]))):
        # assess the noise level in this window
        (lower_noise_frame_1_index, upper_noise_frame_1_index) = find_frame_indices(lower_noise_eval_frame_1, upper_noise_eval_frame_1)
        noise_level_1 = int(np.average(clusters_v[lower_noise_frame_1_index:upper_noise_frame_1_index,CLUSTER_INTENSITY_SUM_IDX]))
        noise_level_readings.append(noise_level_1)

    lower_noise_eval_frame_2 = search_end_frame + int((NOISE_ASSESSMENT_OFFSET)/NUMBER_OF_SECONDS_PER_FRAME)
    upper_noise_eval_frame_2 = search_end_frame + int(NOISE_ASSESSMENT_OFFSET+NOISE_ASSESSMENT_WIDTH/NUMBER_OF_SECONDS_PER_FRAME)
    if (lower_noise_eval_frame_2 <= int(np.max(clusters_v[:,CLUSTER_FRAME_ID_IDX]))):
        # assess the noise level in this window
        (lower_noise_frame_2_index, upper_noise_frame_2_index) = find_frame_indices(lower_noise_eval_frame_2, upper_noise_eval_frame_2)
        noise_level_2 = int(np.average(clusters_v[lower_noise_frame_2_index:upper_noise_frame_2_index,CLUSTER_INTENSITY_SUM_IDX]))
        noise_level_readings.append(noise_level_2)

    # Seed the search bounds by the properties of the base peaks
    base_max_point_mz = cluster[CLUSTER_BASE_MAX_POINT_MZ_IDX]
    base_max_point_scan = cluster[CLUSTER_BASE_MAX_POINT_SCAN_IDX]
    base_mz_std_dev_offset = standard_deviation(base_max_point_mz) * args.mz_std_dev
    base_scan_std_dev_offset = cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX] * args.scan_std_dev

    lower_isotope_mz = base_max_point_mz - (DELTA_MZ/charge_state)
    upper_isotope_mz = base_max_point_mz + (DELTA_MZ/charge_state)

    # look for other clusters that belong to this feature
    nearby_indices = np.where(
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] >= search_start_frame) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] <= search_end_frame) &
        (clusters_v[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - base_max_point_mz) <= base_mz_std_dev_offset) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]

    # look for other clusters one isotope number down that belong to this feature
    nearby_lower_indices = np.where(
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] >= search_start_frame) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] <= search_end_frame) &
        (clusters_v[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - lower_isotope_mz) <= base_mz_std_dev_offset) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]

    # look for other clusters one isotope number up that belong to this feature
    nearby_upper_indices = np.where(
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] >= search_start_frame) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] <= search_end_frame) &
        (clusters_v[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - upper_isotope_mz) <= base_mz_std_dev_offset) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]

    # join them all together. From https://stackoverflow.com/questions/12427146/combine-two-arrays-and-sort
    c = np.concatenate((nearby_indices, nearby_lower_indices, nearby_upper_indices))
    c.sort(kind='mergesort')
    flag = np.ones(len(c), dtype=bool)
    np.not_equal(c[1:], c[:-1], out=flag[1:])
    feature_indices = c[flag]

    # make sure we don't have more than one cluster from each frame - take the most intense one if there is more than one
    frame_ids = clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX]
    if len(np.unique(frame_ids)) > 1:
        frame_change_indices = np.where(np.roll(frame_ids,1) != frame_ids)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
        feature_indices = feature_indices[frame_change_indices]
    else:
        feature_indices = feature_indices[0]

    # check whether we picked up any remnants of previous features - discard this one if so
    truncated_feature_remnants = len(np.where(clusters_v[feature_indices, CLUSTER_INTENSITY_SUM_IDX] == -1)[0])
    if truncated_feature_remnants == 0:
        quality = 1.0
    else:
        quality = 0.0

    # snip the feature where it crosses the threshold
    # indices_to_delete = np.empty(0)
    # if len(feature_indices) > 0:
    #     lower_idx = find_nearest_low_index_below_threshold(base_index, NOISE_THRESHOLD, feature_indices)
    #     upper_idx = find_nearest_high_index_below_threshold(base_index, NOISE_THRESHOLD, feature_indices)
    #     print("feature indices {}, lower idx {}, upper idx {}".format(feature_indices, lower_idx, upper_idx))
    #     if lower_idx is not None:
    #         indices_to_delete = np.concatenate((indices_to_delete,np.arange(lower_idx)))
    #     if upper_idx is not None:
    #         indices_to_delete = np.concatenate((indices_to_delete,np.arange(upper_idx+1,len(feature_indices))))
    #     feature_indices = np.delete(feature_indices, indices_to_delete, 0)

    # package the result
    results = {}
    results['base_index'] = base_index
    results['base_cluster_frame_id'] = frame_id
    results['base_cluster_id'] = cluster_id
    results['search_start_frame'] = search_start_frame
    results['search_end_frame'] = search_end_frame
    results['cluster_indices'] = feature_indices
    results['charge_state'] = charge_state
    results['noise_readings'] = (noise_level_1, noise_level_2)
    results['quality'] = quality
    return results


parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
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

print("Resetting the feature IDs in the cluster table.")
c.execute("update clusters set feature_id=0 where feature_id!=0;")

print("Loading the clusters information")
c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum from clusters order by frame_id, cluster_id asc;")
clusters_v = np.array(c.fetchall(), dtype=np.float32)

print("clusters array occupies {} bytes".format(clusters_v.nbytes))
print("Finding features")

start_run = time.time()

# go through each cluster and see whether it belongs to an existing feature
while feature_id <= args.number_of_features:
    # find the most intense cluster
    cluster_max_index = np.argmax(clusters_v[:,CLUSTER_INTENSITY_SUM_IDX])
    cluster = clusters_v[cluster_max_index]
    cluster_intensity = int(cluster[CLUSTER_INTENSITY_SUM_IDX])

    feature = find_feature(base_index=cluster_max_index)
    base_cluster_frame_id = feature['base_cluster_frame_id']
    base_cluster_id = feature['base_cluster_id']
    search_start_frame = feature['search_start_frame']
    search_end_frame = feature['search_end_frame']
    cluster_indices = feature['cluster_indices']
    charge_state = feature['charge_state']
    noise_readings = feature['noise_readings']
    quality = feature['quality']

    base_noise_level = int(np.average(noise_level_readings))

    if quality > 0.5:
        print("feature {}, search frames {}-{}, intensity {}, length {}, noise readings {}, base noise level {}".format(feature_id, search_start_frame, search_end_frame, cluster_intensity, len(cluster_indices), noise_readings, base_noise_level))
        # Assign this feature ID to all the clusters in the feature
        for cluster_idx in cluster_indices:
            values = (feature_id, int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), int(clusters_v[cluster_idx][CLUSTER_ID_IDX]))
            cluster_updates.append(values)

        # Add the feature's details to the collection
        values = (feature_id, base_cluster_frame_id, base_cluster_id, charge_state, search_start_frame, search_end_frame, quality)
        feature_updates.append(values)

        feature_id += 1

    # remove the features we've processed from the run
    clusters_v[cluster_indices, CLUSTER_INTENSITY_SUM_IDX] = -1
    if cluster_intensity < base_noise_level:
        break

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
