import sys
import numpy as np
import pandas as pd
import math
import time
import sqlite3
import copy
import argparse
import os.path
from scipy import signal
import sys
import matplotlib.pyplot as plt
import peakutils
from collections import deque
from operator import itemgetter

# cluster array indices
CLUSTER_FRAME_ID_IDX = 0
CLUSTER_ID_IDX = 1
CLUSTER_CHARGE_STATE_IDX = 2
CLUSTER_BASE_SCAN_STD_DEV_IDX = 3
CLUSTER_BASE_MAX_POINT_MZ_IDX = 4
CLUSTER_BASE_MAX_POINT_SCAN_IDX = 5
CLUSTER_INTENSITY_SUM_IDX = 6
CLUSTER_SCAN_LOWER_IDX = 7
CLUSTER_SCAN_UPPER_IDX = 8
CLUSTER_MZ_LOWER_IDX = 9
CLUSTER_MZ_UPPER_IDX = 10

NUMBER_OF_SECONDS_PER_FRAME = 0.1
NUMBER_OF_FRAMES_PER_SECOND = 1.0 / NUMBER_OF_SECONDS_PER_FRAME

DELTA_MZ = 1.003355     # mass difference between Carbon-12 and Carbon-13 isotopes, in Da

NOISE_ASSESSMENT_WIDTH = 1      # length of time in seconds to average the noise level
NOISE_ASSESSMENT_OFFSET = 1     # offset in seconds from the end of the feature frames

TOLERANCE_OF_POOR_QUALITY = 1000

feature_id = 1
feature_updates = []
cluster_updates = []
noise_level_readings = []
base_noise_level = 15000
feature_discovery_history = deque(maxlen=TOLERANCE_OF_POOR_QUALITY)

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

# find the corresponding indices in clusters_v for a given frame_id range
def find_frame_indices(start_frame_id, end_frame_id):
    start_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == start_frame_id)[0]
    end_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == end_frame_id)[0]
    first_start_frame_index = start_frame_indices[0]  # first index of the start frame
    last_end_frame_index = end_frame_indices[len(end_frame_indices)-1]  # last index of the end frame
    return (first_start_frame_index, last_end_frame_index)

def find_nearest_low_index_below_threshold(values, threshold):
    max_index = np.argmax(values)
    values_below_threshold = (values - threshold) < 0
    change_indices = np.where(np.roll(values_below_threshold,1) != values_below_threshold)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
    if len(change_indices) > 0:
        distance_from_base = change_indices - max_index
        distance_from_base[distance_from_base>0] = -sys.maxint
        idx = change_indices[distance_from_base.argmax()]
        if idx >= max_index:
            idx = 0
        # print("low - values {}, values_below_threshold {}, max_index {}, change_indices {}, distance_from_base {}, idx {}".format(values, values_below_threshold, max_index, change_indices, distance_from_base, idx))
    else:
        idx = None
    return idx

def find_nearest_high_index_below_threshold(values, threshold):
    max_index = np.argmax(values)
    values_below_threshold = (values - threshold) < 0
    change_indices = np.where(np.roll(values_below_threshold,1) != values_below_threshold)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
    if len(change_indices) > 0:
        distance_from_base = change_indices - max_index
        distance_from_base[distance_from_base<0] = sys.maxint
        idx = change_indices[distance_from_base.argmin()]
        if idx <= max_index:
            idx = len(values)-1
        # print("high - values {}, values_below_threshold {}, max_index {}, change_indices {}, distance_from_base {}, idx {}".format(values, values_below_threshold, max_index, change_indices, distance_from_base, idx))
    else:
        idx = None
    return idx

# returns True if the number of points in the proposed feature is more than the minimum number per second
def check_min_points_per_second(feature_indices, min_points_per_second):
    max_gap = np.max(np.diff(clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX])) * NUMBER_OF_SECONDS_PER_FRAME
    max_allowed_gap = 1.0 / min_points_per_second
    print("max gap: {} secs, max allowed gap: {} secs".format(max_gap, max_allowed_gap))
    return (max_gap < max_allowed_gap)

def find_feature(base_index):
    global clusters_v
    global noise_level_readings
    global base_noise_level

    noise_level_1 = None
    noise_level_2 = None

    cluster = clusters_v[base_index]

    frame_id = int(cluster[CLUSTER_FRAME_ID_IDX])
    cluster_id = int(cluster[CLUSTER_ID_IDX])
    charge_state = int(cluster[CLUSTER_CHARGE_STATE_IDX])

    search_start_frame = max(frame_id-NUMBER_OF_FRAMES_TO_LOOK, int(np.min(clusters_v[:,CLUSTER_FRAME_ID_IDX])))    # make sure they are valid frame_ids
    search_end_frame = min(frame_id+NUMBER_OF_FRAMES_TO_LOOK, int(np.max(clusters_v[:,CLUSTER_FRAME_ID_IDX])))

    # Seed the search bounds by the properties of the base peaks
    base_max_point_mz = cluster[CLUSTER_BASE_MAX_POINT_MZ_IDX]
    base_max_point_scan = cluster[CLUSTER_BASE_MAX_POINT_SCAN_IDX]
    base_mz_std_dev_offset = standard_deviation(base_max_point_mz) * args.mz_std_dev
    base_scan_std_dev_offset = cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX] * args.scan_std_dev

    lower_isotope_mz = base_max_point_mz - (DELTA_MZ/charge_state)
    upper_isotope_mz = base_max_point_mz + (DELTA_MZ/charge_state)

    # look for other clusters that belong to this feature
    nearby_indices = np.where(
        (clusters_v[:, CLUSTER_INTENSITY_SUM_IDX] > 0) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] >= search_start_frame) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] <= search_end_frame) &
        (clusters_v[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - base_max_point_mz) <= base_mz_std_dev_offset) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]

    # look for other clusters one isotope number down that belong to this feature
    nearby_lower_indices = np.where(
        (clusters_v[:, CLUSTER_INTENSITY_SUM_IDX] > 0) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] >= search_start_frame) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] <= search_end_frame) &
        (clusters_v[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - lower_isotope_mz) <= base_mz_std_dev_offset) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]

    # look for other clusters one isotope number up that belong to this feature
    nearby_upper_indices = np.where(
        (clusters_v[:, CLUSTER_INTENSITY_SUM_IDX] > 0) &
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

    # trim the ends to make sure we only get one feature
    if len(feature_indices) > 20:
        # snip each end where it falls below the intensity threshold
        filtered = signal.savgol_filter(clusters_v[feature_indices, CLUSTER_INTENSITY_SUM_IDX], window_length=11, polyorder=3)
        filtered_max_index = np.argmax(filtered)
        filtered_max_value = filtered[filtered_max_index]

        # f = plt.figure(figsize=(12,8))
        # ax1 = f.add_subplot(111)
        # ax1.plot(clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX], clusters_v[feature_indices, CLUSTER_INTENSITY_SUM_IDX], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        # ax1.plot(clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX], filtered, '-', markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.0, markersize=6)

        low_snip_index = None
        high_snip_index = None

        peak_maxima_indexes = peakutils.indexes(filtered, thres=0.01, min_dist=10)
        peak_minima_indexes = []
        peak_minima_indexes.append(0)
        peak_minima_indexes = peak_minima_indexes + np.where(filtered < base_noise_level)[0].tolist()
        if len(peak_maxima_indexes) > 1:
            for idx,peak_maxima_index in enumerate(peak_maxima_indexes):
                if idx>0:
                    minimum_intensity_index = np.argmin(filtered[peak_maxima_indexes[idx-1]:peak_maxima_indexes[idx]+1]) + peak_maxima_indexes[idx-1]
                    peak_minima_indexes.append(minimum_intensity_index)
        peak_minima_indexes.append(len(filtered)-1)
        peak_minima_indexes = sorted(peak_minima_indexes)

        # find the low snip index
        for idx in peak_minima_indexes:
            if (filtered[idx] < (filtered_max_value / 10.0) or (filtered[idx] < base_noise_level)) and (idx < filtered_max_index):
                low_snip_index = idx
        # find the high snip index
        for idx in reversed(peak_minima_indexes):
            if (filtered[idx] < (filtered_max_value / 10.0) or (filtered[idx] < base_noise_level)) and (idx > filtered_max_index):
                high_snip_index = idx

        # visualise what's going on
        # if low_snip_index is not None:
        #     ax1.plot(clusters_v[feature_indices[low_snip_index], CLUSTER_FRAME_ID_IDX], filtered[low_snip_index], 'x', markerfacecolor='red', markeredgecolor='red', markeredgewidth=4.0, markersize=15, alpha=1.0)
        # else:
        #     ax1.plot(clusters_v[feature_indices[0], CLUSTER_FRAME_ID_IDX], filtered[0], 'x', markerfacecolor='red', markeredgecolor='red', markeredgewidth=4.0, markersize=15, alpha=1.0)

        # if high_snip_index is not None:
        #     ax1.plot(clusters_v[feature_indices[high_snip_index], CLUSTER_FRAME_ID_IDX], filtered[high_snip_index], 'x', markerfacecolor='red', markeredgecolor='red', markeredgewidth=4.0, markersize=15, alpha=1.0)
        # else:
        #     ax1.plot(clusters_v[feature_indices[len(filtered)-1], CLUSTER_FRAME_ID_IDX], filtered[len(filtered)-1], 'x', markerfacecolor='red', markeredgecolor='red', markeredgewidth=4.0, markersize=15, alpha=1.0)
        # plt.xlabel('frame')
        # plt.ylabel('intensity')
        # plt.margins(0.02)
        # plt.show()

        indices_to_delete = np.empty(0)
        if low_snip_index is not None:
            indices_to_delete = np.concatenate((indices_to_delete,np.arange(low_snip_index)))
        if high_snip_index is not None:
            indices_to_delete = np.concatenate((indices_to_delete,np.arange(high_snip_index+1,len(filtered))))
        feature_indices = np.delete(feature_indices, indices_to_delete, 0)

    # score the feature quality
    feature_start_frame = int(clusters_v[feature_indices[0],CLUSTER_FRAME_ID_IDX])
    feature_end_frame = int(clusters_v[feature_indices[len(feature_indices)-1],CLUSTER_FRAME_ID_IDX])
    print("number of frames: {}, minimum frames {}".format(feature_end_frame-feature_start_frame, MINIMUM_NUMBER_OF_FRAMES))
    if ((feature_end_frame-feature_start_frame) >= MINIMUM_NUMBER_OF_FRAMES) and (check_min_points_per_second(feature_indices, args.minimum_points_per_second)):
        quality = 1.0

        # find the feature's intensity
        feature_summed_intensity = int(sum(clusters_v[feature_indices,CLUSTER_INTENSITY_SUM_IDX]))

        # find the feature's scan range
        feature_scan_lower = int(min(clusters_v[feature_indices,CLUSTER_SCAN_LOWER_IDX]))
        feature_scan_upper = int(max(clusters_v[feature_indices,CLUSTER_SCAN_UPPER_IDX]))

        # find the feature's m/z range
        feature_mz_lower = float(min(clusters_v[feature_indices,CLUSTER_MZ_LOWER_IDX]))
        feature_mz_upper = float(max(clusters_v[feature_indices,CLUSTER_MZ_UPPER_IDX]))

        # update the noise estimate
        lower_noise_eval_frame_1 = feature_start_frame - int((NOISE_ASSESSMENT_OFFSET+NOISE_ASSESSMENT_WIDTH)/NUMBER_OF_SECONDS_PER_FRAME)
        upper_noise_eval_frame_1 = feature_start_frame - int(NOISE_ASSESSMENT_OFFSET/NUMBER_OF_SECONDS_PER_FRAME)
        if (lower_noise_eval_frame_1 >= int(np.min(clusters_v[:,CLUSTER_FRAME_ID_IDX]))):
            # assess the noise level in this window
            (lower_noise_frame_1_index, upper_noise_frame_1_index) = find_frame_indices(lower_noise_eval_frame_1, upper_noise_eval_frame_1)
            noise_indices = np.where(clusters_v[lower_noise_frame_1_index:upper_noise_frame_1_index,CLUSTER_INTENSITY_SUM_IDX] > 0)[0]
            noise_level_1 = int(np.average(clusters_v[lower_noise_frame_1_index:upper_noise_frame_1_index,CLUSTER_INTENSITY_SUM_IDX][noise_indices]))
            noise_level_readings.append(noise_level_1)

        lower_noise_eval_frame_2 = feature_end_frame + int((NOISE_ASSESSMENT_OFFSET)/NUMBER_OF_SECONDS_PER_FRAME)
        upper_noise_eval_frame_2 = feature_end_frame + int((NOISE_ASSESSMENT_OFFSET+NOISE_ASSESSMENT_WIDTH)/NUMBER_OF_SECONDS_PER_FRAME)
        if (upper_noise_eval_frame_2 <= int(np.max(clusters_v[:,CLUSTER_FRAME_ID_IDX]))):
            # assess the noise level in this window
            (lower_noise_frame_2_index, upper_noise_frame_2_index) = find_frame_indices(lower_noise_eval_frame_2, upper_noise_eval_frame_2)
            noise_indices = np.where(clusters_v[lower_noise_frame_2_index:upper_noise_frame_2_index,CLUSTER_INTENSITY_SUM_IDX] > 0)[0]
            noise_level_2 = int(np.average(clusters_v[lower_noise_frame_2_index:upper_noise_frame_2_index,CLUSTER_INTENSITY_SUM_IDX][noise_indices]))
            noise_level_readings.append(noise_level_2)
    else:
        quality = 0.0
        feature_start_frame = None
        feature_end_frame = None
        lower_noise_eval_frame_1 = None
        upper_noise_eval_frame_1 = None
        lower_noise_eval_frame_2 = None
        upper_noise_eval_frame_2 = None
        noise_level_1 = None
        noise_level_2 = None
        feature_summed_intensity = 0
        feature_scan_lower = 0
        feature_scan_upper = 0
        feature_mz_lower = 0
        feature_mz_upper = 0

        # f = plt.figure(figsize=(12,8))
        # ax1 = f.add_subplot(111)
        # ax1.plot(clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX], clusters_v[feature_indices, CLUSTER_INTENSITY_SUM_IDX], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        # plt.title("Rejected Feature")
        # plt.xlabel('frame')
        # plt.ylabel('intensity')
        # plt.margins(0.02)
        # plt.show()

    # package the result
    results = {}
    results['base_index'] = base_index
    results['base_cluster_frame_id'] = frame_id
    results['base_cluster_id'] = cluster_id
    results['feature_frames'] = (feature_start_frame, feature_end_frame)
    results['noise_low_frames'] = (lower_noise_eval_frame_1, upper_noise_eval_frame_1)
    results['noise_high_frames'] = (lower_noise_eval_frame_2, upper_noise_eval_frame_2)
    results['cluster_indices'] = feature_indices
    results['charge_state'] = charge_state
    results['noise_readings'] = (noise_level_1, noise_level_2)
    results['quality'] = quality
    results['summed_intensity'] = feature_summed_intensity
    results['scan_range'] = (feature_scan_lower, feature_scan_upper)
    results['mz_range'] = (feature_mz_lower, feature_mz_upper)
    return results


parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
parser.add_argument('-nf','--number_of_features', type=int, help='Maximum number of features to find.', required=False)
parser.add_argument('-ns','--number_of_seconds_each_side', type=int, default=20, help='Number of seconds to look either side of the maximum cluster.', required=False)
parser.add_argument('-ml','--minimum_feature_length', type=int, default=6, help='Minimum number of seconds for a feature to be valid.', required=False)
parser.add_argument('-pps','--minimum_points_per_second', type=int, default=1, help='Minimum number of points per second for a feature to be valid.', required=False)
args = parser.parse_args()

NUMBER_OF_FRAMES_TO_LOOK = int(args.number_of_seconds_each_side / NUMBER_OF_SECONDS_PER_FRAME)
MINIMUM_NUMBER_OF_FRAMES = int(args.minimum_feature_length / NUMBER_OF_SECONDS_PER_FRAME)

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
                                        `summed_intensity` INTEGER, 
                                        `scan_lower` INTEGER, 
                                        `scan_upper` INTEGER, 
                                        `mz_lower` REAL, 
                                        `mz_upper` REAL, 
                                        PRIMARY KEY(`feature_id`) )''')
c.execute('''DROP INDEX IF EXISTS idx_features''')
c.execute('''CREATE INDEX idx_features ON features (feature_id)''')

c.execute('''DROP TABLE IF EXISTS feature_info''')
c.execute('''CREATE TABLE feature_info (item TEXT, value TEXT)''')

print("Resetting the feature IDs in the cluster table.")
c.execute("update clusters set feature_id=0 where feature_id!=0;")

print("Loading the clusters information")
c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum, scan_lower, scan_upper, mz_lower, mz_upper from clusters order by frame_id, cluster_id asc;")
clusters_v = np.array(c.fetchall(), dtype=np.float32)

print("clusters array occupies {} bytes".format(clusters_v.nbytes))
print("Finding features")

start_run = time.time()

# go through each cluster and see whether it belongs to an existing feature
while True:
    # find the most intense cluster
    cluster_max_index = np.argmax(clusters_v[:,CLUSTER_INTENSITY_SUM_IDX])
    cluster = clusters_v[cluster_max_index]
    cluster_intensity = int(cluster[CLUSTER_INTENSITY_SUM_IDX])

    feature = find_feature(base_index=cluster_max_index)
    base_cluster_frame_id = feature['base_cluster_frame_id']
    base_cluster_id = feature['base_cluster_id']
    feature_frames = feature['feature_frames']
    noise_low_frames = feature['noise_low_frames']
    noise_high_frames = feature['noise_high_frames']
    cluster_indices = feature['cluster_indices']
    charge_state = feature['charge_state']
    noise_readings = feature['noise_readings']
    quality = feature['quality']
    summed_intensity = feature['summed_intensity']
    scan_range = feature['scan_range']
    mz_range = feature['mz_range']

    base_noise_level = int(np.average(noise_level_readings))
    feature_discovery_history.append(quality)

    if quality > 0.5:
        print("feature {}, feature frames {}, intensity {}, length {}, noise low frames {}, noise high frames {}, noise readings {}, base noise level {}, scan range {}, m/z range {}".format(feature_id, feature_frames, cluster_intensity, len(cluster_indices), noise_low_frames, noise_high_frames, noise_readings, base_noise_level, scan_range, mz_range))
        # Assign this feature ID to all the clusters in the feature
        for cluster_idx in cluster_indices:
            values = (feature_id, int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), int(clusters_v[cluster_idx][CLUSTER_ID_IDX]))
            cluster_updates.append(values)

        # Add the feature's details to the collection
        values = (feature_id, base_cluster_frame_id, base_cluster_id, charge_state, feature_frames[0], feature_frames[1], quality, summed_intensity, scan_range[0], scan_range[1], mz_range[0], mz_range[1])
        feature_updates.append(values)

        feature_id += 1
    else:
        print("poor quality feature - discarding (intensity {}, base noise level {})".format(cluster_intensity, base_noise_level))

    # remove the features we've processed from the run
    clusters_v[cluster_indices, CLUSTER_INTENSITY_SUM_IDX] = -1

    # check whether we have finished
    if (cluster_intensity < base_noise_level):
        print("Reached base noise level")
        break
    if ((args.number_of_features is not None) and (feature_id > args.number_of_features)):
        print("Reached the maximum number of features")
        break
    if (feature_discovery_history.count(0.0) == TOLERANCE_OF_POOR_QUALITY):
        print("Reached maximum number of consecutive rejected features")
        break

stop_run = time.time()

print("found {} features in {} seconds".format(max(feature_updates, key=itemgetter(0))[0], stop_run-start_run))

print("updating the clusters table")
c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", cluster_updates)
print("updating the features table")
c.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", feature_updates)

feature_info.append(("run processing time (sec)", stop_run-start_run))
feature_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO feature_info VALUES (?, ?)", feature_info)

source_conn.commit()
source_conn.close()
