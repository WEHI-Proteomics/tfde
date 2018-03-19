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

DELTA_MZ = 1.003355     # mass difference between Carbon-12 and Carbon-13 isotopes, in Da

NOISE_ASSESSMENT_WIDTH = 1      # length of time in seconds to average the noise level
NOISE_ASSESSMENT_OFFSET = 1     # offset in seconds from the end of the feature frames

FEATURE_DISCOVERY_HISTORY_LENGTH = 100
MAX_PROPORTION_POOR_QUALITY = 0.8   # stop looking if the proportion of poor quality features exceeds this level

COMMIT_BATCH_SIZE = 1000         # save the features to the database every COMMIT_BATCH_SIZE features
EVALUATE_NOISE_LEVEL_RATE = 500 # evaluate the base noise level every EVALUATE_NOISE_LEVEL_RATE features

feature_id = 1
feature_updates = []
cluster_updates = []
base_noise_level = 15000
noise_level_readings = [base_noise_level]
feature_discovery_history = deque(maxlen=FEATURE_DISCOVERY_HISTORY_LENGTH)

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

# find the corresponding indices in clusters_v for a given frame_id range
def find_frame_indices(start_frame_id, end_frame_id):
    start_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == start_frame_id)[0]
    end_frame_indices = np.where(clusters_v[:,CLUSTER_FRAME_ID_IDX] == end_frame_id)[0]
    if len(start_frame_indices) > 0:
        first_start_frame_index = start_frame_indices[0]  # first index of the start frame
    else:
        first_start_frame_index = None
    if len(end_frame_indices) > 0:
        last_end_frame_index = end_frame_indices[len(end_frame_indices)-1]  # last index of the end frame
    else:
        last_end_frame_index = None
    return (first_start_frame_index, last_end_frame_index)

# returns True if the gap between points is within acceptable limit
def check_gap_between_points(feature_indices, max_gap_in_seconds):
    features_max_gap_in_seconds = np.max(np.diff(clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX])) / args.frames_per_second
    # print("frame ids {}, max gap {}".format(clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX], features_max_gap_in_seconds))
    return (features_max_gap_in_seconds <= max_gap_in_seconds)

def find_feature(base_index):
    global clusters_v
    global noise_level_readings
    global base_noise_level
    global frame_lower
    global frame_upper

    noise_level_1 = None
    noise_level_2 = None

    cluster = clusters_v[base_index]

    frame_id = int(cluster[CLUSTER_FRAME_ID_IDX])
    cluster_id = int(cluster[CLUSTER_ID_IDX])
    charge_state = int(cluster[CLUSTER_CHARGE_STATE_IDX])

    search_start_frame = max(frame_id-NUMBER_OF_FRAMES_TO_LOOK, int(frame_lower))    # make sure they are valid frame_ids
    search_end_frame = min(frame_id+NUMBER_OF_FRAMES_TO_LOOK, int(frame_upper))

    # Seed the search bounds by the properties of the base peaks
    base_max_point_mz = cluster[CLUSTER_BASE_MAX_POINT_MZ_IDX]
    base_max_point_scan = cluster[CLUSTER_BASE_MAX_POINT_SCAN_IDX]
    base_mz_std_dev_offset = standard_deviation(base_max_point_mz) * args.mz_std_dev
    base_scan_std_dev_offset = cluster[CLUSTER_BASE_SCAN_STD_DEV_IDX] * args.scan_std_dev

    lower_isotope_mz = base_max_point_mz - (DELTA_MZ/charge_state)
    upper_isotope_mz = base_max_point_mz + (DELTA_MZ/charge_state)

    # look for other clusters that belong to this feature
    feature_indices = np.where(
        (clusters_v[:, CLUSTER_INTENSITY_SUM_IDX] > 0) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] >= search_start_frame) &
        (clusters_v[:, CLUSTER_FRAME_ID_IDX] <= search_end_frame) &
        (clusters_v[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (
            (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - base_max_point_mz) <= base_mz_std_dev_offset) | 
            (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - lower_isotope_mz) <= base_mz_std_dev_offset) | 
            (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - upper_isotope_mz) <= base_mz_std_dev_offset)
        ) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]

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
            if (filtered[idx] < (filtered_max_value * args.magnitude_for_feature_endpoints) or (filtered[idx] < base_noise_level)) and (idx < filtered_max_index):
                low_snip_index = idx
        # find the high snip index
        for idx in reversed(peak_minima_indexes):
            if (filtered[idx] < (filtered_max_value * args.magnitude_for_feature_endpoints) or (filtered[idx] < base_noise_level)) and (idx > filtered_max_index):
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

    passed_minimum_length_test = (feature_end_frame-feature_start_frame) >= MINIMUM_NUMBER_OF_FRAMES
    if args.maximum_gap_between_points is not None:
        passed_maximum_gap_test = check_gap_between_points(feature_indices, args.maximum_gap_between_points)
    else:
        passed_maximum_gap_test = True

    if (passed_minimum_length_test and passed_maximum_gap_test):
        quality = 1.0

        # find the feature's intensity
        feature_summed_intensity = int(sum(clusters_v[feature_indices,CLUSTER_INTENSITY_SUM_IDX]))

        # find the feature's scan range
        feature_max_intensity_idx = np.argmax(clusters_v[feature_indices,CLUSTER_INTENSITY_SUM_IDX])
        feature_max_intensity_scan = int(clusters_v[feature_indices[feature_max_intensity_idx],CLUSTER_BASE_MAX_POINT_SCAN_IDX])

        feature_scan_lower = max(int(min(clusters_v[feature_indices,CLUSTER_SCAN_LOWER_IDX])), feature_max_intensity_scan - 10)
        feature_scan_upper = min(int(max(clusters_v[feature_indices,CLUSTER_SCAN_UPPER_IDX])), feature_max_intensity_scan + 10)

        # find the feature's m/z range
        feature_mz_lower = float(min(clusters_v[feature_indices,CLUSTER_MZ_LOWER_IDX]))
        feature_mz_upper = float(max(clusters_v[feature_indices,CLUSTER_MZ_UPPER_IDX]))

        if feature_id % EVALUATE_NOISE_LEVEL_RATE == 0:
            print("evaluating noise level...")

            # update the noise estimate from the lower window
            lower_noise_eval_frame_1 = feature_start_frame - int((NOISE_ASSESSMENT_OFFSET+NOISE_ASSESSMENT_WIDTH) * args.frames_per_second)
            upper_noise_eval_frame_1 = feature_start_frame - int(NOISE_ASSESSMENT_OFFSET * args.frames_per_second)
            if (lower_noise_eval_frame_1 >= int(np.min(clusters_v[:,CLUSTER_FRAME_ID_IDX]))):
                # assess the noise level in this window
                (lower_noise_frame_1_index, upper_noise_frame_1_index) = find_frame_indices(lower_noise_eval_frame_1, upper_noise_eval_frame_1)
                if (lower_noise_frame_1_index is not None) and (upper_noise_frame_1_index is not None):
                    noise_indices = np.where(clusters_v[lower_noise_frame_1_index:upper_noise_frame_1_index,CLUSTER_INTENSITY_SUM_IDX] > 0)[0]
                    if len(noise_indices) > 0:
                        noise_level_1 = int(np.average(clusters_v[lower_noise_frame_1_index:upper_noise_frame_1_index,CLUSTER_INTENSITY_SUM_IDX][noise_indices]))
                        noise_level_readings.append(noise_level_1)

            # update the noise estimate from the upper window
            lower_noise_eval_frame_2 = feature_end_frame + int((NOISE_ASSESSMENT_OFFSET) * args.frames_per_second)
            upper_noise_eval_frame_2 = feature_end_frame + int((NOISE_ASSESSMENT_OFFSET+NOISE_ASSESSMENT_WIDTH) * args.frames_per_second)
            if (upper_noise_eval_frame_2 <= int(np.max(clusters_v[:,CLUSTER_FRAME_ID_IDX]))):
                # assess the noise level in this window
                (lower_noise_frame_2_index, upper_noise_frame_2_index) = find_frame_indices(lower_noise_eval_frame_2, upper_noise_eval_frame_2)
                if (lower_noise_frame_2_index is not None) and (upper_noise_frame_2_index is not None):
                    noise_indices = np.where(clusters_v[lower_noise_frame_2_index:upper_noise_frame_2_index,CLUSTER_INTENSITY_SUM_IDX] > 0)[0]
                    if len(noise_indices) > 0:
                        noise_level_2 = int(np.average(clusters_v[lower_noise_frame_2_index:upper_noise_frame_2_index,CLUSTER_INTENSITY_SUM_IDX][noise_indices]))
                        noise_level_readings.append(noise_level_2)

    else:
        print("length test {}, gap test {}".format(passed_minimum_length_test,passed_maximum_gap_test))
        quality = 0.0
        feature_start_frame = None
        feature_end_frame = None
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
    results['cluster_indices'] = feature_indices
    results['charge_state'] = charge_state
    results['quality'] = quality
    results['summed_intensity'] = feature_summed_intensity
    results['scan_range'] = (feature_scan_lower, feature_scan_upper)
    results['mz_range'] = (feature_mz_lower, feature_mz_upper)
    return results


def main():
    #
    # Note: this script's scope is global - it will detect features across all the frames in the database
    #

    parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
    parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
    parser.add_argument('-md','--mz_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
    parser.add_argument('-sd','--scan_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
    parser.add_argument('-ns','--number_of_seconds_each_side', type=int, default=20, help='Number of seconds to look either side of the maximum cluster.', required=False)
    parser.add_argument('-ml','--minimum_feature_length', type=float, default=3.0, help='Minimum number of seconds for a feature to be valid.', required=False)
    parser.add_argument('-gbp','--maximum_gap_between_points', type=float, help='Maximum number of seconds between points. Gap is ignored if this parameter is not set.', required=False)
    parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
    parser.add_argument('-mfe','--magnitude_for_feature_endpoints', type=float, default=0.8, help='Proportion of a feature\'s magnitude to take for its endpoints', required=False)
    parser.add_argument('-fps','--frames_per_second', type=float, default=2.0, help='Frame rate.', required=False)
    parser.add_argument('-nbf','--number_of_features', type=int, help='The number of features to find.', required=False)
    args = parser.parse_args()

    NUMBER_OF_FRAMES_TO_LOOK = int(args.number_of_seconds_each_side * args.frames_per_second)
    MINIMUM_NUMBER_OF_FRAMES = int(args.minimum_feature_length * args.frames_per_second)

    # Store the arguments as metadata in the database for later reference
    feature_info = []
    for arg in vars(args):
        feature_info.append((arg, getattr(args, arg)))

    # Connect to the database file
    source_conn = sqlite3.connect(args.database_name)
    c = source_conn.cursor()

    # find out the frame range
    c.execute("SELECT min(frame_id) FROM clusters")
    row = c.fetchone()
    frame_lower = int(row[0])

    c.execute("SELECT max(frame_id) FROM clusters")
    row = c.fetchone()
    frame_upper = int(row[0])

    print("frame range: {} to {}".format(frame_lower, frame_upper))

    print("Setting up tables...")

    c.execute("DROP TABLE IF EXISTS features")
    c.execute("DROP TABLE IF EXISTS feature_info")

    c.execute("CREATE TABLE features (feature_id INTEGER, base_frame_id INTEGER, base_cluster_id INTEGER, charge_state INTEGER, start_frame INTEGER, end_frame INTEGER, quality_score REAL, summed_intensity INTEGER, scan_lower INTEGER, scan_upper INTEGER, mz_lower REAL, mz_upper REAL, PRIMARY KEY(feature_id))")
    c.execute("CREATE TABLE feature_info (item TEXT, value TEXT)")

    print("Setting up indexes...")

    c.execute("CREATE INDEX IF NOT EXISTS idx_clusters_1 ON clusters (feature_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_clusters_2 ON clusters (frame_id, cluster_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_clusters_3 ON clusters (charge_state, frame_id, cluster_id)")

    print("Resetting the feature IDs in the cluster table.")
    c.execute("update clusters set feature_id=0 where feature_id!=0;")

    print("Loading the clusters information")
    c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum, scan_lower, scan_upper, mz_lower, mz_upper from clusters where charge_state >= {} order by frame_id, cluster_id asc;".format(args.minimum_charge_state))
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
        cluster_indices = feature['cluster_indices']
        charge_state = feature['charge_state']
        quality = feature['quality']
        summed_intensity = feature['summed_intensity']
        scan_range = feature['scan_range']
        mz_range = feature['mz_range']

        feature_discovery_history.append(quality)

        if quality > 0.5:
            print("feature {}, feature frames {}, intensity {}, length {}, scan range {}, m/z range {}".format(feature_id, feature_frames, cluster_intensity, len(cluster_indices), scan_range, mz_range))
            # Assign this feature ID to all the clusters in the feature
            for cluster_idx in cluster_indices:
                values = (feature_id, int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), int(clusters_v[cluster_idx][CLUSTER_ID_IDX]))
                cluster_updates.append(values)

            # Add the feature's details to the collection
            values = (int(feature_id), int(base_cluster_frame_id), int(base_cluster_id), int(charge_state), int(feature_frames[0]), int(feature_frames[1]), float(quality), int(summed_intensity), int(scan_range[0]), int(scan_range[1]), float(mz_range[0]), float(mz_range[1]))
            feature_updates.append(values)

            feature_id += 1
        else:
            print("poor quality feature - discarding (intensity {}, base noise level {})".format(cluster_intensity, base_noise_level))

        # remove the features we've processed from the run
        clusters_v[cluster_indices, CLUSTER_INTENSITY_SUM_IDX] = -1

        if feature_id % COMMIT_BATCH_SIZE == 0:
            print("writing features out to the database...")
            # save this in the database
            c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", cluster_updates)
            del cluster_updates[:]
            c.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", feature_updates)
            del feature_updates[:]
            source_conn.commit()

        # check whether we have finished
        if (cluster_intensity < int(np.average(noise_level_readings))):
            print("Reached base noise level")
            break
        if ((float(feature_discovery_history.count(0.0)) / FEATURE_DISCOVERY_HISTORY_LENGTH) > MAX_PROPORTION_POOR_QUALITY):
            print("Exceeded max proportion of poor quality features")
            break
        if (args.number_of_features is not None) and (feature_id > args.number_of_features):
            print("Found the specified number of features")
            break

    # Write what we have left
    if len(feature_updates) > 0:
        c.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", feature_updates)
        del feature_updates[:]

    if len(cluster_updates) > 0:
        c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", cluster_updates)
        del cluster_updates[:]

    stop_run = time.time()

    print("found {} features in {} seconds".format(feature_id-1, stop_run-start_run))

    feature_info.append(("run processing time (sec)", stop_run-start_run))
    feature_info.append(("processed", time.ctime()))
    c.executemany("INSERT INTO feature_info VALUES (?, ?)", feature_info)

    source_conn.commit()
    source_conn.close()


if __name__ == "__main__":
    main()
