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
CLUSTER_RT_IDX = 11

DELTA_MZ = 1.003355     # mass difference between Carbon-12 and Carbon-13 isotopes, in Da

NOISE_ASSESSMENT_WIDTH_SECS = 1      # length of time in seconds to average the noise level
NOISE_ASSESSMENT_OFFSET_SECS = 1     # offset in seconds from the end of the feature frames

FEATURE_DISCOVERY_HISTORY_LENGTH = 100
MAX_PROPORTION_POOR_QUALITY = 0.8   # stop looking if the proportion of poor quality features exceeds this level

EVALUATE_NOISE_LEVEL_RATE = 500 # evaluate the base noise level every EVALUATE_NOISE_LEVEL_RATE features

feature_id = 1
feature_updates = []
cluster_updates = []
estimated_noise_level = 5000
noise_level_readings = []
feature_discovery_history = deque(maxlen=FEATURE_DISCOVERY_HISTORY_LENGTH)

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

# find the corresponding indices in clusters_v for a given RT range
def find_frame_indices(start_frame_rt, end_frame_rt):
    print("find_frame_indices: rt {}..{}".format(start_frame_rt, end_frame_rt))
    frame_indices = np.where((clusters_v[:,CLUSTER_RT_IDX] >= start_frame_rt) & (clusters_v[:,CLUSTER_RT_IDX] <= end_frame_rt))
    start_frame_index = np.min(frame_indices)
    end_frame_index = np.max(frame_indices)
    print("find_frame_indices: indices {}..{}".format(start_frame_index, end_frame_index))
    return (start_frame_index, end_frame_index)

# returns True if the gap between points is within acceptable limit
def check_gap_between_points(feature_indices, max_gap_in_seconds):
    features_max_gap_in_seconds = np.max(np.diff(clusters_v[feature_indices, CLUSTER_RT_IDX]))
    return (features_max_gap_in_seconds <= max_gap_in_seconds)

def find_feature(base_index):
    global noise_level_readings

    noise_level_1 = None
    noise_level_2 = None

    cluster = clusters_v[base_index]

    frame_id = int(cluster[CLUSTER_FRAME_ID_IDX])
    cluster_id = int(cluster[CLUSTER_ID_IDX])
    charge_state = int(cluster[CLUSTER_CHARGE_STATE_IDX])
    retention_time_secs = cluster[CLUSTER_RT_IDX]

    search_start_rt = retention_time_secs - args.number_of_seconds_each_side
    search_end_rt = retention_time_secs + args.number_of_seconds_each_side

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
        (clusters_v[:, CLUSTER_RT_IDX] >= search_start_rt) &
        (clusters_v[:, CLUSTER_RT_IDX] <= search_end_rt) &
        (clusters_v[:, CLUSTER_CHARGE_STATE_IDX] == charge_state) &
        (
            # look at the isotope we expect but also one up and one down, in case we missed the correct isotope
            (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - base_max_point_mz) <= base_mz_std_dev_offset) | 
            (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - lower_isotope_mz) <= base_mz_std_dev_offset) | 
            (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_MZ_IDX] - upper_isotope_mz) <= base_mz_std_dev_offset)
        ) &
        (abs(clusters_v[:, CLUSTER_BASE_MAX_POINT_SCAN_IDX] - base_max_point_scan) <= base_scan_std_dev_offset))[0]

    # make sure we don't have more than one cluster from each frame - take the most intense one if there is more than one
    frame_ids_list = clusters_v[feature_indices, CLUSTER_FRAME_ID_IDX].astype(int).tolist()
    intensities_list = clusters_v[feature_indices, CLUSTER_INTENSITY_SUM_IDX].astype(int).tolist()
    feature_indices_list = feature_indices.tolist()
    df = pd.DataFrame()
    df['frame_id'] = frame_ids_list
    df['intensity'] = intensities_list
    df['feature_index'] = feature_indices_list
    df.sort_values('intensity', ascending=False, inplace=True)
    df.drop_duplicates(subset=['frame_id'], keep='first', inplace=True)
    df.sort_values('frame_id', ascending=True, inplace=True)
    feature_indices = df.feature_index.values

    # trim the ends to make sure we only get one feature
    if len(feature_indices) > 20:
        # snip each end where it falls below the intensity threshold
        filtered = signal.savgol_filter(clusters_v[feature_indices, CLUSTER_INTENSITY_SUM_IDX], window_length=11, polyorder=3)
        filtered_max_index = np.argmax(filtered)
        filtered_max_value = filtered[filtered_max_index]

        low_snip_index = None
        high_snip_index = None

        peak_maxima_indexes = peakutils.indexes(filtered, thres=0.01, min_dist=10)
        peak_minima_indexes = []
        peak_minima_indexes.append(0)
        peak_minima_indexes = peak_minima_indexes + np.where(filtered < estimated_noise_level)[0].tolist()
        if len(peak_maxima_indexes) > 1:
            for idx,peak_maxima_index in enumerate(peak_maxima_indexes):
                if idx>0:
                    minimum_intensity_index = np.argmin(filtered[peak_maxima_indexes[idx-1]:peak_maxima_indexes[idx]+1]) + peak_maxima_indexes[idx-1]
                    peak_minima_indexes.append(minimum_intensity_index)
        peak_minima_indexes.append(len(filtered)-1)
        peak_minima_indexes = sorted(peak_minima_indexes)

        # find the low snip index
        for idx in peak_minima_indexes:
            if (filtered[idx] < (filtered_max_value * args.magnitude_for_feature_endpoints) or (filtered[idx] < estimated_noise_level)) and (idx < filtered_max_index):
                low_snip_index = idx
        # find the high snip index
        for idx in reversed(peak_minima_indexes):
            if (filtered[idx] < (filtered_max_value * args.magnitude_for_feature_endpoints) or (filtered[idx] < estimated_noise_level)) and (idx > filtered_max_index):
                high_snip_index = idx

        indices_to_delete = np.empty(0)
        if low_snip_index is not None:
            indices_to_delete = np.concatenate((indices_to_delete,np.arange(low_snip_index)))
        if high_snip_index is not None:
            indices_to_delete = np.concatenate((indices_to_delete,np.arange(high_snip_index+1,len(filtered))))
        feature_indices = np.delete(feature_indices, indices_to_delete, 0)

    # score the feature quality
    feature_start_rt = int(clusters_v[feature_indices[0],CLUSTER_RT_IDX])
    feature_end_rt = int(clusters_v[feature_indices[len(feature_indices)-1],CLUSTER_RT_IDX])

    passed_minimum_length_test = (feature_end_rt-feature_start_rt) >= args.minimum_feature_length_secs
    if args.maximum_gap_between_points is not None:
        passed_maximum_gap_test = check_gap_between_points(feature_indices, args.maximum_gap_between_points)
    else:
        passed_maximum_gap_test = True

    print("passed length test {}, passed gap test {}".format(passed_minimum_length_test, passed_maximum_gap_test))

    # estimate the base noise level
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

        if (feature_id == 1) or (feature_id % EVALUATE_NOISE_LEVEL_RATE == 0):
            print("evaluating noise level...")

            # update the noise estimate from the lower window
            lower_noise_eval_rt_1 = feature_start_rt - (NOISE_ASSESSMENT_OFFSET_SECS+NOISE_ASSESSMENT_WIDTH_SECS)
            upper_noise_eval_rt_1 = feature_start_rt - NOISE_ASSESSMENT_OFFSET_SECS
            if (lower_noise_eval_rt_1 >= int(np.min(clusters_v[:,CLUSTER_RT_IDX]))):
                # assess the noise level in this window
                (lower_noise_frame_1_index, upper_noise_frame_1_index) = find_frame_indices(lower_noise_eval_rt_1, upper_noise_eval_rt_1)
                if (lower_noise_frame_1_index is not None) and (upper_noise_frame_1_index is not None):
                    noise_indices = np.where(clusters_v[lower_noise_frame_1_index:upper_noise_frame_1_index,CLUSTER_INTENSITY_SUM_IDX] > 0)[0]
                    if len(noise_indices) > 0:
                        noise_level_1 = int(np.average(clusters_v[lower_noise_frame_1_index:upper_noise_frame_1_index,CLUSTER_INTENSITY_SUM_IDX][noise_indices]))
                        noise_level_readings.append(noise_level_1)

            # update the noise estimate from the upper window
            lower_noise_eval_rt_2 = feature_end_rt + NOISE_ASSESSMENT_OFFSET_SECS
            upper_noise_eval_rt_2 = feature_end_rt + (NOISE_ASSESSMENT_OFFSET_SECS+NOISE_ASSESSMENT_WIDTH_SECS)
            if (upper_noise_eval_rt_2 <= int(np.max(clusters_v[:,CLUSTER_RT_IDX]))):
                # assess the noise level in this window
                (lower_noise_frame_2_index, upper_noise_frame_2_index) = find_frame_indices(lower_noise_eval_rt_2, upper_noise_eval_rt_2)
                if (lower_noise_frame_2_index is not None) and (upper_noise_frame_2_index is not None):
                    noise_indices = np.where(clusters_v[lower_noise_frame_2_index:upper_noise_frame_2_index,CLUSTER_INTENSITY_SUM_IDX] > 0)[0]
                    if len(noise_indices) > 0:
                        noise_level_2 = int(np.average(clusters_v[lower_noise_frame_2_index:upper_noise_frame_2_index,CLUSTER_INTENSITY_SUM_IDX][noise_indices]))
                        noise_level_readings.append(noise_level_2)

    else:
        quality = 0.0
        feature_start_rt = None
        feature_end_rt = None
        feature_summed_intensity = 0
        feature_scan_lower = 0
        feature_scan_upper = 0
        feature_mz_lower = 0
        feature_mz_upper = 0

    # package the result
    results = {}
    results['base_index'] = base_index
    results['base_cluster_frame_id'] = frame_id
    results['base_cluster_id'] = cluster_id
    results['feature_rt_range'] = (feature_start_rt, feature_end_rt)
    results['cluster_indices'] = feature_indices
    results['charge_state'] = charge_state
    results['quality'] = quality
    results['summed_intensity'] = feature_summed_intensity
    results['scan_range'] = (feature_scan_lower, feature_scan_upper)
    results['mz_range'] = (feature_mz_lower, feature_mz_upper)
    return results


#
# Note: this script's scope is global - it will detect features across all the frames in the database
#

parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-md','--mz_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the m/z dimension.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=4, help='Number of standard deviations to look either side of the base peak, in the scan dimension.', required=False)
parser.add_argument('-ns','--number_of_seconds_each_side', type=int, default=20, help='Number of seconds to look either side of the maximum cluster.', required=False)
parser.add_argument('-mfl','--minimum_feature_length_secs', type=int, default=1, help='Minimum feature length in seconds for it to be valid.', required=False)
parser.add_argument('-gbp','--maximum_gap_between_points', type=float, help='Maximum number of seconds between points. Gap is ignored if this parameter is not set.', required=False)
parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
parser.add_argument('-mfe','--magnitude_for_feature_endpoints', type=float, default=0.8, help='Proportion of a feature\'s magnitude to take for its endpoints', required=False)
parser.add_argument('-nbf','--number_of_features', type=int, help='The number of features to find.', required=False)
parser.add_argument('-bs','--batch_size', type=int, default=10000, help='The number of features to be written to the database.', required=False)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)
c = source_conn.cursor()

# from https://stackoverflow.com/questions/43741185/sqlite3-disk-io-error
c.execute("PRAGMA journal_mode = TRUNCATE")

# find out the frame range
c.execute("SELECT min(frame_id) FROM clusters")
row = c.fetchone()
frame_lower = int(row[0])

c.execute("SELECT max(frame_id) FROM clusters")
row = c.fetchone()
frame_upper = int(row[0])

print("frame range of clusters detected: {} to {}".format(frame_lower, frame_upper))

print("Setting up tables...")

c.execute("DROP TABLE IF EXISTS features")
c.execute("DROP TABLE IF EXISTS feature_info")

c.execute("CREATE TABLE features (feature_id INTEGER, base_frame_id INTEGER, base_cluster_id INTEGER, charge_state INTEGER, start_rt REAL, end_rt REAL, quality_score REAL, summed_intensity INTEGER, scan_lower INTEGER, scan_upper INTEGER, mz_lower REAL, mz_upper REAL, PRIMARY KEY(feature_id))")
c.execute("CREATE TABLE feature_info (item TEXT, value TEXT)")

print("Setting up indexes...")

c.execute("CREATE INDEX IF NOT EXISTS idx_clusters_1 ON clusters (feature_id)")
c.execute("CREATE INDEX IF NOT EXISTS idx_clusters_2 ON clusters (frame_id, cluster_id)")
c.execute("CREATE INDEX IF NOT EXISTS idx_clusters_3 ON clusters (charge_state, frame_id, cluster_id)")

print("Resetting the feature IDs in the cluster table.")
c.execute("update clusters set feature_id=0 where feature_id!=0;")

print("Loading the clusters information")
c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum, scan_lower, scan_upper, mz_lower, mz_upper, retention_time_secs from clusters where charge_state >= {} and frame_id >= {} and frame_id <= {} order by frame_id, cluster_id asc;".format(args.minimum_charge_state, frame_lower, frame_upper))
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
    feature_rt_range = feature['feature_rt_range']
    cluster_indices = feature['cluster_indices']
    charge_state = feature['charge_state']
    quality = feature['quality']
    summed_intensity = feature['summed_intensity']
    scan_range = feature['scan_range']
    mz_range = feature['mz_range']

    feature_discovery_history.append(quality)

    poor_quality_rate = float(feature_discovery_history.count(0.0)) / FEATURE_DISCOVERY_HISTORY_LENGTH
    if len(noise_level_readings) > 0:
        estimated_noise_level = int(np.average(noise_level_readings))

    print("cluster max index {}, cluster id {}, cluster frame {}, cluster intensity {}, cluster indices {}".format(cluster_max_index, int(cluster[CLUSTER_ID_IDX]), int(cluster[CLUSTER_FRAME_ID_IDX]), cluster_intensity, cluster_indices))
    if quality > 0.5:
        print("feature {}, intensity {}, clusters {} (poor quality rate {}, base noise {})".format(feature_id, cluster_intensity, len(cluster_indices), poor_quality_rate, estimated_noise_level))
        # Assign this feature ID to all the clusters in the feature
        for cluster_idx in cluster_indices:
            values = (feature_id, int(clusters_v[cluster_idx][CLUSTER_FRAME_ID_IDX]), int(clusters_v[cluster_idx][CLUSTER_ID_IDX]))
            cluster_updates.append(values)

        # Add the feature's details to the collection
        values = (int(feature_id), int(base_cluster_frame_id), int(base_cluster_id), int(charge_state), feature_rt_range[0], feature_rt_range[1], float(quality), int(summed_intensity), int(scan_range[0]), int(scan_range[1]), float(mz_range[0]), float(mz_range[1]))
        feature_updates.append(values)

        feature_id += 1
    else:
        print("poor quality feature - discarding, intensity {}, length {} (poor quality rate {}, base noise {})".format(cluster_intensity, len(cluster_indices), poor_quality_rate, estimated_noise_level))
    print("")

    # remove the features we've processed from the run
    clusters_v[cluster_indices, CLUSTER_INTENSITY_SUM_IDX] = -1

    if (feature_id % args.batch_size) == 0:
        print("writing features out to the database...")
        # save this in the database
        c.executemany("UPDATE clusters SET feature_id=? WHERE frame_id=? AND cluster_id=?", cluster_updates)
        del cluster_updates[:]
        c.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", feature_updates)
        del feature_updates[:]
        source_conn.commit()

    # check whether we have finished
    if (cluster_intensity < estimated_noise_level):
        print("Cluster intensity ({}) is less than estimated base noise level ({})".format(cluster_intensity,estimated_noise_level))
        break
    if (poor_quality_rate > MAX_PROPORTION_POOR_QUALITY):
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

info.append(("features found", feature_id-1))
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))

print("{} info: {}".format(parser.prog, info))

c.executemany("INSERT INTO feature_info VALUES (?, ?)", info)

source_conn.commit()
source_conn.close()
