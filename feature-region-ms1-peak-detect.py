from __future__ import print_function
import sys
import pymysql
import pandas as pd
import argparse
import numpy as np
from scipy import signal
import peakutils
import time
import math
import collections
import json
import os
from operator import itemgetter

# MS1 summed region array indices
REGION_POINT_ID_IDX = 0
REGION_POINT_MZ_IDX = 1
REGION_POINT_SCAN_IDX = 2
REGION_POINT_INTENSITY_IDX = 3

# feature array indices
FEATURE_ID_IDX = 0
FEATURE_START_FRAME_IDX = 1
FEATURE_END_FRAME_IDX = 2
FEATURE_SCAN_LOWER_IDX = 3
FEATURE_SCAN_UPPER_IDX = 4
FEATURE_MZ_LOWER_IDX = 5
FEATURE_MZ_UPPER_IDX = 6

MIN_POINTS_IN_PEAK_TO_CHECK_FOR_TROUGHS = 10

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

# Source: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy/2415343
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (float(average), float(math.sqrt(variance)))

def findNearestGreaterThan(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff<0] = sys.maxint
    idx = diff.argmin()
    return idx, inputData[idx]

def findNearestLessThan(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff>0] = -sys.maxint
    idx = diff.argmax()
    return idx, inputData[idx]

parser = argparse.ArgumentParser(description='Detect peaks in MS1 feature regions.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
parser.add_argument('-es','--empty_scans', type=int, default=2, help='Maximum number of empty scans to tolerate.', required=False)
parser.add_argument('-sd','--standard_deviations', type=int, default=4, help='Number of standard deviations to look either side of a point.', required=False)
parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
args = parser.parse_args()

source_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database="{}".format(args.database_name))
src_c = source_conn.cursor()

if args.feature_id_lower is None:
    src_c.execute("SELECT MIN(feature_id) FROM features")
    row = src_c.fetchone()
    args.feature_id_lower = int(row[0])
    print("feature_id_lower set to {} from the data".format(args.feature_id_lower))

if args.feature_id_upper is None:
    src_c.execute("SELECT MAX(feature_id) FROM features")
    row = src_c.fetchone()
    args.feature_id_upper = int(row[0])
    print("feature_id_upper set to {} from the data".format(args.feature_id_upper))

print("Setting up tables and indexes")
src_c.execute("CREATE OR REPLACE TABLE ms1_feature_region_peaks (feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, centroid_scan REAL, intensity_sum INTEGER, scan_upper INTEGER, scan_lower INTEGER, std_dev_mz REAL, std_dev_scan REAL, rationale TEXT, intensity_max INTEGER, peak_max_mz REAL, peak_max_scan INTEGER, PRIMARY KEY (feature_id, peak_id))")
src_c.execute("CREATE OR REPLACE TABLE ms1_feature_region_peak_detect_info (item TEXT, value TEXT)")
src_c.execute("CREATE OR REPLACE TABLE feature_base_peaks (feature_id INTEGER, base_peak_id INTEGER, PRIMARY KEY (feature_id, base_peak_id))")

print("Resetting peak IDs")
src_c.execute("update summed_ms1_regions set peak_id=0 where peak_id!=0")

# Store the arguments as metadata in the database for later reference
ms1_feature_region_peak_detect_info = []
for arg in vars(args):
    ms1_feature_region_peak_detect_info.append((arg, getattr(args, arg)))

mono_peaks = []
point_updates = []
base_peaks = []
start_run = time.time()

print("Loading the MS1 features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
features_df = pd.read_sql_query("select feature_id,start_frame,end_frame,scan_lower,scan_upper,mz_lower,mz_upper from features where feature_id >= {} and feature_id <= {} and charge_state >= {} order by feature_id ASC;".format(args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state), source_conn)
features_v = features_df.values

for feature in features_v:
    feature_id = int(feature[FEATURE_ID_IDX])
    feature_start_frame = int(feature[FEATURE_START_FRAME_IDX])
    feature_end_frame = int(feature[FEATURE_END_FRAME_IDX])
    feature_scan_lower = int(feature[FEATURE_SCAN_LOWER_IDX])
    feature_scan_upper = int(feature[FEATURE_SCAN_UPPER_IDX])
    feature_mz_lower = feature[FEATURE_MZ_LOWER_IDX]
    feature_mz_upper = feature[FEATURE_MZ_UPPER_IDX]

    peak_id = 1
    ms1_feature_df = pd.read_sql_query("select point_id,mz,scan,intensity from summed_ms1_regions where feature_id={} order by mz, scan asc;".format(feature_id), source_conn)
    print("Processing MS1 feature {}".format(feature_id))
    start_feature = time.time()
    ms1_feature_v = ms1_feature_df.values
    print("frame occupies {} bytes".format(ms1_feature_v.nbytes))
    scan_lower = int(np.min(ms1_feature_v[:,REGION_POINT_SCAN_IDX]))
    scan_upper = int(np.max(ms1_feature_v[:,REGION_POINT_SCAN_IDX]))
    print("scan range {}-{}".format(scan_lower,scan_upper))
    while len(ms1_feature_v) > 0:
        peak_indices = np.empty(0, dtype=int)

        rationale = collections.OrderedDict()
        max_intensity_index = ms1_feature_v.argmax(axis=0)[REGION_POINT_INTENSITY_IDX]
        mz = ms1_feature_v[max_intensity_index][REGION_POINT_MZ_IDX]
        scan = int(ms1_feature_v[max_intensity_index][REGION_POINT_SCAN_IDX])
        intensity = int(ms1_feature_v[max_intensity_index][REGION_POINT_INTENSITY_IDX])
        point_id = int(ms1_feature_v[max_intensity_index][REGION_POINT_ID_IDX])
        peak_indices = np.append(peak_indices, max_intensity_index)
        rationale["highest intensity point id"] = point_id

        # Look for other points belonging to this peak
        std_dev_window = standard_deviation(mz) * args.standard_deviations
        # Look in the 'up' direction
        scan_offset = 1
        missed_scans = 0
        while (missed_scans < args.empty_scans) and (scan-scan_offset >= scan_lower):
            # print("looking in scan {}".format(scan-scan_offset))
            nearby_indices_up = np.where((ms1_feature_v[:,REGION_POINT_SCAN_IDX] == scan-scan_offset) & (ms1_feature_v[:,REGION_POINT_MZ_IDX] >= mz - std_dev_window) & (ms1_feature_v[:,REGION_POINT_MZ_IDX] <= mz + std_dev_window))[0]
            nearby_points_up = ms1_feature_v[nearby_indices_up]
            # print("nearby indices: {}".format(nearby_indices_up))
            if len(nearby_indices_up) == 0:
                missed_scans += 1
                # print("found no points")
            else:
                if len(nearby_indices_up) > 1:
                    # take the most intense point if there's more than one point found on this scan
                    ms1_feature_v_index_to_use = nearby_indices_up[np.argmax(nearby_points_up[:,REGION_POINT_INTENSITY_IDX])]
                else:
                    ms1_feature_v_index_to_use = nearby_indices_up[0]
                # Update the m/z window
                mz = ms1_feature_v[ms1_feature_v_index_to_use][REGION_POINT_MZ_IDX]
                std_dev_window = standard_deviation(mz) * args.standard_deviations
                missed_scans = 0
                # print("found {} points".format(len(nearby_indices_up)))
                peak_indices = np.append(peak_indices, ms1_feature_v_index_to_use)
            scan_offset += 1
        # Look in the 'down' direction
        scan_offset = 1
        missed_scans = 0
        mz = ms1_feature_v[max_intensity_index][REGION_POINT_MZ_IDX]
        std_dev_window = standard_deviation(mz) * args.standard_deviations
        while (missed_scans < args.empty_scans) and (scan+scan_offset <= scan_upper):
            # print("looking in scan {}".format(scan+scan_offset))
            nearby_indices_down = np.where((ms1_feature_v[:,REGION_POINT_SCAN_IDX] == scan+scan_offset) & (ms1_feature_v[:,REGION_POINT_MZ_IDX] >= mz - std_dev_window) & (ms1_feature_v[:,REGION_POINT_MZ_IDX] <= mz + std_dev_window))[0]
            nearby_points_down = ms1_feature_v[nearby_indices_down]
            if len(nearby_indices_down) == 0:
                missed_scans += 1
                # print("found no points")
            else:
                if len(nearby_indices_down) > 1:
                    # take the most intense point if there's more than one point found on this scan
                    ms1_feature_v_index_to_use = nearby_indices_down[np.argmax(nearby_points_down[:,REGION_POINT_INTENSITY_IDX])]
                else:
                    ms1_feature_v_index_to_use = nearby_indices_down[0]
                
                # Update the m/z window
                mz = ms1_feature_v[ms1_feature_v_index_to_use][REGION_POINT_MZ_IDX]
                std_dev_window = standard_deviation(mz) * args.standard_deviations
                missed_scans = 0
                peak_indices = np.append(peak_indices, ms1_feature_v_index_to_use)
            scan_offset += 1

        if len(peak_indices) > 1:

            if len(peak_indices) > MIN_POINTS_IN_PEAK_TO_CHECK_FOR_TROUGHS:
                # Check whether it has more than one peak
                # filter the intensity with a Gaussian filter
                sorted_peaks_indexes = np.argsort(ms1_feature_v[peak_indices][:,REGION_POINT_SCAN_IDX])
                peaks_sorted = ms1_feature_v[peak_indices[sorted_peaks_indexes]]
                rationale["point ids"] = peaks_sorted[:,REGION_POINT_ID_IDX].astype(int).tolist()
                filtered = signal.savgol_filter(peaks_sorted[:,REGION_POINT_INTENSITY_IDX], 9, 5)
                max_index = np.argmax(peaks_sorted[:,REGION_POINT_INTENSITY_IDX])

                peak_maxima_indexes = peakutils.indexes(filtered, thres=0.05, min_dist=2)
                peak_minima_indexes = []
                if len(peak_maxima_indexes) > 1:
                    for idx,peak_maxima_index in enumerate(peak_maxima_indexes):
                        if idx>0:
                            intensities_between_maxima = filtered[peak_maxima_indexes[idx-1]:peak_maxima_indexes[idx]+1]
                            minimum_intensity_index = np.argmin(intensities_between_maxima)+peak_maxima_indexes[idx-1]
                            peak_minima_indexes.append(minimum_intensity_index)

                indices_to_delete = np.empty(0)
                if len(peak_minima_indexes) > 0:
                    idx,lower_snip = findNearestLessThan(max_index, peak_minima_indexes)
                    idx,upper_snip = findNearestGreaterThan(max_index, peak_minima_indexes)
                    if lower_snip < max_index:
                        indices_to_delete = np.concatenate((indices_to_delete,np.arange(lower_snip)))
                    if upper_snip > max_index:
                        indices_to_delete = np.concatenate((indices_to_delete,np.arange(upper_snip+1,len(peaks_sorted))))
                    sorted_peaks_indexes = np.delete(sorted_peaks_indexes, indices_to_delete, 0)
                    peak_indices = peak_indices[sorted_peaks_indexes]
                    peaks_sorted = ms1_feature_v[peak_indices]
                    rationale["point ids after trimming"] = peaks_sorted[:,REGION_POINT_ID_IDX].astype(int).tolist()

            # Add the peak to the collection
            peak_mz = []
            peak_scan = []
            peak_intensity = []

            # Update database
            for p in peak_indices:
                # Collect all the points in the peak
                peak_mz.append(ms1_feature_v[p][REGION_POINT_MZ_IDX])
                peak_scan.append(ms1_feature_v[p][REGION_POINT_SCAN_IDX])
                peak_intensity.append(int(ms1_feature_v[p][REGION_POINT_INTENSITY_IDX]))

                # Assign this peak ID to all the points in the peak
                point_updates.append((peak_id, feature_id, int(ms1_feature_v[int(p)][REGION_POINT_ID_IDX])))

            # Add the peak's details to the collection
            peak_intensity_sum = int(np.sum(peak_intensity))
            peak_intensity_max = int(np.max(peak_intensity))
            peak_scan_upper = int(np.max(peak_scan))
            peak_scan_lower = int(np.min(peak_scan))
            peak_mz_centroid, peak_std_dev_mz = weighted_avg_and_std(values=peak_mz, weights=peak_intensity)
            peak_scan_centroid, peak_std_dev_scan = weighted_avg_and_std(values=peak_scan, weights=peak_intensity)
            peak_points = ms1_feature_v[peak_indices]
            peak_max_index = peak_points[:,REGION_POINT_INTENSITY_IDX].argmax()
            peak_max_mz = float(peak_points[peak_max_index][REGION_POINT_MZ_IDX])
            peak_max_scan = int(peak_points[peak_max_index][REGION_POINT_SCAN_IDX])
            mono_peaks.append((feature_id, peak_id, peak_mz_centroid, peak_scan_centroid, peak_intensity_sum, peak_scan_upper, peak_scan_lower, peak_std_dev_mz, peak_std_dev_scan, json.dumps(rationale), peak_intensity_max, peak_max_mz, peak_max_scan))

            peak_id += 1

        # remove the points we've processed
        ms1_feature_v = np.delete(ms1_feature_v, peak_indices, 0)

    # Remember the base peak for the feature
    base_peak_id = max(mono_peaks, key=itemgetter(4))[1]    # find the max peak_intensity_sum, return its peak_id
    base_peaks.append((feature_id, base_peak_id))

    # Write out the peaks for this feature
    print("Writing out the peaks for this feature.")
    src_c.executemany("INSERT INTO ms1_feature_region_peaks VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", mono_peaks)
    mono_peaks = []

    # Update the points in the summed_ms1_regions table
    print("Updating the points in the summed_ms1_regions table.")
    src_c.executemany("UPDATE summed_ms1_regions SET peak_id=%s WHERE feature_id=%s AND point_id=%s", point_updates)
    point_updates = []

    stop_feature = time.time()
    print("{} seconds to process feature {} ({} peaks)".format(stop_feature-start_feature, feature_id, peak_id))

print("Write out the base peaks")
src_c.executemany("INSERT INTO feature_base_peaks VALUES (%s, %s)", base_peaks)

stop_run = time.time()
print("{} seconds to process features {} to {}".format(stop_run-start_run, args.feature_id_lower, args.feature_id_upper))

# Keep a record of the features we actually processed
ms1_feature_region_peak_detect_info.append(("feature_id_lower", args.feature_id_lower))
ms1_feature_region_peak_detect_info.append(("feature_id_upper", args.feature_id_upper))

# write out the processing info
ms1_feature_region_peak_detect_info.append(("run processing time (sec)", stop_run-start_run))
ms1_feature_region_peak_detect_info.append(("processed", time.ctime()))

ms1_feature_region_peak_detect_info_entry = []
ms1_feature_region_peak_detect_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in ms1_feature_region_peak_detect_info)))

src_c.executemany("INSERT INTO ms1_feature_region_peak_detect_info VALUES (%s, %s)", ms1_feature_region_peak_detect_info_entry)

source_conn.commit()
source_conn.close()

# plt.close('all')
