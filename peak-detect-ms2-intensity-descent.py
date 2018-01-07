from __future__ import print_function
import sys
import sqlite3
import pandas as pd
import argparse
import numpy as np
# import matplotlib.pyplot as plt
from scipy import signal
import peakutils
import time
import math
import collections
import json
import os

# MS1 summed region array indices
REGION_POINT_ID_IDX = 0
REGION_POINT_MZ_IDX = 1
REGION_POINT_SCAN_IDX = 2
REGION_POINT_INTENSITY_IDX = 3


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
    return (average, math.sqrt(variance))

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

parser = argparse.ArgumentParser(description='A tree descent method for MS2 peak detection.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-es','--empty_scans', type=int, default=2, help='Maximum number of empty scans to tolerate.', required=False)
parser.add_argument('-sd','--standard_deviations', type=int, default=4, help='Number of standard deviations to look either side of a point.', required=False)

args = parser.parse_args()

source_conn = sqlite3.connect(database=args.database_name, timeout=60)

c = source_conn.cursor()
print("Setting up tables and indexes")

c.execute('''DROP TABLE IF EXISTS ms2_peaks''')
c.execute('''CREATE TABLE ms2_peaks (feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, centroid_scan REAL, intensity_sum INTEGER, scan_upper INTEGER, scan_lower INTEGER, std_dev_mz REAL, std_dev_scan REAL, cluster_id INTEGER, 'rationale' TEXT, 'state' TEXT, intensity_max INTEGER, peak_max_mz REAL, peak_max_scan INTEGER, PRIMARY KEY (feature_id, peak_id))''')

c.execute('''DROP TABLE IF EXISTS ms2_peak_detect_info''')
c.execute('''CREATE TABLE ms2_peak_detect_info (item TEXT, value TEXT)''')


print("Resetting peak IDs")
c.execute("update summed_ms2_regions set peak_id=0 where peak_id!=0")

q = c.execute("SELECT value FROM ms2_feature_info WHERE item=\"ms1_feature_id_lower\"")
row = q.fetchone()
ms1_feature_id_lower = int(row[0])
print("ms1_feature_id_lower set to {} from the data".format(ms1_feature_id_lower))

q = c.execute("SELECT value FROM ms2_feature_info WHERE item=\"ms1_feature_id_upper\"")
row = q.fetchone()
ms1_feature_id_upper = int(row[0])
print("ms1_feature_id_upper set to {} from the data".format(ms1_feature_id_upper))

# Store the arguments as metadata in the database for later reference
ms2_peak_detect_info = []
for arg in vars(args):
    ms2_peak_detect_info.append((arg, getattr(args, arg)))

mono_peaks = []
point_updates = []
start_run = time.time()
for ms1_feature_id in range(ms1_feature_id_lower, ms1_feature_id_upper+1):
    peak_id = 1
    ms1_feature_df = pd.read_sql_query("select point_id,mz,scan,intensity from summed_ms2_regions where ms1_feature_id={} order by mz, scan asc;".format(ms1_feature_id), source_conn)
    print("Processing MS1 feature {}".format(ms1_feature_id))
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
                mz = ms1_feature_v[ms1_feature_v_index_to_use][REGION_POINT_INTENSITY_IDX]
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

                # f = plt.figure()
                # ax1 = f.add_subplot(111)
                # ax1.plot(peaks_sorted[:,1], peaks_sorted[:,2], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
                # ax1.plot(peaks_sorted[:,1], filtered, '-', markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.0, markersize=6)

                peak_maxima_indexes = peakutils.indexes(filtered, thres=0.05, min_dist=2)
                peak_minima_indexes = []
                if len(peak_maxima_indexes) > 1:
                    for idx,peak_maxima_index in enumerate(peak_maxima_indexes):
                        # ax1.plot(peaks_sorted[peak_maxima_index,1], peaks_sorted[peak_maxima_index,2], 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=10, alpha=0.5)
                        if idx>0:
                            intensities_between_maxima = filtered[peak_maxima_indexes[idx-1]:peak_maxima_indexes[idx]+1]
                            minimum_intensity_index = np.argmin(intensities_between_maxima)+peak_maxima_indexes[idx-1]
                            peak_minima_indexes.append(minimum_intensity_index)
                            # ax1.plot(peaks_sorted[minimum_intensity_index,1], peaks_sorted[minimum_intensity_index,2], 'x', markerfacecolor='purple', markeredgecolor='black', markeredgewidth=6.0, markersize=10, alpha=0.5)
                # print("peak maximum: {}".format(max_index))
                # print("peak minima: {}".format(peak_minima_indexes))
                indices_to_delete = np.empty(0)
                if len(peak_minima_indexes) > 0:
                    idx,lower_snip = findNearestLessThan(max_index, peak_minima_indexes)
                    idx,upper_snip = findNearestGreaterThan(max_index, peak_minima_indexes)
                    if lower_snip < max_index:
                        # ax1.plot(peaks_sorted[lower_snip,1], peaks_sorted[lower_snip,2], '+', markerfacecolor='purple', markeredgecolor='red', markeredgewidth=2.0, markersize=20, alpha=1.0)
                        indices_to_delete = np.concatenate((indices_to_delete,np.arange(lower_snip)))
                    if upper_snip > max_index:
                        # ax1.plot(peaks_sorted[upper_snip,1], peaks_sorted[upper_snip,2], '+', markerfacecolor='purple', markeredgecolor='red', markeredgewidth=2.0, markersize=20, alpha=1.0)
                        indices_to_delete = np.concatenate((indices_to_delete,np.arange(upper_snip+1,len(peaks_sorted))))
                    sorted_peaks_indexes = np.delete(sorted_peaks_indexes, indices_to_delete, 0)
                    peak_indices = peak_indices[sorted_peaks_indexes]
                    peaks_sorted = ms1_feature_v[peak_indices]
                    # ax1.plot(peaks_sorted[:,1], peaks_sorted[:,2], 'o', markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
                    rationale["point ids after trimming"] = peaks_sorted[:,REGION_POINT_ID_IDX].astype(int).tolist()

                # plt.title("Peak {}".format(peak_id))
                # plt.xlabel('scan')
                # plt.ylabel('intensity')
                # plt.margins(0.02)
                # plt.show()

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
                point_updates.append((peak_id, ms1_feature_id, ms1_feature_v[int(p)][REGION_POINT_ID_IDX]))

            # Add the peak's details to the collection
            peak_intensity_sum = np.sum(peak_intensity)
            peak_intensity_max = np.max(peak_intensity)
            peak_scan_upper = np.max(peak_scan)
            peak_scan_lower = np.min(peak_scan)
            peak_mz_centroid, peak_std_dev_mz = weighted_avg_and_std(values=peak_mz, weights=peak_intensity)
            peak_scan_centroid, peak_std_dev_scan = weighted_avg_and_std(values=peak_scan, weights=peak_intensity)
            peak_points = ms1_feature_v[peak_indices]
            peak_max_index = peak_points[:,REGION_POINT_INTENSITY_IDX].argmax()
            peak_max_mz = peak_points[peak_max_index][REGION_POINT_MZ_IDX]
            peak_max_scan = peak_points[peak_max_index][REGION_POINT_SCAN_IDX].astype(int)
            mono_peaks.append((ms1_feature_id, peak_id, peak_mz_centroid, peak_scan_centroid, peak_intensity_sum, peak_scan_upper, peak_scan_lower, peak_std_dev_mz, peak_std_dev_scan, json.dumps(rationale), ' ', peak_intensity_max, peak_max_mz, peak_max_scan))

            peak_id += 1

        # remove the points we've processed
        ms1_feature_v = np.delete(ms1_feature_v, peak_indices, 0)

    # Write out the peaks we found for this feature
    print("Writing out the peaks we found for this feature.")
    c.executemany("INSERT INTO ms2_peaks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)", mono_peaks)
    mono_peaks = []

    # Update the points in the summed_ms2_regions table
    print("Updating the points in the summed_ms2_regions table.")
    c.executemany("UPDATE summed_ms2_regions SET peak_id=? WHERE ms1_feature_id=? AND point_id=?", point_updates)
    point_updates = []

    stop_feature = time.time()
    print("{} seconds to process feature {} - {} peaks".format(stop_feature-start_feature, ms1_feature_id, peak_id))

stop_run = time.time()
print("{} seconds to process features {} to {}".format(stop_run-start_run, ms1_feature_id_lower, ms1_feature_id_upper))

# write out the processing info
ms2_peak_detect_info.append(("run processing time (sec)", stop_run-start_run))
ms2_peak_detect_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO ms2_peak_detect_info VALUES (?, ?)", ms2_peak_detect_info)
source_conn.commit()
source_conn.close()

# plt.close('all')
