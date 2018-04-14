from __future__ import print_function
import sys
import sqlite3
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
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=True)
parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=True)
parser.add_argument('-sd','--standard_deviations', type=int, default=10, help='Number of standard deviations to look either side of a point.', required=False)
parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
args = parser.parse_args()

src_conn = sqlite3.connect(args.source_database_name)
src_c = src_conn.cursor()

dest_conn = sqlite3.connect(args.destination_database_name)
dest_c = dest_conn.cursor()

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

# Store the arguments as metadata in the database for later reference
ms1_feature_region_peak_detect_info = []
for arg in vars(args):
    ms1_feature_region_peak_detect_info.append((arg, getattr(args, arg)))

print("Setting up tables and indexes")
dest_c.execute("DROP TABLE IF EXISTS ms1_feature_region_peaks")
dest_c.execute("DROP TABLE IF EXISTS ms1_feature_region_peak_detect_info")
dest_c.execute("DROP TABLE IF EXISTS feature_base_peaks")

dest_c.execute("CREATE TABLE ms1_feature_region_peaks (feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, centroid_scan REAL, intensity_sum INTEGER, scan_upper INTEGER, scan_lower INTEGER, std_dev_mz REAL, std_dev_scan REAL, rationale TEXT, intensity_max INTEGER, peak_max_mz REAL, peak_max_scan INTEGER, PRIMARY KEY (feature_id, peak_id))")
dest_c.execute("CREATE TABLE ms1_feature_region_peak_detect_info (item TEXT, value TEXT)")
dest_c.execute("CREATE TABLE feature_base_peaks (feature_id INTEGER, base_peak_id INTEGER, PRIMARY KEY (feature_id, base_peak_id))")

dest_c.execute("CREATE INDEX IF NOT EXISTS idx_ms1_region_peaks_1 ON summed_ms1_regions (feature_id)")
dest_c.execute("CREATE INDEX IF NOT EXISTS idx_ms1_region_peaks_2 ON summed_ms1_regions (feature_id,point_id)")

mono_peaks = []
point_updates = []
base_peaks = []
start_run = time.time()

# Take the ms1 features within the m/z band of interest, and detect peaks in the summed regions

print("Loading the MS1 features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
features_df = pd.read_sql_query("""select feature_id from features where feature_id >= {} and 
    feature_id <= {} and charge_state >= {} and mz_lower <= {} and mz_upper >= {} order by feature_id ASC;"""
    .format(args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state, args.mz_upper, args.mz_lower), src_conn)
features_v = features_df.values

for feature in features_v:
    feature_id = int(feature[FEATURE_ID_IDX])

    peak_id = 1
    ms1_feature_df = pd.read_sql_query("select point_id,mz,scan,intensity from summed_ms1_regions where feature_id={} order by mz, scan asc;".format(feature_id), dest_conn)
    ms1_feature_v = ms1_feature_df.values
    if len(ms1_feature_v) > 0:
        print("Processing MS1 feature {}".format(feature_id))
        start_feature = time.time()
        scan_lower = int(np.min(ms1_feature_v[:,REGION_POINT_SCAN_IDX]))
        scan_upper = int(np.max(ms1_feature_v[:,REGION_POINT_SCAN_IDX]))
        while len(ms1_feature_v) > 0:
            peak_indices = np.empty(0, dtype=int)

            rationale = collections.OrderedDict()
            max_intensity_index = np.argmax(ms1_feature_v[:,REGION_POINT_INTENSITY_IDX])
            mz = ms1_feature_v[max_intensity_index][REGION_POINT_MZ_IDX]
            scan = int(ms1_feature_v[max_intensity_index][REGION_POINT_SCAN_IDX])
            intensity = int(ms1_feature_v[max_intensity_index][REGION_POINT_INTENSITY_IDX])
            point_id = int(ms1_feature_v[max_intensity_index][REGION_POINT_ID_IDX])
            peak_indices = np.append(peak_indices, max_intensity_index)
            rationale["highest intensity point id"] = point_id
            print("\nmax point mz {}, scan {}".format(mz, scan))


            # Look for other points belonging to this peak
            std_dev_window = standard_deviation(mz) * args.standard_deviations
            # Look in the 'up' direction
            print("looking up")
            search_scan = scan
            missed_scans = 0
            while (search_scan >= scan_lower):
                print("scan {}".format(search_scan))
                print("points on scan line {}".format(ms1_feature_v[:,REGION_POINT_SCAN_IDX] == search_scan))
                print("distance from {}: {}".format(mz, abs(ms1_feature_v[:,REGION_POINT_MZ_IDX] - mz)))
                nearby_indices_up = np.where((ms1_feature_v[:,REGION_POINT_SCAN_IDX] == search_scan) & 
                    (abs(ms1_feature_v[:,REGION_POINT_MZ_IDX] - mz) <= std_dev_window))[0]
                print("found points {}".format(ms1_feature_v[nearby_indices_up,REGION_POINT_MZ_IDX]))
                if len(nearby_indices_up) > 0:
                    # Update the m/z window
                    max_intensity_nearby_index = np.argmax(ms1_feature_v[nearby_indices_up, REGION_POINT_INTENSITY_IDX])
                    mz = ms1_feature_v[nearby_indices_up[max_intensity_nearby_index],REGION_POINT_MZ_IDX]
                    std_dev_window = standard_deviation(mz) * args.standard_deviations
                    peak_indices = np.append(peak_indices, nearby_indices_up)
                search_scan -= 1
            # Look in the 'down' direction
            print("\nlooking down")
            search_scan = scan+1    # don't count the points on the starting scan line again
            missed_scans = 0
            mz = ms1_feature_v[max_intensity_index][REGION_POINT_MZ_IDX]
            std_dev_window = standard_deviation(mz) * args.standard_deviations
            while (search_scan <= scan_upper):
                print("scan {}".format(search_scan))
                print("points on scan line {}".format(ms1_feature_v[:,REGION_POINT_SCAN_IDX] == search_scan))
                print("distance from {}: {}".format(mz, abs(ms1_feature_v[:,REGION_POINT_MZ_IDX] - mz)))
                nearby_indices_down = np.where((ms1_feature_v[:,REGION_POINT_SCAN_IDX] == search_scan) & 
                    (abs(ms1_feature_v[:,REGION_POINT_MZ_IDX] - mz) <= std_dev_window))[0]
                print("found points {}".format(ms1_feature_v[nearby_indices_down,REGION_POINT_MZ_IDX]))
                if len(nearby_indices_down) > 0:
                    # Update the m/z window
                    max_intensity_nearby_index = np.argmax(ms1_feature_v[nearby_indices_down, REGION_POINT_INTENSITY_IDX])
                    mz = ms1_feature_v[nearby_indices_down[max_intensity_nearby_index],REGION_POINT_MZ_IDX]
                    std_dev_window = standard_deviation(mz) * args.standard_deviations
                    peak_indices = np.append(peak_indices, nearby_indices_down)
                search_scan += 1

            if len(peak_indices) > 1:
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
                peak_max_index = np.argmax(peak_points[:,REGION_POINT_INTENSITY_IDX])
                peak_max_mz = float(peak_points[peak_max_index][REGION_POINT_MZ_IDX])
                peak_max_scan = int(peak_points[peak_max_index][REGION_POINT_SCAN_IDX])
                mono_peaks.append((feature_id, peak_id, peak_mz_centroid, peak_scan_centroid, peak_intensity_sum, peak_scan_upper, peak_scan_lower, peak_std_dev_mz, peak_std_dev_scan, json.dumps(rationale), peak_intensity_max, peak_max_mz, peak_max_scan))

                peak_id += 1

            # remove the points we've processed
            ms1_feature_v = np.delete(ms1_feature_v, peak_indices, 0)

        # Remember the base peak for the feature
        if len(mono_peaks) > 0:
            base_peak_id = max(mono_peaks, key=itemgetter(4))[1]    # find the max peak_intensity_sum, return its peak_id
            base_peaks.append((feature_id, base_peak_id))

            # Write out the peaks for this feature
            src_c.executemany("INSERT INTO ms1_feature_region_peaks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", mono_peaks)
            mono_peaks = []

        # Update the points in the summed_ms1_regions table
        if len(point_updates) > 0:
            src_c.executemany("UPDATE summed_ms1_regions SET peak_id=? WHERE feature_id=? AND point_id=?", point_updates)
            point_updates = []

        stop_feature = time.time()
        # print("{} seconds to process feature {} ({} peaks)".format(stop_feature-start_feature, feature_id, peak_id))

print("Write out the base peaks")
src_c.executemany("INSERT INTO feature_base_peaks VALUES (?, ?)", base_peaks)

stop_run = time.time()
print("{} seconds to process features {} to {}".format(stop_run-start_run, args.feature_id_lower, args.feature_id_upper))

# write out the processing info
ms1_feature_region_peak_detect_info.append(("run processing time (sec)", stop_run-start_run))
ms1_feature_region_peak_detect_info.append(("processed", time.ctime()))

ms1_feature_region_peak_detect_info_entry = []
ms1_feature_region_peak_detect_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in ms1_feature_region_peak_detect_info)))

dest_c.executemany("INSERT INTO ms1_feature_region_peak_detect_info VALUES (?, ?)", ms1_feature_region_peak_detect_info_entry)

dest_conn.commit()
dest_conn.close()

src_conn.close()

# plt.close('all')
