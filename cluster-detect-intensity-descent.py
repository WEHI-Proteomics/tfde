import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import copy
import argparse
import os.path

# def standard_deviation(mz):
#     instrument_resolution = 40000.0
#     return (mz / instrument_resolution) / 2.35482


parser = argparse.ArgumentParser(description='A tree descent method for clustering peaks.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number to process.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number to process.', required=True)
parser.add_argument('-sl','--scan_lower', type=int, default=0, help='The lower scan number to process.', required=False)
parser.add_argument('-su','--scan_upper', type=int, default=138, help='The upper scan number to process.', required=False)
parser.add_argument('-ir','--isotope_number_right', type=int, default=5, help='Isotope numbers to look on the right.', required=False)
parser.add_argument('-il','--isotope_number_left', type=int, default=2, help='Isotope numbers to look on the left.', required=False)
parser.add_argument('-mi','--minimum_peak_intensity', type=int, default=250, help='Minimum peak intensity to process.', required=False)
parser.add_argument('-mp','--minimum_peaks_nearby', type=int, default=3, help='A peak must have more peaks in its neighbourhood for processing.', required=False)
parser.add_argument('-cs','--maximum_charge_state', type=int, default=5, help='Maximum charge state to look for.', required=False)
parser.add_argument('-sd','--scan_std_dev', type=int, default=2, help='Number of weighted standard deviations to look either side of the intense peak, in the scan dimension.', required=False)
parser.add_argument('-md','--mz_std_dev', type=int, default=2, help='Number of weighted standard deviations to look either side of the intense peak, in the m/z dimension.', required=False)

args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
cluster_detect_info = []
for arg in vars(args):
    cluster_detect_info.append((arg, getattr(args, arg)))

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)
c = source_conn.cursor()

print("Setting up tables and indexes")
c.execute('''DROP TABLE IF EXISTS clusters''')
c.execute('''CREATE TABLE `clusters` ( `frame_id` INTEGER, `cluster_id` INTEGER, `charge_state` INTEGER, 'base_isotope_peak_id' INTEGER, 'cluster_quality' REAL, PRIMARY KEY(`cluster_id`,`frame_id`) )''')
c.execute('''DROP INDEX IF EXISTS idx_clusters''')
c.execute('''CREATE INDEX idx_clusters ON clusters (frame_id,cluster_id)''')
c.execute("update peaks set cluster_id=0")
c.execute('''DROP TABLE IF EXISTS cluster_detect_info''')
c.execute('''CREATE TABLE cluster_detect_info (item TEXT, value TEXT)''')
source_conn.commit()

DELTA_MZ = 1.003355     # mass difference between Carbon-12 and Carbon-13

clusters = []
start_run = time.time()
for frame_id in range(args.frame_lower, args.frame_upper+1):
    start_frame = time.time()
    cluster_id = 1
    peak_id = 1
    # Get all the peaks for this frame
    peaks_df = pd.read_sql_query("select peak_id,centroid_mz,centroid_scan,intensity_sum,scan_upper,scan_lower,std_dev_mz,std_dev_scan from peaks where frame_id={} order by peak_id asc;"
        .format(frame_id), source_conn)
    peaks_v = peaks_df.values
    # for i in range(1,2):
    while len(peaks_v) > 0:
        # print("{} peaks remaining.".format(len(peaks_v)))
        spectra = []
        cluster_peak_indices = np.empty(0, dtype=int)

        max_intensity_index = peaks_v.argmax(axis=0)[3]
        cluster_peak_indices = np.append(cluster_peak_indices, max_intensity_index)
        peak_id = int(peaks_v[max_intensity_index][0])

        print("peak id {}".format(peak_id))

        peak_mz = peaks_v[max_intensity_index][1]
        peak_scan = peaks_v[max_intensity_index][2]
        peak_intensity = int(peaks_v[max_intensity_index][3])
        peak_scan_lower = peak_scan - args.scan_std_dev*peaks_v[max_intensity_index][7]
        peak_scan_upper = peak_scan + args.scan_std_dev*peaks_v[max_intensity_index][7]
        mz_comparison_tolerance = args.mz_std_dev*peaks_v[max_intensity_index][6]
        print("m/z tolerance: +/- {}".format(mz_comparison_tolerance))
        if peak_intensity < args.minimum_peak_intensity:
            print "Reached minimum peak intensity - exiting."
            break
        peaks_nearby_indices = np.where((peaks_v[:,1] <= peak_mz + DELTA_MZ*args.isotope_number_right) & (peaks_v[:,1] >= peak_mz - DELTA_MZ*args.isotope_number_left) & (peaks_v[:,2] >= peak_scan_lower) & (peaks_v[:,2] <= peak_scan_upper))[0]
        peaks_nearby = peaks_v[peaks_nearby_indices]
        peaks_nearby_sorted = peaks_nearby[np.argsort(peaks_nearby[:,1])]
        print("found {} peaks nearby".format(len(peaks_nearby_indices)))

        if len(peaks_nearby_indices) >= args.minimum_peaks_nearby:
            # Go through the charge states and isotopes and find the combination with maximum total intensity
            isotope_search_results = [[0, np.empty(0, dtype=int)] for x in range(args.maximum_charge_state+1)]      # array of summed intensity of the peaks, peak IDs
            for charge_state in range(1,args.maximum_charge_state+1):
                # Pick out the peaks belonging to this cluster from the peaks nearby
                # To the right...
                for isotope_number in range(1,args.isotope_number_right+1):
                    mz = peak_mz + (isotope_number*DELTA_MZ/charge_state)
                    cluster_peak_indices_right = np.where((abs(peaks_nearby[:,1] - mz) < mz_comparison_tolerance))[0]
                    if len(cluster_peak_indices_right) > 0:
                        # Add the sum of the peak(s) intensity to this charge state in the matrix
                        isotope_search_results[charge_state][0] += np.sum(peaks_nearby[cluster_peak_indices_right][:,3])
                        # Add the ID of the peak(s) intensity to this charge state in the matrix
                        isotope_search_results[charge_state][1] = np.append(isotope_search_results[charge_state][1], peaks_nearby_indices[cluster_peak_indices_right])

                # To the left...
                for isotope_number in range(1,args.isotope_number_left+1):
                    mz = peak_mz - (isotope_number*DELTA_MZ/charge_state)
                    cluster_peak_indices_left = np.where((abs(peaks_nearby[:,1] - mz) < mz_comparison_tolerance))[0]
                    if len(cluster_peak_indices_left) > 0:
                        # Add the sum of the peak(s) intensity to this charge state in the matrix
                        isotope_search_results[charge_state][0] += np.sum(peaks_nearby[cluster_peak_indices_left][:,3])
                        # Add the ID of the peak(s) intensity to this charge state in the matrix
                        isotope_search_results[charge_state][1] = np.append(isotope_search_results[charge_state][1], peaks_nearby_indices[cluster_peak_indices_left])
            # Find the charge state with the maximum summed intensity
            max_intensity = 0
            max_peaks = np.empty(0, dtype=int)
            charge = 0
            for idx, r in enumerate(isotope_search_results):
                if r[0] > max_intensity:
                    max_intensity = r[0]
                    max_peaks = r[1]
                    charge = idx
            if max_intensity > 0:
                cluster_peak_indices = np.append(cluster_peak_indices, max_peaks)
                cluster_quality = 1.0
                print("cluster id {}, charge {}".format(cluster_id, charge))
                # Reflect the clusters in the peak table of the database
                cluster_peak_indices = np.unique(cluster_peak_indices)
                for p in cluster_peak_indices:
                    p_id = int(peaks_v[p][0])
                    # Update the peaks in the peaks table with their cluster ID
                    values = (cluster_id, frame_id, p_id)
                    c.execute("update peaks set cluster_id=? where frame_id=? and peak_id=?", values)
                clusters.append((frame_id, cluster_id, charge, peak_id, cluster_quality))
                cluster_id += 1
            else:
                print "Found no isotopic peaks either side of this peak."
        else:
            print "Found less than {} peaks nearby - skipping peak search.".format(args.minimum_peaks_nearby)

        # remove the points we've processed from the frame
        print("removing peak ids {} - {} peaks remaining\n".format(peaks_v[cluster_peak_indices,0].astype(int), len(peaks_v)))
        peaks_v = np.delete(peaks_v, cluster_peak_indices, 0)

    stop_frame = time.time()
    print("{} seconds to process frame - {} peaks".format(stop_frame-start_frame, peak_id))

# Write out all the peaks to the database
c.executemany("INSERT INTO clusters VALUES (?, ?, ?, ?, ?)", clusters)
stop_run = time.time()

cluster_detect_info.append(("run processing time (sec)", stop_run-start_run))
cluster_detect_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO cluster_detect_info VALUES (?, ?)", cluster_detect_info)

source_conn.commit()
source_conn.close()
