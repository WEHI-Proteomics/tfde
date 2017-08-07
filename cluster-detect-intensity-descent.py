import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import copy
import argparse
from pyteomics import mgf
import os.path
from glob import glob
import csv
import subprocess
from shutil import copyfile


parser = argparse.ArgumentParser(description='A tree descent method for clustering peaks.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number.', required=True)
parser.add_argument('-sl','--scan_lower', type=int, default=0, help='The lower scan number.', required=False)
parser.add_argument('-su','--scan_upper', type=int, default=138, help='The upper scan number.', required=False)
parser.add_argument('-ir','--isotope_number_right', type=int, default=10, help='Isotope numbers to look on the right.', required=False)
parser.add_argument('-il','--isotope_number_left', type=int, default=5, help='Isotope numbers to look on the left.', required=False)
parser.add_argument('-mi','--minimum_peak_intensity', type=int, default=250, help='Minimum peak intensity to search.', required=False)
parser.add_argument('-mp','--minimum_peaks_nearby', type=int, default=3, help='Minimum peaks nearby to give to Hardklor.', required=False)

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

DELTA_MZ = 1.003355
MZ_COMPARISON_TOLERANCE = 0.01

for f in glob('..\\..\\data\\hk\\spectra*'):
    os.remove(f)

clusters = []
start_run = time.time()
for frame_id in range(args.frame_lower, args.frame_upper+1):
    start_frame = time.time()
    cluster_id = 1
    peak_id = 1
    # Get all the peaks for this frame
    peaks_df = pd.read_sql_query("select peak_id,centroid_mz,centroid_scan,intensity_sum,scan_upper,scan_lower from peaks where frame_id={} order by peak_id asc;"
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
        mgf_file_name = "..\\..\\data\\hk\\spectra-frame-{}-peak-{}.mgf".format(frame_id, peak_id)
        hk_file_name = "..\\..\\data\\hk\\spectra-frame-{}-peak-{}.hk".format(frame_id, peak_id)
        tmp_mgf_file_name = "..\\..\\data\\hk\\spectra.mgf"
        tmp_hk_file_name = "..\\..\\data\\hk\\spectra.hk"

        peak_mz = peaks_v[max_intensity_index][1]
        peak_scan = peaks_v[max_intensity_index][2]
        peak_intensity = int(peaks_v[max_intensity_index][3])
        if peak_intensity < args.minimum_peak_intensity:
            print "Reached minimum peak intensity - exiting."
            break
        peak_scan_upper = int(peaks_v[max_intensity_index][4])
        peak_scan_lower = int(peaks_v[max_intensity_index][5])
        peak_indices = np.where((peaks_v[:,1] <= peak_mz + DELTA_MZ*args.isotope_number_right) & (peaks_v[:,1] >= peak_mz - DELTA_MZ*args.isotope_number_left) & (peaks_v[:,2] >= peak_scan_lower) & (peaks_v[:,2] <= peak_scan_upper))[0]
        peaks_nearby = peaks_v[peak_indices]
        peaks_nearby_sorted = peaks_nearby[np.argsort(peaks_nearby[:,1])]
        print("found {} peaks nearby".format(len(peak_indices)))

        if len(peak_indices) >= args.minimum_peaks_nearby:
        
            # Write out the spectrum for this peak's neighbourhood
            spectrum = {}
            spectrum["m/z array"] = peaks_nearby_sorted[:,1]
            spectrum["intensity array"] = peaks_nearby_sorted[:,3].astype(int)
            params = {}
            params["TITLE"] = "Frame ID {}, peak ID {}".format(frame_id, peak_id)
            params["PEPMASS"] = peak_mz
            params["PEAK_INTENSITY"] = peak_intensity
            params["PEAK_WEIGHTED_AVERAGE_MZ"] = peak_mz
            params["PEAK_WEIGHTED_AVERAGE_SCAN"] = peak_scan
            params["PEAK_SCAN_UPPER"] = peak_scan_upper
            params["PEAK_SCAN_LOWER"] = peak_scan_lower
            params["FRAME_ID"] = frame_id
            params["PEAK_ID"] = peak_id
            spectrum["params"] = params
            spectra.append(spectrum)

            # Write out the MGF file. We need to do this on the peak level because we need to know which 
            # peaks to cluster and remove from the frame
            if os.path.isfile(mgf_file_name):
                os.remove(mgf_file_name)
            if os.path.isfile(hk_file_name):
                os.remove(hk_file_name)
            mgf.write(output=mgf_file_name, spectra=spectra)

            # Run Hardklor on the file
            copyfile(mgf_file_name, tmp_mgf_file_name)
            subprocess.call(["C:\Hardklor-win64-2_3_0\Hardklor.exe", ".\Hardklor.conf"])
            copyfile(tmp_hk_file_name, hk_file_name)

            # Process the results file
            base_isotopic_mass = 0.0
            charge = 0
            analysis_window_min = 0.0
            analysis_window_max = 0.0
            cluster_quality = 0.0
            hk_df = pd.read_csv(hk_file_name, delimiter='\t', header=0, skiprows=0, skip_blank_lines=True, skipinitialspace=True,
                names=['line_type','monoisotopic_mass','charge','intensity','base_isotopic_mass','analysis_window','not_used','modifications','correlation_score'])
            hk_v = hk_df.values
            if len(hk_v) > 0:
                # Check whether there's a HK feature at peak_mz. The feature's quality score increases if it's the most intense feature
                peak_mz_match_indices = np.where((abs(hk_v[:,4] - peak_mz) < MZ_COMPARISON_TOLERANCE))[0]
                peak_mz_matches = hk_v[peak_mz_match_indices]
                if len(peak_mz_match_indices) > 0:
                    max_intensity_index = hk_v.argmax(axis=0)[3]
                    if max_intensity_index in peak_mz_match_indices:
                        print "Matched m/z and intensity"
                        peak_mz_match_index = max_intensity_index
                        cluster_quality = 1.0
                    else:
                        # Take the most intense of the peak_mz matches
                        print "Matched m/z only"
                        peak_mz_match_index = peak_mz_match_indices[peak_mz_matches.argmax(axis=0)[3]]
                        cluster_quality = 0.5

                    base_isotopic_mass = hk_v[peak_mz_match_index][4]
                    intensity = hk_v[peak_mz_match_index][3]
                    charge = hk_v[peak_mz_match_index][2]
                    analysis_window_min = float(hk_v[peak_mz_match_index][5].split('-')[0])
                    analysis_window_max = float(hk_v[peak_mz_match_index][5].split('-')[1])
                    print("base_isotope_peak: {}, charge: {}, analysis_window_min: {}, analysis_window_max: {}".format(base_isotopic_mass, charge, analysis_window_min, analysis_window_max))

                    # Pick out the peaks belonging to this cluster from the peaks nearby
                    # To the right...
                    for isotope_number in range(1,args.isotope_number_right+1):
                        mz = peak_mz + (isotope_number*DELTA_MZ/charge)
                        if mz <= analysis_window_max:
                            cluster_peak_indices_up = np.where((abs(peaks_nearby[:,1] - mz) < MZ_COMPARISON_TOLERANCE))[0]
                            if len(cluster_peak_indices_up) > 0:
                                cluster_peak_indices = np.append(cluster_peak_indices, peak_indices[cluster_peak_indices_up])
                    # To the left...
                    for isotope_number in range(1,args.isotope_number_left+1):
                        mz = peak_mz - (isotope_number*DELTA_MZ/charge)
                        if mz >= analysis_window_min:
                            cluster_peak_indices_down = np.where((abs(peaks_nearby[:,1] - mz) < MZ_COMPARISON_TOLERANCE))[0]
                            if len(cluster_peak_indices_down) > 0:
                                cluster_peak_indices = np.append(cluster_peak_indices, peak_indices[cluster_peak_indices_down])
                    # Reflect the clusters in the peak table of the database
                    print "cluster ID {}".format(cluster_id)
                    cluster_peak_indices = np.unique(cluster_peak_indices)
                    for p in cluster_peak_indices:
                        p_id = int(peaks_v[p][0])
                        print p_id
                        # Update the peaks in the peaks table with their cluster ID
                        values = (cluster_id, frame_id, p_id)
                        c.execute("update peaks set cluster_id=? where frame_id=? and peak_id=?", values)
                    clusters.append((frame_id, cluster_id, charge, peak_id, cluster_quality))
                    cluster_id += 1
                else:
                    print "FEATURE MISMATCH: Could not find a match for this peak in Hardklor's features."
            else:
                print "Hardklor made no suggestions for this peak."
        else:
            print "Found less than {} peaks nearby - skipping Hardklor".format(args.minimum_peaks_nearby)

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
