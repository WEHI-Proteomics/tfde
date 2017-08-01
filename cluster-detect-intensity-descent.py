import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import time
import sqlite3
import copy
import argparse
from pyteomics import mgf
import os.path
import csv
import subprocess


parser = argparse.ArgumentParser(description='A tree descent method for clustering peaks.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number.', required=True)
parser.add_argument('-sl','--scan_lower', type=int, default=0, help='The lower scan number.', required=False)
parser.add_argument('-su','--scan_upper', type=int, default=138, help='The upper scan number.', required=False)
parser.add_argument('-ir','--isotope_number_right', type=int, default=10, help='Isotope numbers to look on the right.', required=False)
parser.add_argument('-il','--isotope_number_left', type=int, default=5, help='Isotope numbers to look on the left.', required=False)

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
c.execute('''CREATE TABLE `clusters` ( `frame_id` INTEGER, `cluster_id` INTEGER, `charge_state` INTEGER, 'base_isotope_peak_id' INTEGER, PRIMARY KEY(`cluster_id`,`frame_id`) )''')
c.execute('''DROP INDEX IF EXISTS idx_clusters''')
c.execute('''CREATE INDEX idx_clusters ON clusters (frame_id,cluster_id)''')
c.execute("update peaks set cluster_id=0")
c.execute('''DROP TABLE IF EXISTS cluster_detect_info''')
c.execute('''CREATE TABLE cluster_detect_info (item TEXT, value TEXT)''')

DELTA_MZ = 1.003355
MZ_COMPARISON_TOLERANCE = 0.01
MGF_FILE_NAME = "\\temp\\spectra.mgf"
HK_FILE_NAME = "\\temp\\spectra.hk"

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
    while len(peaks_v) > 0:
        # print("{} peaks remaining.".format(len(peaks_v)))
        spectra = []
        cluster_peak_indices = np.empty(0, dtype=int)

        max_intensity_index = peaks_v.argmax(axis=0)[3]
        cluster_peak_indices = np.append(cluster_peak_indices, max_intensity_index)

        peak_id = int(peaks_v[max_intensity_index][0])
        print("peak id {}".format(peak_id))
        peak_mz = peaks_v[max_intensity_index][1]
        peak_intensity = int(peaks_v[max_intensity_index][3])
        peak_scan_upper = int(peaks_v[max_intensity_index][4])
        peak_scan_lower = int(peaks_v[max_intensity_index][5])
        peak_indices = np.where((peaks_v[:,1] <= peak_mz + DELTA_MZ*args.isotope_number_right) & (peaks_v[:,1] >= peak_mz - DELTA_MZ*args.isotope_number_left) & (peaks_v[:,2] >= peak_scan_lower) & (peaks_v[:,2] <= peak_scan_upper))[0]
        peaks_nearby = peaks_v[peak_indices]
        peaks_nearby_sorted = peaks_nearby[np.argsort(peaks_nearby[:,1])]
        print("found {} peaks nearby".format(len(peak_indices)))
        
        # Write out the spectrum for this peak's neighbourhood
        spectrum = {}
        spectrum["m/z array"] = peaks_nearby_sorted[:,1]
        spectrum["intensity array"] = peaks_nearby_sorted[:,3].astype(int)
        params = {}
        params["TITLE"] = "Frame {}, neighbourhood of peak {} ({} intensity, {} m/z, {}-{} scan)".format(frame_id, peak_id, peak_intensity, peak_mz, peak_scan_lower, peak_scan_upper)
        params["PEPMASS"] = "{}".format(peak_mz)
        spectrum["params"] = params
        spectra.append(spectrum)

        # Write out the MGF file. We need to do this on the peak level because we need to know which 
        # peaks to cluster and remove from the frame
        if os.path.isfile(MGF_FILE_NAME):
            os.remove(MGF_FILE_NAME)
        if os.path.isfile(HK_FILE_NAME):
            os.remove(HK_FILE_NAME)
        mgf.write(output=MGF_FILE_NAME, spectra=spectra)

        # Run Hardklor on the file
        subprocess.call("C:\Hardklor-win64-2_3_0\Hardklor.exe .\Hardklor.conf", shell=True)

        # Process the results file
        base_isotope_peak = 0.0
        charge = 0
        analysis_window_min = 0.0
        analysis_window_max = 0.0
        found = False
        with open(HK_FILE_NAME, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if (row[0] == 'P') & (abs(float(row[4]) - peak_mz) < MZ_COMPARISON_TOLERANCE):
                    base_isotope_peak = float(row[4])
                    charge = int(row[2])
                    analysis_window_min = float(row[5].split('-')[0])
                    analysis_window_max = float(row[5].split('-')[1])
                    print("base_isotope_peak: {}, charge: {}, analysis_window_min: {}, analysis_window_max: {}".format(base_isotope_peak, charge, analysis_window_min, analysis_window_max))
                    found = True    # This feature matches the peak
                    break

        if found:
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
            cluster_peak_indices = np.unique(cluster_peak_indices)
            for p in cluster_peak_indices:
                p_id = int(peaks_v[p][0])
                print p_id
                # Update the peaks in the peaks table with their cluster ID
                values = (cluster_id, frame_id, p_id)
                c.execute("update peaks set cluster_id=? where frame_id=? and peak_id=?", values)
            clusters.append((frame_id, cluster_id, charge, peak_id))
            cluster_id += 1
        else:
            print "No feature found for this peak."

        # remove the points we've processed from the frame
        print("removing peak ids {} - {} peaks remaining\n".format(peaks_v[cluster_peak_indices,0].astype(int), len(peaks_v)))
        peaks_v = np.delete(peaks_v, cluster_peak_indices, 0)

    stop_frame = time.time()
    print("{} seconds to process frame - {} peaks".format(stop_frame-start_frame, peak_id))

# Write out all the peaks to the database
c.executemany("INSERT INTO clusters VALUES (?, ?, ?, ?)", clusters)
stop_run = time.time()

cluster_detect_info.append(("run processing time (sec)", stop_run-start_run))
cluster_detect_info.append(("processed", time.ctime()))
c.executemany("INSERT INTO cluster_detect_info VALUES (?, ?)", cluster_detect_info)

source_conn.commit()
source_conn.close()
