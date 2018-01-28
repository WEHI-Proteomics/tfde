import sys
import numpy as np
import pandas as pd
import time
import sqlite3
from pyteomics import mgf
import operator
import os.path

FRAME_ID = 30000
FILE_NAME = "clusters.mgf"

# Formula from https://en.wikipedia.org/wiki/Gaussian_function
def gaussian(x, amplitude, peak, stddev):
    num = np.power((x-peak), 2.)
    den = 2. * np.power(stddev, 2.)
    return amplitude * np.exp(-num/den)

# For each candidate cluster in the frame, sum the intensities to derive an m/z versus intensity spectra

# Read the clusters from the database
sqlite_file = "\\temp\\frames-th-85-29900-30100-V4.sqlite"
conn = sqlite3.connect(sqlite_file)

# Get the clusters for this frame
spectra = []
clusters_df = pd.read_sql_query("select cluster_id,centroid_mz,centroid_scan from clusters where frame_id={};".format(FRAME_ID), conn)
for cluster_row in clusters_df.iterrows():
    cluster = cluster_row[1]
    print("Cluster {}".format(int(cluster.cluster_id)))
    cluster_points = []
    # Get the peaks in the cluster
    peaks_df = pd.read_sql_query("select peak_id from peaks where frame_id={} and cluster_id={};".format(FRAME_ID, cluster.cluster_id), conn)
    peak_id_list = ','.join(map(str, list(peaks_df.peak_id)))
    # Find the unique scan numbers for the cluster
    scan_numbers_df = pd.read_sql_query("select distinct scan from frames where frame_id={} and peak_id in ({}) order by scan;".format(FRAME_ID, peak_id_list), conn)
    # Find the unique m/z values for the cluster
    mz_df = pd.read_sql_query("select distinct mz from frames where frame_id={} and peak_id in ({}) order by mz;".format(FRAME_ID, peak_id_list), conn)
    mz_arr = mz_df.mz.values
    summed_intensities_arr = np.zeros(len(mz_arr))
    # Go through the scans and add up the intensity at the unique m/z points
    for scan_number_row in scan_numbers_df.iterrows():
        scan_number = scan_number_row[1]
        # Get all the points for this scan number, and calculate their contribution to the intensity for each unique m/z
        points_df = pd.read_sql_query("select mz,intensity from frames where frame_id={} and scan={} and peak_id in ({}) order by mz;".format(FRAME_ID, scan_number.scan, peak_id_list), conn)
        for point_row in points_df.iterrows():
            point = point_row[1]
            # Work out this point's contribution to intensity at each of the unique m/z values
            for i in range(0,len(mz_arr)):
                summed_intensities_arr[i] += gaussian(x=mz_arr[i], amplitude=point.intensity, peak=point.mz, stddev=0.1)

    # Write out the spectrum
    spectrum = {}
    spectrum["m/z array"] = mz_arr
    spectrum["intensity array"] = summed_intensities_arr
    params = {}
    params["TITLE"] = "Frame {}, Cluster {} (centroid: {} m/z, {} scan)".format(FRAME_ID, int(cluster.cluster_id), cluster.centroid_mz, int(cluster.centroid_scan))
    params["PEPMASS"] = "{}".format(cluster.centroid_mz)
    spectrum["params"] = params
    spectra.append(spectrum)

conn.close()

# Write out the MGF file
if os.path.isfile(FILE_NAME):
    os.remove(FILE_NAME)
mgf.write(output=FILE_NAME, spectra=spectra)
