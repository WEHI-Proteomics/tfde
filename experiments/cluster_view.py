import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sqlite3
import os.path
import numpy as np


parser = argparse.ArgumentParser(description='Visualise how the clustering algorithm works.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-f','--frame_id', type=int, help='The frame to display.', required=True)
parser.add_argument('-c','--cluster_id', type=int, help='The cluster to display.', required=True)

args = parser.parse_args()

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)

cluster_detect_info_df = pd.read_sql_query("select item,value from cluster_detect_info;")
mz_std_dev = int(cluster_detect_info_df[cluster_detect_info_df.item == 'mz_std_dev'].values[0][1])
scan_std_dev = int(cluster_detect_info_df[cluster_detect_info_df.item == 'scan_std_dev'].values[0][1])
isotope_number_left = int(cluster_detect_info_df[cluster_detect_info_df.item == 'isotope_number_left'].values[0][1])
isotope_number_right = int(cluster_detect_info_df[cluster_detect_info_df.item == 'isotope_number_right'].values[0][1])
scan_lower = int(cluster_detect_info_df[cluster_detect_info_df.item == 'scan_lower'].values[0][1])



		f = plt.figure()
		ax1 = f.add_subplot(111)
		ax2 = ax1.twinx()
		plt.title("Database {}, frame {}, peak {}, cluster {}".format(args.database_name, args.frame_id, args.peak_id, cluster_peaks_df.cluster_id.values[0]))
		ax1.set_xlabel('m/z')
		ax1.set_ylabel('Actual intensity')
		ax2.set_ylabel('Theoretical intensity')
		ax1.set_ylim(ymin=0.0-mgf_df.intensity.max()*0.01, ymax=mgf_df.intensity.max()+mgf_df.intensity.max()*0.01)
		ax2.set_ylim(ymin=0.0-hk_df.intensity.max()*0.01, ymax=hk_df.intensity.max()+hk_df.intensity.max()*0.01)
		plt.plot((analysis_window_min, analysis_window_min), (0, hk_df.intensity.max()), 'b-')
		plt.plot((analysis_window_max, analysis_window_max), (0, hk_df.intensity.max()), 'b-')
		for index, row in hk_df.iterrows():
		    ax2.annotate(row.charge, xy=(row.base_isotopic_mass,row.intensity), xytext=(6,3), textcoords='offset points', color='blue')
		for index, row in cluster_peaks_df.iterrows():
		    ax1.annotate(int(row.peak_id), xy=(row.centroid_mz,row.intensity_sum), xytext=(6,-10), textcoords='offset points', color='red')
		ax1.plot(mgf_df.mz, mgf_df.intensity, 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=8, alpha=0.8)
		ax1.plot(cluster_peaks_df.centroid_mz, cluster_peaks_df.intensity_sum, 'x', markerfacecolor='red', markeredgecolor='red', markeredgewidth=2.0, markersize=8, alpha=0.6)
		ax2.plot(hk_df.base_isotopic_mass, hk_df.intensity, 'o', markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.0, markersize=5, alpha=1.0)
		plt.margins(0.02)
		plt.show()
