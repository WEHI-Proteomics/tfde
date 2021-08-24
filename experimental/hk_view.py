import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sqlite3
import os.path
import numpy as np


parser = argparse.ArgumentParser(description='Display the mass spectra sent to Hardklor, and its best match.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-f','--frame_id', type=int, help='The frame to display.', required=True)
parser.add_argument('-p','--peak_id', type=int, help='The peak to display.', required=True)

args = parser.parse_args()

mgf_file_name = "z:\\spectra-frame-{}-peak-{}.mgf".format(args.frame_id, args.peak_id)
hk_file_name = "z:\\spectra-frame-{}-peak-{}.hk".format(args.frame_id, args.peak_id)

MZ_COMPARISON_TOLERANCE = 0.01

if os.path.isfile(mgf_file_name) and os.path.isfile(hk_file_name) and os.path.isfile(args.database_name):

	# Connect to the database file
	source_conn = sqlite3.connect(args.database_name)

	# Read the MGF file
	mgf_df = pd.read_csv(mgf_file_name, delimiter=' ', skiprows=10, skipfooter=2, 
	engine='python', na_filter=False, names=['mz','intensity',''])
	mgf_v = mgf_df.values
	# Read the HK file
	hk_df = pd.read_csv(hk_file_name, delimiter='\t', header=0, skiprows=0, skip_blank_lines=True, skipinitialspace=True,
	    names=['line_type','monoisotopic_mass','charge','intensity','base_isotopic_mass','analysis_window','not_used','modifications','correlation_score'])
	hk_v = hk_df.values

	if len(hk_v) > 0:

		peak_mz = mgf_v[mgf_v.argmax(axis=0)[1]][0]
		print("peak_mz from the MGF {}".format(peak_mz))
		peak_mz_match_indices = np.where((abs(hk_v[:,4] - peak_mz) < MZ_COMPARISON_TOLERANCE))[0]
		peak_mz_matches = hk_v[peak_mz_match_indices]
		print("peak_mz_match_indices {}".format(peak_mz_match_indices))
		max_intensity_index = hk_v.argmax(axis=0)[3]
		print("max_intensity_index {}".format(max_intensity_index))
		if max_intensity_index in peak_mz_match_indices:
		    print "Matched m/z and intensity"
		    peak_mz_match_index = max_intensity_index
		else:
		    # Take the most intense of the peak_mz matches
		    print "Matched m/z only"
		    peak_mz_match_index = peak_mz_match_indices[peak_mz_matches.argmax(axis=0)[3]]
		    print("peak_mz_match_index {}".format(peak_mz_match_index))


		intensity = hk_v[peak_mz_match_index][3]
		charge = hk_v[peak_mz_match_index][2]
		base_isotopic_mass = hk_v[peak_mz_match_index][4]
		analysis_window_min = hk_v[peak_mz_match_index][5].split('-')[0]
		analysis_window_max = hk_v[peak_mz_match_index][5].split('-')[1]
		print("base_isotope_peak: {}, charge: {}, analysis_window_min: {}, analysis_window_max: {}".format(base_isotopic_mass, charge, analysis_window_min, analysis_window_max))

		cluster_peaks_df = pd.read_sql_query("select peak_id,centroid_mz,intensity_sum,cluster_id from peaks where frame_id={} and cluster_id in (select cluster_id from peaks where frame_id={} and peak_id={}) order by peak_id asc;"
		    .format(args.frame_id, args.frame_id, args.peak_id), source_conn)

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
	else:
		print "Hardklor made no suggestions for this peak"

else:
	print("Error: could not find input files ({}, {}, {})".format(mgf_file_name, hk_file_name, args.database_name))
