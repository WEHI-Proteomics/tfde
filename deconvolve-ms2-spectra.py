# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import time
import sqlite3
from pyteomics import mgf
import operator
import os.path
import argparse
import os
from multiprocessing import Pool
import json

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

parser = argparse.ArgumentParser(description='Use Hardklor to deconvolve and deisotope the ms2 spectra.')
parser.add_argument('-fdb','--features_database', type=str, help='The name of the features database.', required=True)
parser.add_argument('-frdb','--feature_region_database', type=str, help='The name of the feature region database.', required=True)
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-mpc','--minimum_peak_correlation', type=float, default=0.6, help='Process ms2 peaks with at least this much correlation with the feature''s ms1 base peak.', required=False)
parser.add_argument('-fps','--frames_per_second', type=float, help='Effective frame rate.', required=True)
parser.add_argument('-nrtd','--negative_rt_delta_tolerance', type=float, default=-0.25, help='The negative RT delta tolerance.', required=False)
parser.add_argument('-prtd','--positive_rt_delta_tolerance', type=float, default=0.25, help='The positive RT delta tolerance.', required=False)
parser.add_argument('-nsd','--negative_scan_delta_tolerance', type=float, default=-4.0, help='The negative scan delta tolerance.', required=False)
parser.add_argument('-psd','--positive_scan_delta_tolerance', type=float, default=4.0, help='The positive scan delta tolerance.', required=False)
parser.add_argument('-mnp','--maximum_number_of_peaks_per_feature', type=int, default=50000, help='The maximum number of peaks per feature.', required=False)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

start_run = time.time()

mgf_directory = "{}/mgf".format(args.data_directory)
hk_directory = "{}/hk".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)

info.append(("mgf directory", mgf_directory))
info.append(("hk directory", hk_directory))
info.append(("search headers directory", search_headers_directory))

print("Setting up indexes")
db_conn = sqlite3.connect(args.feature_region_database)
db_conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 ON peak_correlation (feature_id)")
db_conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_2 ON peak_correlation (feature_id, rt_delta, scan_delta)")

db_conn.cursor().execute("DROP TABLE IF EXISTS ms2_peaks_within_window")
db_conn.close()

db_conn = sqlite3.connect(args.feature_region_database)
feature_list_df = pd.read_sql_query("select * from feature_list", db_conn)
db_conn.close()

feature_id_lower = int(feature_list_df.feature_id.min())
feature_id_upper = int(feature_list_df.feature_id.max())

hk_processes = []

for feature_list_idx in range(0,len(feature_list_df)):
    feature_id = feature_list_df.loc[feature_list_idx].feature_id.astype(int)
    charge_state = feature_list_df.loc[feature_list_idx].charge_state.astype(int)
    cluster_mz_centroid = feature_list_df.loc[feature_list_idx].cluster_mz_centroid
    cluster_summed_intensity = feature_list_df.loc[feature_list_idx].cluster_summed_intensity.astype(int)
    retention_time_secs = feature_list_df.loc[feature_list_idx].retention_time_secs
    base_frame_number = int(retention_time_secs * args.frames_per_second)

    print("Generating Hardklor file for feature {} ({}% complete)".format(feature_id, round(feature_id/feature_id_upper*100,1)))

    db_conn = sqlite3.connect(args.feature_region_database)
    # get the ms2 peaks
    peak_correlation_df = pd.read_sql_query("select * from peak_correlation where feature_id=={} and rt_delta >= {} and rt_delta <= {} and scan_delta >= {} and scan_delta <= {} order by ms2_peak_id ASC limit {}".format(feature_id, args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature), db_conn)
    peak_correlation_df["feature_id-ms2_peak_id"] = peak_correlation_df.feature_id.astype(str) + '-' + peak_correlation_df.ms2_peak_id.astype(str)
    # a bit of a hack to avoid the single-element tuple with trailing comma upsetting the SQL query
    if len(peak_correlation_df) == 1:
        peak_correlation_df = peak_correlation_df.append(peak_correlation_df)
    ms2_peaks_df = pd.read_sql_query("select feature_id,peak_id,centroid_mz,intensity from ms2_peaks where feature_id || '-' || peak_id in {}".format(tuple(peak_correlation_df["feature_id-ms2_peak_id"])), db_conn)
    # write out the ms2 peaks we are about to deconvolve and deisotope, for later matching with MSCypher output
    ms2_peaks_df.to_sql(name='ms2_peaks_within_window', con=db_conn, if_exists='append', index=False)
    db_conn.close()
    pairs_df = ms2_peaks_df[['centroid_mz', 'intensity']].copy().sort_values(by=['intensity'], ascending=False)

    # Write out the spectrum
    spectra = []
    spectrum = {}
    spectrum["m/z array"] = pairs_df.centroid_mz.values
    spectrum["intensity array"] = pairs_df.intensity.values
    params = {}
    params["TITLE"] = "RawFile: {} Index: 1318 precursor: 1 Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(args.features_database).split('.')[0], charge_state, cluster_summed_intensity, feature_id, retention_time_secs)
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(cluster_mz_centroid,6), cluster_summed_intensity)
    params["CHARGE"] = "{}+".format(charge_state)
    params["RTINSECONDS"] = "{}".format(retention_time_secs)
    params["SCANS"] = "{}".format(base_frame_number)
    spectrum["params"] = params
    spectra.append(spectrum)

    mgf_filename = "{}/feature-{}-correlation-{}.mgf".format(mgf_directory, feature_id, args.minimum_peak_correlation)
    hk_filename = "{}/feature-{}-correlation-{}.hk".format(hk_directory, feature_id, args.minimum_peak_correlation)
    header_filename = "{}/feature-{}-correlation-{}.txt".format(search_headers_directory, feature_id, args.minimum_peak_correlation)

    # write out the MGF file
    if os.path.isfile(mgf_filename):
        os.remove(mgf_filename)
    if os.path.isfile(hk_filename):
        os.remove(hk_filename)
    if os.path.isfile(header_filename):
        os.remove(header_filename)
    mgf.write(output=mgf_filename, spectra=spectra)

    # remove blank lines from the MGF file
    with open(mgf_filename, 'r') as file_handler:
        file_content = file_handler.readlines()
    file_content = [x for x in file_content if not x == '\n']
    with open(mgf_filename, 'w') as file_handler:
        file_handler.writelines(file_content)

    # write out the header with no fragment ions (with which to build the search MGF)
    spectra = []
    spectrum["m/z array"] = np.empty(0)
    spectrum["intensity array"] = np.empty(0)
    spectra.append(spectrum)
    mgf.write(output=header_filename, spectra=spectra)

    # append the Hardklor command to process it
    hk_processes.append("./hardklor/hardklor -cmd -instrument TOF -resolution 40000 -centroided 1 -ms_level 2 -algorithm Version2 -charge_algorithm Quick -charge_min 1 -charge_max {} -correlation {} -mz_window 5.25 -sensitivity 2 -depth 2 -max_features 12 -distribution_area 1 -xml 0 {} {}".format(charge_state, args.minimum_peak_correlation, mgf_filename, hk_filename))

# Set up the processing pool for Hardklor
print("running Hardklor...")
pool = Pool()
pool.map(run_process, hk_processes)

stop_run = time.time()

info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))

print("{} info: {}".format(parser.prog, info))

info_entry = []
info_entry.append(("{}".format(os.path.basename(args.feature_region_database).split('.')[0]), json.dumps(info)))

info_entry_df = pd.DataFrame(info_entry, columns=['item', 'value'])
db_conn = sqlite3.connect(args.feature_region_database)
info_entry_df.to_sql(name='deconvolve_ms2_spectra_info', con=db_conn, if_exists='replace', index=False)
db_conn.close()
