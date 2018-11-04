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

db_conn = sqlite3.connect(args.feature_region_database)
feature_list_df = pd.read_sql_query("select * from feature_list", db_conn)
db_conn.close()

if len(feature_list_df) > 0:
    print("Setting up indexes")
    db_conn = sqlite3.connect(args.feature_region_database)
    db_conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 ON peak_correlation (feature_id)")
    db_conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_2 ON peak_correlation (feature_id, rt_delta, scan_delta)")

    db_conn.cursor().execute("DROP TABLE IF EXISTS ms2_peaks_within_window")
    db_conn.close()

    feature_id_lower = int(feature_list_df.feature_id.min())
    feature_id_upper = int(feature_list_df.feature_id.max())

    hk_processes = []

    for feature_list_idx in range(0,len(feature_list_df)):
        feature_id = feature_list_df.loc[feature_list_idx].feature_id.astype(int)
        charge_state = feature_list_df.loc[feature_list_idx].charge_state.astype(int)
        monoisotopic_mass = feature_list_df.loc[feature_list_idx].monoisotopic_mass
        cluster_summed_intensity = feature_list_df.loc[feature_list_idx].feature_summed_intensity.astype(int)
        retention_time_secs = feature_list_df.loc[feature_list_idx].feature_centroid_rt

        print("Generating Hardklor file for feature {} ({}% complete)".format(feature_id, round(float(feature_id-feature_id_lower)/(feature_id_upper-feature_id_lower+1)*100,1)))

        # get all the precursors for this feature
        db_conn = sqlite3.connect(args.feature_region_database)
        precursors_df = pd.read_sql_query("select distinct(precursor_id) from feature_isolation_matches where feature_id={}".format(feature_id), db_conn)
        db_conn.close()

        for precursor_idx in range(len(precursors_df)):
            precursor_id = precursors_df.loc[precursor_idx].precursor_id.astype(int)

            # get all the ms2 peaks for this feature
            db_conn = sqlite3.connect(args.feature_region_database)
            ms2_peaks_df = pd.read_sql_query("select * from ms2_peaks where feature_id={} and precursor={} order by peak_id ASC".format(feature_id, precursor_id), db_conn)
            db_conn.close()

            if len(ms2_peaks_df) > 0:
                pairs_df = ms2_peaks_df[['centroid_mz', 'intensity']].copy().sort_values(by=['intensity'], ascending=False)

                # Write out the spectrum
                spectra = []
                spectrum = {}
                spectrum["m/z array"] = pairs_df.centroid_mz.values
                spectrum["intensity array"] = pairs_df.intensity.values
                params = {}
                params["TITLE"] = "RawFile: {} Index: 1 precursor: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(args.features_database).split('.')[0], precursor_id, charge_state, cluster_summed_intensity, feature_id, retention_time_secs)
                params["INSTRUMENT"] = "ESI-QUAD-TOF"
                params["PEPMASS"] = "{} {}".format(round(monoisotopic_mass,6), cluster_summed_intensity)
                params["CHARGE"] = "{}+".format(charge_state)
                params["RTINSECONDS"] = "{}".format(retention_time_secs)
                params["SCANS"] = "{}".format(int(retention_time_secs))
                spectrum["params"] = params
                spectra.append(spectrum)

                mgf_filename = "{}/feature-{}-precursor-{}.mgf".format(mgf_directory, feature_id, precursor_id)
                hk_filename = "{}/feature-{}-precursor-{}.hk".format(hk_directory, feature_id, precursor_id)
                header_filename = "{}/feature-{}-precursor-{}.txt".format(search_headers_directory, feature_id, precursor_id)

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
                hk_processes.append("./hardklor/hardklor -cmd -instrument TOF -resolution 40000 -centroided 1 -ms_level 2 -algorithm Version2 -charge_algorithm Senko -charge_min 1 -charge_max {} -correlation {} -mz_window 5.25 -sensitivity 2 -depth 2 -max_features 12 -distribution_area 1 -xml 0 {} {}".format(charge_state, args.minimum_peak_correlation, mgf_filename, hk_filename))
            else:
                print("There were no ms2 peaks in the ms2_peaks table for feature {} and precursor {}.".format(feature_id, precursor_id))

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
