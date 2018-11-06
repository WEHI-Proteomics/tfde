import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import operator
import os.path
import argparse
import json
import traceback
import ms_deisotope

PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

# nohup python -u ./otf-peak-detect/create-search-mgf.py -fdb './UPS2_allion/UPS2_allion-features-1-1097.sqlite' -bfn features-1-1097 -dbd ./UPS2_allion > search-mgf.log 2>&1 &

parser = argparse.ArgumentParser(description='A tree descent method for MS2 peak detection.')
parser.add_argument('-fdb','--features_database', type=str, help='The name of the features database.', required=True)
parser.add_argument('-bfn','--base_mgf_filename', type=str, help='The base name of the MGF.', required=True)
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

start_run = time.time()

# set up the directories for processing
raw_mgf_directory = "{}/raw-mgf".format(args.data_directory)
search_mgf_directory = "{}/search-mgf".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)

info.append(("raw_mgf_directory", raw_mgf_directory))
info.append(("search_mgf_directory", search_mgf_directory))
info.append(("search_headers_directory", search_headers_directory))

try:
    db_conn = sqlite3.connect(args.features_database)
    feature_ids_df = pd.read_sql_query("select feature_id,charge_state from feature_list", db_conn)
    db_conn.cursor().execute("DROP TABLE IF EXISTS deconvoluted_ions")
    db_conn.close()

    for feature_ids_idx in range(0,len(feature_ids_df)):
        feature_id = feature_ids_df.loc[feature_ids_idx].feature_id.astype(int)
        feature_charge_state = feature_ids_df.loc[feature_ids_idx].charge_state.astype(int)
        print("Processing feature {}".format(feature_id))

        # get all the precursors for this feature
        db_conn = sqlite3.connect(args.features_database)
        precursors_df = pd.read_sql_query("select distinct(precursor_id) from feature_isolation_matches where feature_id={}".format(feature_id), db_conn)
        db_conn.close()

        for precursor_idx in range(len(precursors_df)):
            precursor_id = precursors_df.loc[precursor_idx].precursor_id.astype(int)

            raw_mgf_filename = "{}/feature-{}-precursor-{}.mgf".format(raw_mgf_directory, feature_id, precursor_id)
            search_mgf_filename = "{}/feature-{}-precursor-{}.mgf".format(search_mgf_directory, feature_id, precursor_id)
            header_filename = "{}/feature-{}-precursor-{}.txt".format(search_headers_directory, feature_id, precursor_id)

            reader = ms_deisotope.MSFileLoader(raw_mgf_filename)
            scan = next(reader)
            scan.pick_peaks().deconvolute(scorer=ms_deisotope.MSDeconVFitter(10), 
                                        averagine=ms_deisotope.peptide,
                                        charge_range=(1,feature_charge_state),
                                        truncate_after=0.8)

            deconvoluted_peaks = []
            for peak in scan.deconvoluted_peak_set:
                deconvoluted_peaks.append((peak.neutral_mass, peak.intensity))
            deconvoluted_peaks_df = pd.DataFrame(deconvoluted_peaks, columns=['neutral_mass','intensity'])

            # write out the deconvolved and de-isotoped peaks reported by ms_deisotope
            db_conn = sqlite3.connect(args.features_database)
            deconvoluted_peaks_df.to_sql(name='deconvoluted_ions', con=db_conn, if_exists='append', index=False)
            db_conn.close()

            fragments_df = deconvoluted_peaks_df.copy().sort_values(by=['neutral_mass'], ascending=True)

            # read the header for this feature
            with open(header_filename) as f:
                header_content = f.readlines()
            header_content = [x for x in header_content if not x == '\n']  # remove the empty lines

            # compile the fragments from the ms_deisotope deconvolution
            fragments = []
            for row in fragments_df.iterrows():
                index, data = row
                fragments.append("{} {}\n".format(round(data.neutral_mass,4), int(data.intensity)))

            # write out the individual search MGFs
            print("adding {} fragments to {}".format(len(fragments), search_mgf_filename))
            with open(search_mgf_filename, 'a') as file_handler:
                # write the header
                for item in header_content[:len(header_content)-1]:
                    file_handler.write("{}".format(item))
                # write the fragments
                for item in fragments:
                    file_handler.write("{}".format(item))
                # close off the feature
                for item in header_content[len(header_content)-1:]:
                    file_handler.write("{}".format(item))

    stop_run = time.time()

    info.append(("run processing time (sec)", stop_run-start_run))
    info.append(("processed", time.ctime()))
    info.append(("processor", parser.prog))

    print("{} info: {}".format(parser.prog, info))

    info_entry = []
    info_entry.append(("features {}".format(args.features_database), json.dumps(info)))

    info_entry_df = pd.DataFrame(info_entry, columns=['item', 'value'])
    db_conn = sqlite3.connect(args.features_database)
    info_entry_df.to_sql(name='search_mgf_info', con=db_conn, if_exists='replace', index=False)
    db_conn.close()

except Exception as e:
    print("Exception {} caught in {} for {}".format(traceback.format_exc(), parser.prog, info))
