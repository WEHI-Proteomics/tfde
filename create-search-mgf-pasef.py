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

PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

# nohup python -u ./otf-peak-detect/create-search-mgf.py -fdb './UPS2_allion/UPS2_allion-features-1-1097.sqlite' -bfn features-1-1097 -dbd ./UPS2_allion > search-mgf.log 2>&1 &

parser = argparse.ArgumentParser(description='A tree descent method for MS2 peak detection.')
parser.add_argument('-fdb','--features_database', type=str, help='The name of the features database.', required=True)
parser.add_argument('-bfn','--base_mgf_filename', type=str, help='The base name of the MGF.', required=True)
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-od','--output_directory', type=str, help='The directory for the output.', required=True)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

start_run = time.time()

hk_directory = "{}/hk".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)
output_directory = args.output_directory

info.append(("hk_directory", hk_directory))
info.append(("search_headers_directory", search_headers_directory))
info.append(("output_directory", output_directory))

try:
    db_conn = sqlite3.connect(args.features_database)
    feature_ids_df = pd.read_sql_query("select feature_id from feature_list", db_conn)
    db_conn.cursor().execute("DROP TABLE IF EXISTS deconvoluted_ions")
    db_conn.close()

    # delete the MGF if it already exists
    mgf_filename = "{}/{}-search.mgf".format(output_directory, args.base_mgf_filename)
    if os.path.isfile(mgf_filename):
        os.remove(mgf_filename)
    info.append(("mgf_filename", mgf_filename))

    for feature_ids_idx in range(0,len(feature_ids_df)):
        feature_id = feature_ids_df.loc[feature_ids_idx].feature_id.astype(int)
        print("Processing feature {}".format(feature_id))

        # get all the precursors for this feature
        db_conn = sqlite3.connect(args.feature_region_database)
        precursors_df = pd.read_sql_query("select * from feature_isolation_matches where feature_id={}".format(feature_id), db_conn)
        db_conn.close()

        for precursor_idx in range(len(precursors_df)):
            precursor_id = precursors_df.loc[precursor_idx].precursor_id.astype(int)

            hk_filename = "{}/feature-{}-precursor-{}.hk".format(hk_directory, feature_id, precursor_id)
            header_filename = "{}/feature-{}-precursor-{}.txt".format(search_headers_directory, feature_id, precursor_id)

            # parse the Hardklor output to create the search MGF
            # see https://proteome.gs.washington.edu/software/hardklor/docs/hardklorresults.html
            if os.path.isfile(hk_filename):
                hk_results_df = pd.read_table(hk_filename, skiprows=1, header=None, names=['monoisotopic_mass','charge','intensity','base_isotope_peak','analysis_window','deprecated','modifications','correlation'])
                if len(hk_results_df) > 0:
                    # rename the HK columns to be clearer
                    hk_results_df.rename(columns={'monoisotopic_mass':'hk_monoisotopic_mass', 'charge':'hk_charge', 'intensity':'hk_intensity', 'base_isotope_peak':'hk_base_isotope_peak'}, inplace=True)

                    # the monoisotopic_mass from Hardklor is the zero charge M, so we add the proton mass to get M+H
                    hk_results_df.hk_monoisotopic_mass += PROTON_MASS

                    # drop the columns we don't need
                    hk_results_df.drop(['analysis_window','deprecated','modifications'], inplace=True, axis=1)

                    # add the feature and precursor so we can match them up later if needed
                    hk_results_df['feature_id'] = feature_id
                    hk_results_df['precursor_id'] = precursor_id

                    # rearrange the column order to be a bit nicer
                    hk_results_df = hk_results_df[['feature_id','precursor_id','hk_monoisotopic_mass','hk_charge','hk_intensity','hk_base_isotope_peak']]

                    # write out the deconvolved and de-isotoped peaks reported by HK
                    db_conn = sqlite3.connect(args.features_database)
                    hk_results_df.to_sql(name='deconvoluted_ions', con=db_conn, if_exists='append', index=False)
                    db_conn.close()

                    fragments_df = hk_results_df[['hk_monoisotopic_mass', 'hk_intensity']].copy().sort_values(by=['hk_monoisotopic_mass'], ascending=True)

                    # read the header for this feature
                    with open(header_filename) as f:
                        header_content = f.readlines()
                    header_content = [x for x in header_content if not x == '\n']  # remove the empty lines

                    # compile the fragments from the Hardklor deconvolution
                    fragments = []
                    for row in fragments_df.iterrows():
                        index, data = row
                        fragments.append("{} {}\n".format(round(data.hk_monoisotopic_mass,4), data.hk_intensity_peak_id))

                    with open(mgf_filename, 'a') as file_handler:
                        # write the header
                        for item in header_content[:len(header_content)-1]:
                            file_handler.write("{}".format(item))
                        # write the fragments
                        for item in fragments:
                            file_handler.write("{}".format(item))
                        # close off the feature
                        for item in header_content[len(header_content)-1:]:
                            file_handler.write("{}".format(item))
                else:
                    print("Hardklor gave no results for feature {}".format(feature_id))
            else:
                print("Could not find the file {} for feature {}".format(hk_filename, feature_id))

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
