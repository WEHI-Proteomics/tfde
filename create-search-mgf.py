import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import operator
import os.path
import argparse

PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

parser = argparse.ArgumentParser(description='A tree descent method for MS2 peak detection.')
parser.add_argument('-srdb','--summed_regions_database', type=str, help='The name of the summed regions database.', required=True)
parser.add_argument('-bfn','--base_mgf_filename', type=str, help='The base name of the MGF.', required=True)
parser.add_argument('-mgfd','--mgf_directory', type=str, default='./mgf', help='The MGF directory.', required=False)
parser.add_argument('-hkd','--hk_directory', type=str, default='./hk', help='The HK directory.', required=False)
parser.add_argument('-shd','--search_headers_directory', type=str, default='./mgf_headers', help='The directory for the headers used to build the search MGF.', required=False)
parser.add_argument('-mc','--minimum_correlation', type=float, default=0.6, help='Process ms2 peaks with at least this much correlation with the feature''s ms1 base peak.')
args = parser.parse_args()

db_conn = sqlite3.connect(args.summed_regions_database)
feature_ids_df = pd.read_sql_query("select distinct(feature_id) from peak_correlation", db_conn)
db_conn.close()

mgf_filename = "{}/{}-search-correlation-{}.mgf".format(args.mgf_directory, args.base_mgf_filename, args.minimum_correlation)
if os.path.isfile(mgf_filename):
    os.remove(mgf_filename)

for feature_ids_idx in range(0,len(feature_ids_df)):
    feature_id = feature_ids_df.loc[feature_ids_idx].feature_id.astype(int)
    print("Processing feature {}".format(feature_id))

    hk_filename = "{}/{}-feature-{}-correlation-{}.hk".format(args.hk_directory, args.base_mgf_filename, feature_id, args.minimum_correlation)
    header_filename = "{}/{}-feature-{}-correlation-{}.txt".format(args.search_headers_directory, args.base_mgf_filename, feature_id, args.minimum_correlation)

    # parse the Hardklor output to create the search MGF
    # see https://proteome.gs.washington.edu/software/hardklor/docs/hardklorresults.html
    hk_results_df = pd.read_table(hk_filename, skiprows=1, header=None, names=['monoisotopic_mass','charge','intensity','base_isotope_peak','analysis_window','deprecated','modifications','correlation'])
    fragments_df = hk_results_df[['monoisotopic_mass', 'intensity']].copy().sort_values(by=['monoisotopic_mass'], ascending=True)
    fragments_df.monoisotopic_mass += PROTON_MASS  # the monoisotopic_mass from Hardklor is the zero charge M, so we add the proton mass to get M+H

    # read the header for this feature
    with open(header_filename) as f:
        header_content = f.readlines()
    header_content = header_content[:-1]  # remove the last line

    # compile the fragments from the Hardklor deconvolution
    fragments = []
    for row in fragments_df.iterrows():
        index, data = row
        fragments.append("{} {}\n".format(round(data.monoisotopic_mass,4), data.intensity.astype(int)))

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
