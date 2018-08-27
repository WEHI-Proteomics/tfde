import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import operator
import os.path
import argparse

PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

# nohup python -u ./otf-peak-detect/create-search-mgf.py -fdb './UPS2_allion/UPS2_allion-features-1-1097.sqlite' -bfn features-1-1097 -dbd ./UPS2_allion > search-mgf.log 2>&1 &

parser = argparse.ArgumentParser(description='A tree descent method for MS2 peak detection.')
parser.add_argument('-fdb','--features_database', type=str, help='The name of the features database.', required=True)
parser.add_argument('-bfn','--base_mgf_filename', type=str, help='The base name of the MGF.', required=True)
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-mpc','--minimum_peak_correlation', type=float, default=0.6, help='Process ms2 peaks with at least this much correlation with the feature''s ms1 base peak.', required=False)
args = parser.parse_args()

mgf_directory = "{}/mgf".format(args.data_directory)
hk_directory = "{}/hk".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)
output_directory = "{}/search".format(mgf_directory)

# make sure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)    

db_conn = sqlite3.connect(args.features_database)
feature_ids_df = pd.read_sql_query("select distinct(feature_id) from peak_correlation", db_conn)
db_conn.cursor().execute("DROP TABLE IF EXISTS deconvoluted_ions")
db_conn.close()

# delete the MGF if it already exists
mgf_filename = "{}/{}-search.mgf".format(output_directory, args.base_mgf_filename)
if os.path.isfile(mgf_filename):
    os.remove(mgf_filename)

for feature_ids_idx in range(0,len(feature_ids_df)):
    feature_id = feature_ids_df.loc[feature_ids_idx].feature_id.astype(int)
    ion_id = 0
    print("Processing feature {}".format(feature_id))

    hk_filename = "{}/feature-{}-correlation-{}.hk".format(hk_directory, feature_id, args.minimum_peak_correlation)
    header_filename = "{}/feature-{}-correlation-{}.txt".format(search_headers_directory, feature_id, args.minimum_peak_correlation)

    # parse the Hardklor output to create the search MGF
    # see https://proteome.gs.washington.edu/software/hardklor/docs/hardklorresults.html
    hk_results_df = pd.read_table(hk_filename, skiprows=1, header=None, names=['monoisotopic_mass','charge','intensity','base_isotope_peak','analysis_window','deprecated','modifications','correlation'])
    if len(hk_results_df) > 0:
        # the monoisotopic_mass from Hardklor is the zero charge M, so we add the proton mass to get M+H
        hk_results_df.monoisotopic_mass += PROTON_MASS
        # get the ms2 peaks for this feature within the extraction window
        db_conn = sqlite3.connect(args.features_database)
        ms2_peaks_df = pd.read_sql_query("select feature_id,peak_id,centroid_mz from ms2_peaks_within_window where feature_id={}".format(feature_id), db_conn)
        db_conn.close()
        # annotate the HK rows with the corresponding ms2 peak ID
        hk_results_df = pd.merge(hk_results_df, ms2_peaks_df, how='left', left_on=['base_isotope_peak'], right_on=['centroid_mz'])
        # write out the deconvolved and de-isotoped peaks from HK
        db_conn = sqlite3.connect(args.features_database)
        hk_results_df.to_sql(name='deconvoluted_ions', con=db_conn, if_exists='append', index=False)
        db_conn.close()
        # append the peak ID to the intensity
        hk_results_df['intensity_peak_id'] = hk_results_df['intensity'].astype(str) + "." + hk_results_df['peak_id'].map('{0:05d}'.format)

        fragments_df = hk_results_df[['monoisotopic_mass', 'intensity', 'intensity_peak_id']].copy().sort_values(by=['monoisotopic_mass'], ascending=True)

        # read the header for this feature
        with open(header_filename) as f:
            header_content = f.readlines()
        header_content = [x for x in header_content if not x == '\n']  # remove the empty lines

        # compile the fragments from the Hardklor deconvolution
        fragments = []
        for row in fragments_df.iterrows():
            index, data = row
            fragments.append("{} {}\n".format(round(data.monoisotopic_mass,4), data.intensity_peak_id))

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

print("writing out the search MGF to {}".format(mgf_filename))
