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
parser.add_argument('-fdb','--features_database', type=str, help='The name of the features database.', required=True)
parser.add_argument('-bfn','--base_mgf_filename', type=str, help='The base name of the MGF.', required=True)
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-mpc','--minimum_peak_correlation', type=float, default=0.6, help='Process ms2 peaks with at least this much correlation with the feature''s ms1 base peak.', required=False)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=True)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=True)
args = parser.parse_args()

mgf_directory = "{}/mgf".format(args.data_directory)
hk_directory = "{}/hk".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)
output_directory = "{}/search".format(mgf_directory)

# make sure the output directory exist
if not os.path.exists(mgf_directory):
    os.makedirs(mgf_directory)    
if not os.path.exists(hk_directory):
    os.makedirs(hk_directory)    
if not os.path.exists(search_headers_directory):
    os.makedirs(search_headers_directory)    
if not os.path.exists(output_directory):
    os.makedirs(output_directory)    

print("Setting up tables and indexes")
db_conn = sqlite3.connect(args.features_database)
# db_conn.cursor().execute("DROP TABLE IF EXISTS deconvoluted_ions")
db_conn.cursor().execute("CREATE TABLE IF NOT EXISTS deconvoluted_ions (feature_id INTEGER, minimum_correlation REAL, ion_id INTEGER, mz REAL, intensity INTEGER, PRIMARY KEY (feature_id, minimum_correlation, ion_id))")
db_conn.cursor().execute("delete from deconvoluted_ions where minimum_correlation={}".format(args.minimum_peak_correlation))
db_conn.commit()
db_conn.close()

db_conn = sqlite3.connect(args.features_database)
feature_ids_df = pd.read_sql_query("select distinct(feature_id) from peak_correlation", db_conn)
db_conn.close()

mgf_filename = "{}/{}-search-correlation-{}-features-{}-{}.mgf".format(output_directory, args.base_mgf_filename, args.minimum_peak_correlation, args.feature_id_lower, args.feature_id_upper)
if os.path.isfile(mgf_filename):
    os.remove(mgf_filename)

deconvoluted_ions = []

for feature_ids_idx in range(0,len(feature_ids_df)):
    feature_id = feature_ids_df.loc[feature_ids_idx].feature_id.astype(int)
    ion_id = 0
    print("Processing feature {}".format(feature_id))

    hk_filename = "{}/{}-feature-{}-correlation-{}.hk".format(hk_directory, args.base_mgf_filename, feature_id, args.minimum_peak_correlation)
    header_filename = "{}/{}-feature-{}-correlation-{}.txt".format(search_headers_directory, args.base_mgf_filename, feature_id, args.minimum_peak_correlation)

    # parse the Hardklor output to create the search MGF
    # see https://proteome.gs.washington.edu/software/hardklor/docs/hardklorresults.html
    hk_results_df = pd.read_table(hk_filename, skiprows=1, header=None, names=['monoisotopic_mass','charge','intensity','base_isotope_peak','analysis_window','deprecated','modifications','correlation'])
    if len(hk_results_df) > 0:
        fragments_df = hk_results_df[['monoisotopic_mass', 'intensity']].copy().sort_values(by=['monoisotopic_mass'], ascending=True)
        fragments_df.monoisotopic_mass += PROTON_MASS  # the monoisotopic_mass from Hardklor is the zero charge M, so we add the proton mass to get M+H

        # read the header for this feature
        with open(header_filename) as f:
            header_content = f.readlines()
        header_content = [x for x in header_content if not x == '\n']  # remove the empty lines

        # compile the fragments from the Hardklor deconvolution
        fragments = []
        for row in fragments_df.iterrows():
            index, data = row
            fragments.append("{} {}\n".format(round(data.monoisotopic_mass,4), data.intensity.astype(int)))
            ion_id += 1
            deconvoluted_ions.append((int(feature_id), float(args.minimum_peak_correlation), int(ion_id), round(data.monoisotopic_mass,4), int(data.intensity)))

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

# store the deconvoluted ions in the database
db_conn = sqlite3.connect(args.features_database)
db_conn.cursor().executemany("INSERT INTO deconvoluted_ions (feature_id, minimum_correlation, ion_id, mz, intensity) VALUES (?, ?, ?, ?, ?)", deconvoluted_ions)
db_conn.commit()
db_conn.close()
