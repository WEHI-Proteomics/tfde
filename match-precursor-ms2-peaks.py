import pandas as pd
import argparse
import numpy as np
import sqlite3
import time

#
# python -u ./otf-peak-detect/match-precursor-ms2-peaks.py -db /media/data-drive/Hela_20A_20R_500.sqlite -fl 1 -fu 100000
#

parser = argparse.ArgumentParser(description='Find the precursor\'s base peak in ms2.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=True)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=True)
parser.add_argument('-ppm','--mz_tolerance_ppm', type=int, default=2, help='m/z matching tolerance in PPM.', required=False)
args = parser.parse_args()

db_conn = sqlite3.connect(args.database_name)
db_c = db_conn.cursor()

if args.feature_id_lower is None:
    feature_id_range_df = pd.read_sql_query("select min(feature_id) from feature_base_peaks", db_conn)
    args.feature_id_lower = feature_id_range_df.loc[0][0]
    print("feature_id_lower set to {} from the data".format(args.feature_id_lower))

if args.feature_id_upper is None:
    feature_id_range_df = pd.read_sql_query("select max(feature_id) from feature_base_peaks", db_conn)
    args.feature_id_upper = feature_id_range_df.loc[0][0]
    print("feature_id_upper set to {} from the data".format(args.feature_id_upper))

# Store the arguments as metadata in the database for later reference
precursor_ms2_peak_match_info = []
for arg in vars(args):
    precursor_ms2_peak_match_info.append((arg, getattr(args, arg)))

print("Setting up tables and indexes")
db_c.execute("DROP TABLE IF EXISTS precursor_ms2_peak_matches")
db_c.execute("DROP TABLE IF EXISTS precursor_ms2_peak_matches_info")
db_c.execute("CREATE TABLE precursor_ms2_peak_matches (feature_id INTEGER, base_peak_id INTEGER, ms2_peak_id INTEGER, mz_delta REAL, scan_delta REAL, PRIMARY KEY (feature_id, base_peak_id))")
db_c.execute("CREATE TABLE precursor_ms2_peak_matches_info (item TEXT, value TEXT)")

start_run = time.time()

print("Loading the MS1 base peaks")
features_df = pd.read_sql_query("select feature_id,base_peak_id from feature_base_peaks where feature_id >= {} and feature_id <= {} order by feature_id ASC;".format(args.feature_id_lower, args.feature_id_upper), db_conn)
print("found features {}-{}".format(np.min(features_df.feature_id), np.max(features_df.feature_id)))

peak_matches = []

print("Finding precursor base peaks in ms2")
for feature_ids_idx in range(0,len(features_df)):
    feature_id = features_df.loc[feature_ids_idx].feature_id.astype(int)
    base_peak_id = features_df.loc[feature_ids_idx].base_peak_id.astype(int)
    print("Matching the base peak for feature {}".format(feature_id))

    ms1_peak_df = pd.read_sql_query("select * from ms1_feature_region_peaks where feature_id={} and peak_id={}".format(feature_id,base_peak_id), db_conn)
    base_centroid_mz = ms1_peak_df.loc[0].centroid_mz.astype(float)
    base_centroid_scan = ms1_peak_df.loc[0].centroid_scan.astype(float)

    # read all the ms2 peaks for this feature
    ms2_peaks_df = pd.read_sql_query("select * from ms2_peaks where feature_id={}".format(feature_id), db_conn)

    # find the matching ms2 peaks within tolerance
    ms2_peaks_df['mz_delta'] = abs(ms2_peaks_df.centroid_mz - base_centroid_mz)
    indexes_to_drop = ms2_peaks_df.mz_delta > (args.mz_tolerance_ppm * 10**-6 * base_centroid_mz)
    ms2_peaks_df.drop(ms2_peaks_df.index[indexes_to_drop], inplace=True)
    ms2_peaks_df.sort_values(by='mz_delta', ascending=True, inplace=True)
    ms2_peaks_df.reset_index(drop=True, inplace=True)

    if len(ms2_peaks_df) > 0:
        ms2_peaks_df['scan_delta'] = abs(ms2_peaks_df.centroid_scan - base_centroid_scan)
        ms2_peak_id = ms2_peaks_df.peak_id.loc[0]
        mz_delta = ms2_peaks_df.mz_delta.loc[0]
        scan_delta = ms2_peaks_df.scan_delta.loc[0]
    else:
        ms2_peak_id = 0
        mz_delta = 0.0
        scan_delta = 0.0

    peak_matches.append((feature_id, base_peak_id, ms2_peak_id, mz_delta, scan_delta))

print("Writing out the peak matches")
db_c.executemany("INSERT INTO precursor_ms2_peak_matches VALUES (?, ?, ?, ?, ?)", peak_matches)

stop_run = time.time()
print("{} seconds to match peaks for features {} to {}".format(stop_run-start_run, args.feature_id_lower, args.feature_id_upper))

# write out the processing info
precursor_ms2_peak_match_info.append(("run processing time (sec)", stop_run-start_run))
precursor_ms2_peak_match_info.append(("processed", time.ctime()))

precursor_ms2_peak_match_info_entry = []
precursor_ms2_peak_match_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in precursor_ms2_peak_match_info)))

db_c.executemany("INSERT INTO precursor_ms2_peak_matches_info VALUES (?, ?)", precursor_ms2_peak_match_info_entry)

db_conn.commit()
db_conn.close()
