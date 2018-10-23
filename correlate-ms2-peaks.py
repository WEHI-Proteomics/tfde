import pandas as pd
import argparse
import numpy as np
import sqlite3
import time
import os
import json

# Number of points either side of the base peak's maximum intensity to check for correlation
BASE_PEAK_CORRELATION_SIDE_POINTS = 3

#
# nohup python -u ./otf-peak-detect/correlate-ms2-peaks.py -db /media/data-drive/Hela_20A_20R_500-features-1-100000-random-1000-sf-1000.sqlite -fl 1 -fu 100000 > correlate-ms2-peaks.log 2>&1 &
#

# so we can use profiling without removing @profile
import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile


@profile
def main():
    parser = argparse.ArgumentParser(description='Calculate correlation between MS1 and MS2 peaks for features.')
    parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
    parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
    parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=True)
    parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=True)
    args = parser.parse_args()

    source_conn = sqlite3.connect(args.database_name)
    src_c = source_conn.cursor()
    src_c.execute("PRAGMA journal_mode = TRUNCATE")

    conv_db_conn = sqlite3.connect(args.converted_database_name)
    conv_c = conv_db_conn.cursor()

    # Store the arguments as metadata in the database for later reference
    info = []
    for arg in vars(args):
        info.append((arg, getattr(args, arg)))

    print("Setting up tables")
    src_c.execute("DROP TABLE IF EXISTS peak_correlation")
    src_c.execute("DROP TABLE IF EXISTS peak_correlation_info")
    src_c.execute("CREATE TABLE peak_correlation (feature_id INTEGER, ms1_scan_centroid REAL, ms1_rt_centroid REAL, ms2_peak_id INTEGER, ms2_scan_centroid REAL, ms2_rt_centroid REAL, scan_delta REAL, rt_delta REAL, correlation REAL, PRIMARY KEY (feature_id, ms2_peak_id))")
    src_c.execute("CREATE TABLE peak_correlation_info (item TEXT, value TEXT)")

    print("Setting up indexes")
    src_c.execute("CREATE INDEX IF NOT EXISTS idx_feature_list_1 ON feature_list (feature_id)")
    src_c.execute("CREATE INDEX IF NOT EXISTS idx_ms2_peaks_1 ON ms2_peaks (feature_id)")

    start_run = time.time()

    print("Loading the MS1 base peaks for the feature range")
    feature_list_df = pd.read_sql_query("select * from feature_list where feature_id >= {} and feature_id <= {} order by feature_id ASC;".format(args.feature_id_lower, args.feature_id_upper), source_conn)

    peak_correlation = []

    print("Finding peak correlations for features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
    for feature_ids_idx in range(0,len(feature_list_df)):
        feature_start_time = time.time()

        feature_id = feature_list_df.loc[feature_ids_idx].feature_id.astype(int)
        ms1_centroid_scan = feature_list_df.loc[feature_ids_idx].feature_centroid_scan
        ms1_centroid_rt = feature_list_df.loc[feature_ids_idx].feature_centroid_rt

        print("processing feature {} ({}% complete)".format(feature_id, round(float(feature_id-args.feature_id_lower)/(args.feature_id_upper-args.feature_id_lower+1)*100,1)))

        ############################################################################################
        # Process the ms2. We've already calculated the ms2 CofI, so now we just determine the delta 
        # compared to the ms1 precursor
        ############################################################################################

        # get the ms2 peak summary information for this feature
        ms2_peaks_df = pd.read_sql_query("select * from ms2_peaks where feature_id={} order by peak_id ASC".format(feature_id), source_conn)
        if len(ms2_peaks_df) > 0:
            print("{} ms2 peaks for feature {}".format(len(ms2_peaks_df), feature_id))

            # calculate the delta
            ms2_peaks_df["scan_delta"] = ms1_centroid_scan - ms2_peaks_df.cofi_scan
            ms2_peaks_df["rt_delta"] = ms1_centroid_rt - ms2_peaks_df.cofi_rt
            ms2_peaks_df["correlation"] = 0.0

            # store the 2D centroid delta for each of the feature's ms2 peaks
            for ms2_peak_idx in range(len(ms2_peaks_df)):
                ms2_peak_id = ms2_peaks_df.loc[ms2_peak_idx].peak_id.astype(int)
                ms2_centroid_scan = ms2_peaks_df.loc[ms2_peak_idx].cofi_scan
                ms2_centroid_rt = ms2_peaks_df.loc[ms2_peak_idx].cofi_rt
                scan_delta = ms2_peaks_df.loc[ms2_peak_idx].scan_delta
                rt_delta = ms2_peaks_df.loc[ms2_peak_idx].rt_delta
                correlation = ms2_peaks_df.loc[ms2_peak_idx].correlation
                peak_correlation.append((feature_id, ms1_centroid_scan, ms1_centroid_rt, ms2_peak_id, ms2_centroid_scan, ms2_centroid_rt, scan_delta, rt_delta, correlation))

        feature_stop_time = time.time()
        print("processed feature {} in {} seconds".format(feature_id, feature_stop_time-feature_start_time))

    if len(peak_correlation) > 0:
        print("Writing out the peak correlations for features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
        # feature_id, ms1_centroid_scan, ms1_centroid_rt, ms2_peak_id, ms2_centroid_scan, ms2_centroid_rt, scan_distance, rt_distance, correlation
        src_c.executemany("INSERT INTO peak_correlation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", peak_correlation)
    else:
        print("Found no peaks for feature range {}-{}".format(args.feature_id_lower, args.feature_id_upper))

    stop_run = time.time()

    # write out the processing info
    info.append(("run processing time (sec)", stop_run-start_run))
    info.append(("processed", time.ctime()))
    info.append(("processor", parser.prog))

    print("{} info: {}".format(parser.prog, info))

    info_entry = []
    info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), json.dumps(info)))

    src_c.executemany("INSERT INTO peak_correlation_info VALUES (?, ?)", info_entry)

    source_conn.commit()
    source_conn.close()

    conv_db_conn.close()

if __name__ == "__main__":
    main()
