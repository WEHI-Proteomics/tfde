from __future__ import print_function
import pandas as pd
import sqlite3
import numpy as np
import os
import argparse

# Write out the top peaks for the top features
DB_NAME = '/home/ubuntu/UPS2_allion/UPS2_allion-features-1-1097.sqlite'
CONV_DB_NAME = '/home/ubuntu/UPS2_allion/UPS2_allion.sqlite'

parser = argparse.ArgumentParser(description='Extract a CSV containing the top ms2 peaks for the top features.')
parser.add_argument('-nrtd','--negative_rt_delta_tolerance', type=float, help='The negative RT delta tolerance.', required=True)
parser.add_argument('-prtd','--positive_rt_delta_tolerance', type=float, help='The positive RT delta tolerance.', required=True)
parser.add_argument('-nsd','--negative_scan_delta_tolerance', type=float, help='The negative scan delta tolerance.', required=True)
parser.add_argument('-psd','--positive_scan_delta_tolerance', type=float, help='The positive scan delta tolerance.', required=True)
parser.add_argument('-mnf','--maximum_number_of_features', type=int, help='The maximum number of features.', required=True)
parser.add_argument('-mnp','--maximum_number_of_peaks_per_feature', type=int, help='The maximum number of peaks per feature.', required=True)
parser.add_argument('-of','--output_filename', type=str, help='The output CSV filename.', required=True)
args = parser.parse_args()

db_conn = sqlite3.connect(DB_NAME)
src_c = db_conn.cursor()
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 ON peak_correlation (feature_id, rt_distance, scan_distance)")
db_conn.close()

db_conn = sqlite3.connect(CONV_DB_NAME)
top_features_df = pd.read_sql_query("select feature_id from features order by feature_id ASC limit {}".format(args.maximum_number_of_features), db_conn)
db_conn.close()

db_conn = sqlite3.connect(DB_NAME)
if os.path.isfile(args.output_filename):
    os.remove(args.output_filename)

for idx in range(len(top_features_df)):
    feature_id = top_features_df.loc[idx].feature_id
    print("feature ID {}".format(feature_id), end="")
    peak_correlation_df = pd.read_sql_query("select * from peak_correlation where feature_id=={} and rt_distance >= {} and rt_distance <= {} and scan_distance >= {} and scan_distance <= {} order by rt_distance ASC limit {}".format(feature_id, args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature), db_conn)
    peak_correlation_df["feature_id-ms2_peak_id"] = peak_correlation_df.feature_id.astype(str) + '-' + peak_correlation_df.ms2_peak_id.astype(str)
    ms2_peaks_df = pd.read_sql_query("select feature_id,peak_id,centroid_mz,intensity from ms2_peaks where feature_id || '-' || peak_id in {}".format(tuple(peak_correlation_df["feature_id-ms2_peak_id"])), db_conn)
    df = pd.merge(ms2_peaks_df, peak_correlation_df, left_on=['feature_id','peak_id'], right_on=['feature_id','ms2_peak_id'])
    df.drop(['peak_id','correlation','feature_id-ms2_peak_id'], inplace=True, axis=1)
    df.rename(columns={'intensity': 'ms2_peak_intensity', 'rt_distance': 'rt_delta', 'scan_distance': 'scan_delta', 'centroid_mz': 'ms2_peak_centroid_mz'}, inplace=True)
    print(" - {} ms2 peaks".format(len(df)))
    # write the CSV
    if os.path.isfile(args.output_filename):
        df.to_csv(args.output_filename, mode='a', sep=',', index=False, header=False)
    else:
        df.to_csv(args.output_filename, mode='a', sep=',', index=False, header=True)

db_conn.close()
