import pandas as pd
import sqlite3
import numpy as np
import os
import argparse

# Write out the top peaks for the top features
DB_NAME = '/home/ubuntu/UPS2_allion/UPS2_allion-features-1-1097.sqlite'
CONV_DB_NAME = '/home/ubuntu/UPS2_allion/UPS2_allion.sqlite'

parser = argparse.ArgumentParser(description='Extract a CSV containing the top ms2 peaks for the top features.')
parser.add_argument('-mrtd','--maximum_rt_delta_tolerance', type=float, help='The maximum RT delta tolerance.', required=True)
parser.add_argument('-msd','--maximum_scan_delta_tolerance', type=float, help='The maximum scan delta tolerance.', required=True)
parser.add_argument('-mnf','--maximum_number_of_features', type=int, help='The maximum number of features.', required=True)
parser.add_argument('-mnp','--maximum_number_of_peaks_per_feature', type=int, help='The maximum number of peaks per feature.', required=True)
parser.add_argument('-of','--output_filename', type=str, help='The output CSV filename.', required=True)
args = parser.parse_args()

db_conn = sqlite3.connect(CONV_DB_NAME)
top_features_df = pd.read_sql_query("select feature_id from features order by feature_id ASC limit {}".format(args.maximum_number_of_features), db_conn)
db_conn.close()

db_conn = sqlite3.connect(DB_NAME)
if os.path.isfile(args.output_filename):
    os.remove(args.output_filename)
    
for idx in range(len(top_features_df)):
    feature_id = top_features_df.loc[idx].feature_id
    print("feature ID {}".format(feature_id))
    df_1 = pd.read_sql_query("select feature_id,peak_id,centroid_mz from ms2_peaks where feature_id || '-' || peak_id in (select feature_id || '-' || ms2_peak_id from peak_correlation where feature_id == {} and abs(rt_distance) <= {} and abs(scan_distance) <= {} order by ms2_peak_id limit {})".format(feature_id, args.maximum_rt_delta_tolerance, args.maximum_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature), db_conn)
    df_2 = pd.read_sql_query("select * from peak_correlation where feature_id=={} and abs(rt_distance) <= {} and abs(scan_distance) <= {} order by ms2_peak_id limit {}".format(feature_id, args.maximum_rt_delta_tolerance, args.maximum_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature), db_conn)
    df = pd.merge(df_1, df_2, left_on=['feature_id','peak_id'], right_on=['feature_id','ms2_peak_id'])
    df.drop(['peak_id','correlation'], inplace=True, axis=1)
    # write the CSV
    if os.path.isfile(args.output_filename):
        df.to_csv(args.output_filename, mode='a', sep=',', index=False, header=False)
    else:
        df.to_csv(args.output_filename, mode='a', sep=',', index=False, header=True)

db_conn.close()
