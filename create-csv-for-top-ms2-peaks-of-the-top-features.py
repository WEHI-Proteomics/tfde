from __future__ import print_function
import pandas as pd
import sqlite3
import numpy as np
import os
import argparse
import glob

# Write out the top peaks for the top features
# python -u ./otf-peak-detect/create-csv-for-top-ms2-peaks-of-the-top-features.py -cdb ./UPS2_allion/UPS2_allion.sqlite -db ./UPS2_allion/UPS2_allion-features-1-1097.sqlite -nrtd -0.25 -prtd 0.25 -nsd -4.0 -psd 0.1 -mnp 500 -of ./top-peaks-features-1-1097-nrtd--0.25-prtd-0.25-nsd-4.0-psd-0.1-mnp-500.csv

parser = argparse.ArgumentParser(description='Extract a CSV containing the top ms2 peaks for the top features.')
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-dbn','--database_base_name', type=str, help='The base name of the destination databases.', required=True)
parser.add_argument('-nrtd','--negative_rt_delta_tolerance', type=float, help='The negative RT delta tolerance.', required=True)
parser.add_argument('-prtd','--positive_rt_delta_tolerance', type=float, help='The positive RT delta tolerance.', required=True)
parser.add_argument('-nsd','--negative_scan_delta_tolerance', type=float, help='The negative scan delta tolerance.', required=True)
parser.add_argument('-psd','--positive_scan_delta_tolerance', type=float, help='The positive scan delta tolerance.', required=True)
parser.add_argument('-mnp','--maximum_number_of_peaks_per_feature', type=int, help='The maximum number of peaks per feature.', required=True)
args = parser.parse_args()

converted_database_name = "{}/{}.sqlite".format(args.data_directory, args.database_base_name)
feature_database_root = "{}/{}-features".format(args.data_directory, args.database_base_name)

db_conn = sqlite3.connect(converted_database_name)
features_df = pd.read_sql_query("select feature_id from features order by feature_id ASC", db_conn)
db_conn.close()

feature_db_list = glob.glob("{}-*.sqlite".format(feature_database_root))

for feature_db in feature_db_list:
    print("processing {}".format(feature_db))
    db_conn = sqlite3.connect(feature_db)
    src_c = db_conn.cursor()
    src_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 ON peak_correlation (feature_id, rt_distance, scan_distance)")

    output_filename = "{}-nrtd-{}-prtd-{}-nsd-{}-psd-{}-mnp-{}.csv".format(feature_db.split(".sqlite")[0], args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature)
    if os.path.isfile(args.output_filename):
        os.remove(args.output_filename)
    
    for idx in range(len(features_df)):
        feature_id = features_df.loc[idx].feature_id
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
