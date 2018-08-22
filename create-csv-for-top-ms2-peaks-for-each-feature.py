from __future__ import print_function
import pandas as pd
import sqlite3
import numpy as np
import os
import argparse
import glob
import sys

# Write out the top peaks for the top features
# python -u ./otf-peak-detect/create-csv-for-top-ms2-peaks-for-each-feature.py -db ./UPS2_allion/UPS2_allion-features-1-1097.sqlite -nrtd -0.25 -prtd 0.25 -nsd -4.0 -psd 4.0 -mnp 50000

parser = argparse.ArgumentParser(description='Extract a CSV containing the top ms2 peaks for all the features in the database.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-msc','--mscypher_output_name', type=str, help='The name of the MSCypher output file.', required=True)
parser.add_argument('-nrtd','--negative_rt_delta_tolerance', type=float, help='The negative RT delta tolerance.', required=True)
parser.add_argument('-prtd','--positive_rt_delta_tolerance', type=float, help='The positive RT delta tolerance.', required=True)
parser.add_argument('-nsd','--negative_scan_delta_tolerance', type=float, help='The negative scan delta tolerance.', required=True)
parser.add_argument('-psd','--positive_scan_delta_tolerance', type=float, help='The positive scan delta tolerance.', required=True)
parser.add_argument('-mnp','--maximum_number_of_peaks_per_feature', type=int, help='The maximum number of peaks per feature.', required=True)
args = parser.parse_args()

if not os.path.exists(args.mscypher_output_name):
    print("Error - the MSCypher output file does not exist. Exiting.")
    sys.exit(1)
else:
    mscypher_df = pd.read_table(args.mscypher_output_name, sep="\t", dtype="str", engine='python')

mscypher_subset_df = mscypher_df[["DiggerPepScore", "FeatureNum", "FragMZ", "FragInt", "FragError", "FragIonTypes", "FragPos", "FragCharge", "DeltaMassDa", "DeltaMassPPM"]].copy()
mscypher_subset_df = mscypher_subset_df.apply(pd.to_numeric, errors='ignore')

print("processing {}".format(args.database_name))
db_conn = sqlite3.connect(args.database_name)
src_c = db_conn.cursor()
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 ON peak_correlation (feature_id, rt_distance, scan_distance)")

output_filename = "{}-nrtd-{}-prtd-{}-nsd-{}-psd-{}-mnp-{}.csv".format(args.database_name.split(".sqlite")[0], args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature)

features_df = pd.read_sql_query("select distinct(feature_id) from peak_correlation order by feature_id ASC", db_conn)
for idx in range(len(features_df)):
    feature_id = features_df.loc[idx].feature_id
    print("feature ID {}".format(feature_id), end="")
    # merge the ms2 peaks with the peak correlation information
    peak_correlation_df = pd.read_sql_query("select * from peak_correlation where feature_id=={} and rt_distance >= {} and rt_distance <= {} and scan_distance >= {} and scan_distance <= {} order by ms2_peak_id ASC limit {}".format(feature_id, args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature), db_conn)
    peak_correlation_df["feature_id-ms2_peak_id"] = peak_correlation_df.feature_id.astype(str) + '-' + peak_correlation_df.ms2_peak_id.astype(str)
    ms2_peaks_df = pd.read_sql_query("select feature_id,peak_id,centroid_mz,intensity from ms2_peaks where feature_id || '-' || peak_id in {}".format(tuple(peak_correlation_df["feature_id-ms2_peak_id"])), db_conn)
    df = pd.merge(ms2_peaks_df, peak_correlation_df, left_on=['feature_id','peak_id'], right_on=['feature_id','ms2_peak_id'])
    df.drop(['peak_id','correlation','feature_id-ms2_peak_id'], inplace=True, axis=1)
    df.rename(columns={'intensity': 'ms2_peak_intensity', 'rt_distance': 'rt_delta', 'scan_distance': 'scan_delta', 'centroid_mz': 'ms2_peak_centroid_mz'}, inplace=True)
    # merge the ms2 peaks with the MSCypher output
    df = pd.merge(df, mscypher_subset_df, how='left', left_on=['feature_id'], right_on=['FeatureNum'])
    # write out the CSV
    print(" - {} ms2 peaks".format(len(df)))
    if idx == 0:
        df.to_csv(output_filename, mode='w', sep=',', index=False, header=True)
    else:
        df.to_csv(output_filename, mode='a', sep=',', index=False, header=False)

db_conn.close()
