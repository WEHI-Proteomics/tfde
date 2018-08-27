from __future__ import print_function
import pandas as pd
import sqlite3
import numpy as np
import os
import argparse
import glob
import sys

# Write out the top peaks for the top features
# python -u ./otf-peak-detect/create-csv-for-top-ms2-peaks-for-each-feature.py -msc UPS2_allion/MSCypher.txt -db ./UPS2_allion/UPS2_allion-features-1-1097.sqlite -nrtd -0.25 -prtd 0.25 -nsd -4.0 -psd 4.0 -mnp 50000

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

# convert the numeric fields and sort the features by their score
mscypher_df = mscypher_df.apply(pd.to_numeric, errors='ignore').sort_values(by=['FeatureNum','DiggerPepScore'], ascending=False)
# remove feature-duplicates from the MSC output, keeping the highest score
mscypher_df.drop_duplicates(subset=['FeatureNum'], keep='first', inplace=True)
# take the subset of fields we need to use for fragment annotation
msc_subset_df = mscypher_df[["DiggerPepScore", "FeatureNum", "FragMZ", "FragInt", "FragError", "FragIonTypes", "FragPos", "FragCharge", "DeltaMassDa", "DeltaMassPPM"]].copy()

print("processing {}".format(args.database_name))
db_conn = sqlite3.connect(args.database_name)
src_c = db_conn.cursor()
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 ON peak_correlation (feature_id, rt_distance, scan_distance)")

output_filename_ms2 = "{}-ms2peaks-nrtd-{}-prtd-{}-nsd-{}-psd-{}-mnp-{}.csv".format(args.database_name.split(".sqlite")[0], args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature)
output_filename_features = "{}-features-nrtd-{}-prtd-{}-nsd-{}-psd-{}-mnp-{}.csv".format(args.database_name.split(".sqlite")[0], args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature)

feature_ids_df = pd.read_sql_query("select distinct(feature_id) from peak_correlation order by feature_id ASC", db_conn)
for idx in range(len(feature_ids_df)):
    feature_id = feature_ids_df.loc[idx].feature_id
    print("feature ID {}".format(feature_id), end="")

    # merge the ms2 peaks with the peak correlation information
    peak_correlation_df = pd.read_sql_query("select * from peak_correlation where feature_id=={} and rt_distance >= {} and rt_distance <= {} and scan_distance >= {} and scan_distance <= {} order by ms2_peak_id ASC limit {}".format(feature_id, args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance, args.maximum_number_of_peaks_per_feature), db_conn)
    peak_correlation_df["feature_id-ms2_peak_id"] = peak_correlation_df.feature_id.astype(str) + '-' + peak_correlation_df.ms2_peak_id.astype(str)
    ms2_peaks_df = pd.read_sql_query("select feature_id,peak_id,centroid_mz,intensity from ms2_peaks_within_window where feature_id || '-' || peak_id in {}".format(tuple(peak_correlation_df["feature_id-ms2_peak_id"])), db_conn)
    ms2_peaks_df = pd.merge(ms2_peaks_df, peak_correlation_df, left_on=['feature_id','peak_id'], right_on=['feature_id','ms2_peak_id'])
    ms2_peaks_df.drop(['peak_id','correlation','feature_id-ms2_peak_id'], inplace=True, axis=1)
    ms2_peaks_df.rename(columns={'intensity': 'ms2_peak_intensity', 'rt_distance': 'rt_delta', 'scan_distance': 'scan_delta', 'centroid_mz': 'peak_centroid_mz'}, inplace=True)

    # break out the fragments reported by MSC for this feature
    if len(msc_subset_df[msc_subset_df.FeatureNum==feature_id]):
        msc_fragments = msc_subset_df[msc_subset_df.FeatureNum==feature_id].iloc[0]
        FragMZ = list(map(float, msc_fragments.FragMZ.split(';')))
        FragInt = list(map(float, msc_fragments.FragInt.split(';')))
        FragError = list(map(float, msc_fragments.FragError.split(';')))
        FragIonTypes = list(map(str, msc_fragments.FragIonTypes.split(';')))
        FragPos = list(map(int, msc_fragments.FragPos.split(';')))
        FragCharge = list(map(int, msc_fragments.FragCharge.split(';')))
        msc_fragments_df = pd.DataFrame(list(zip(FragMZ, FragInt, FragError, FragIonTypes, FragPos, FragCharge)), columns=['FragMZ', 'FragInt', 'FragError', 'FragIonTypes', 'FragPos', 'FragCharge'])
    else:
        msc_fragments_df = pd.DataFrame([], columns=['FragMZ', 'FragInt', 'FragError', 'FragIonTypes', 'FragPos', 'FragCharge'])

    # round the join column to match the ms2 peaks with the fragments reported by MSC
    msc_fragments_df["FragMZ_round"] = msc_fragments_df.FragMZ.round(3)
    ms2_peaks_df["centroid_mz_round"] = ms2_peaks_df.peak_centroid_mz.round(3)

    # match up the ms2 peaks with the fragments reported by MSC
    ms2_peaks_df = pd.merge(ms2_peaks_df, msc_fragments_df, how='left', left_on=['centroid_mz_round'], right_on=['FragMZ_round'])
    ms2_peaks_df.drop(['centroid_mz_round','FragMZ_round'], inplace=True, axis=1)
    ms2_peaks_df.rename(columns={'FragMZ': 'msc_FragMZ', 'FragInt': 'msc_FragInt', 'FragError': 'msc_FragError', 'FragIonTypes': 'msc_FragIonTypes', 'FragPos': 'msc_FragPos', 'FragCharge': 'msc_FragCharge'}, inplace=True)

    # write out the CSV of ms2 peaks
    print(" - {} ms2 peaks".format(len(ms2_peaks_df)))
    if idx == 0:
        ms2_peaks_df.to_csv(output_filename_ms2, mode='w', sep=',', index=False, header=True)
    else:
        ms2_peaks_df.to_csv(output_filename_ms2, mode='a', sep=',', index=False, header=False)

# now write out the anotated features
feature_list_df = pd.read_sql_query("select * from feature_list", db_conn)
# isolate the feature-level attributes reported by MSC
msc_feature_df = msc_subset_df[["FeatureNum","DiggerPepScore","DeltaMassDa","DeltaMassPPM"]]
# annotate the features with the attributes from MSC
# msc_DiggerPepScore of zero or more is what MSC reported. NaN means MSC didn't report the feature at all
feature_list_msc_df = pd.merge(feature_list_df, msc_feature_df, how='left', left_on=['feature_id'], right_on=['FeatureNum'])
feature_list_msc_df.drop(['FeatureNum'], inplace=True, axis=1)
feature_list_msc_df.rename(columns={'DiggerPepScore': 'msc_DiggerPepScore', 'DeltaMassDa': 'msc_DeltaMassDa', 'DeltaMassPPM': 'msc_DeltaMassPPM'}, inplace=True)
feature_list_msc_df.to_csv(output_filename_features, mode='w', sep=',', index=False, header=True)

db_conn.close()
