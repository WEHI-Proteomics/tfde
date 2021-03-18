import pandas as pd
import numpy as np
import sqlite3
import json
import time
import argparse

###################################
parser = argparse.ArgumentParser(description='Remove duplicates of the features detected.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-mz','--mz_tolerance_ppm', type=int, default='5', help='Tolerance for m/z.', required=False)
parser.add_argument('-scan','--scan_tolerance', type=int, default='10', help='Tolerance for scan.', required=False)
parser.add_argument('-rt','--rt_tolerance', type=int, default='5', help='Tolerance for retention time (seconds).', required=False)
parser.add_argument('-rtl','--rt_lower', type=int, default='1650', help='The lower RT limit to de-dup.', required=False)
parser.add_argument('-rtu','--rt_upper', type=int, default='2200', help='The upper RT limit to de-dup.', required=False)
parser.add_argument('-fdm','--feature_detection_method', type=str, choices=['pasef','3did'], help='Which feature detection method to de-dup.', required=True)
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

EXPERIMENT_DIR = '{}/{}'.format(args.experiment_base_dir, args.experiment_name)

if args.feature_detection_method == '3did':
    FEATURES_DIR = '{}/features-3did'.format(EXPERIMENT_DIR)
    FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name)
    MZ_COLUMN_NAME = 'monoisotopic_mz'
    
    features_df = pd.read_pickle(FEATURES_FILE)

if args.feature_detection_method == 'pasef':
    FEATURES_DIR = '{}/features'.format(EXPERIMENT_DIR)
    FEATURES_FILE = '{}/detected-features-no-recal.sqlite'.format(FEATURES_DIR)
    MZ_COLUMN_NAME = 'monoisotopic_mz'

    db_conn = sqlite3.connect(FEATURES_FILE)
    features_df = pd.read_sql_query("select * from features where rt_apex >= {} and rt_apex <= {} and run_name=='{}'".format(args.rt_lower, args.rt_upper, args.run_name), db_conn)
    db_conn.close()

FEATURES_DEDUP_FILE = '{}/exp-{}-run-{}-features-{}-dedup.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name, args.feature_detection_method)

print('there are {} features prior to de-dup'.format(len(features_df)))

MZ_TOLERANCE_PERCENT = args.mz_tolerance_ppm * 10**-4

features_df['dup_mz'] = features_df[MZ_COLUMN_NAME]  # shorthand to reduce verbosity
features_df['dup_mz_ppm_tolerance'] = features_df.dup_mz * MZ_TOLERANCE_PERCENT / 100
features_df['dup_mz_lower'] = features_df.dup_mz - features_df.dup_mz_ppm_tolerance
features_df['dup_mz_upper'] = features_df.dup_mz + features_df.dup_mz_ppm_tolerance
features_df['dup_scan_lower'] = features_df.scan_apex - args.scan_tolerance
features_df['dup_scan_upper'] = features_df.scan_apex + args.scan_tolerance
features_df['dup_rt_lower'] = features_df.rt_apex - args.rt_tolerance
features_df['dup_rt_upper'] = features_df.rt_apex + args.rt_tolerance
features_df['dup_composite_key'] = features_df.apply(lambda row: '{},{}'.format(row.feature_id, row.precursor_id), axis=1)

# remove these after we're finished
columns_to_drop_l = []
columns_to_drop_l.append('dup_mz')
columns_to_drop_l.append('dup_mz_ppm_tolerance')
columns_to_drop_l.append('dup_mz_lower')
columns_to_drop_l.append('dup_mz_upper')
columns_to_drop_l.append('dup_scan_lower')
columns_to_drop_l.append('dup_scan_upper')
columns_to_drop_l.append('dup_rt_lower')
columns_to_drop_l.append('dup_rt_upper')
columns_to_drop_l.append('dup_composite_key')

features_df.sort_values(by=['intensity'], ascending=False, inplace=True)

# see if any detections have a duplicate - if so, find the dup with the highest intensity and keep it
keep_l = []
for row in features_df.itertuples():
    dup_df = features_df[(features_df.dup_mz > row.dup_mz_lower) & (features_df.dup_mz < row.dup_mz_upper) & (features_df.scan_apex > row.dup_scan_lower) & (features_df.scan_apex < row.dup_scan_upper) & (features_df.rt_apex > row.dup_rt_lower) & (features_df.rt_apex < row.dup_rt_upper)].copy()
    # group the dups by charge - take the most intense for each charge
    for group_name,group_df in dup_df.groupby(['charge'], as_index=False):
        keep_l.append(group_df.iloc[0].dup_composite_key)

# remove any features that are not in the keep list
dedup_df = features_df[features_df.dup_composite_key.isin(keep_l)].copy()

number_of_dups = len(features_df)-len(dedup_df)
print('removed {} duplicates ({}% of the original detections)'.format(number_of_dups, round(number_of_dups/len(features_df)*100)))
print('there are {} detected de-duplicated features'.format(len(dedup_df)))

# remove the columns we added earlier
dedup_df.drop(columns_to_drop_l, axis=1, inplace=True)

# save the de-dup features
dedup_df.to_pickle(FEATURES_DEDUP_FILE)
print('wrote {} de-dup features to {}'.format(len(dedup_df), FEATURES_DEDUP_FILE))

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
