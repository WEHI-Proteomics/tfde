import pandas as pd
import glob
import sqlite3
import argparse
import os
import json


parser = argparse.ArgumentParser(description='Gather the detected features for analysis.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Only process one run.', required=False)
args = parser.parse_args()


# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

FEATURES_BASE_DIR = "{}/features".format(EXPERIMENT_DIR)
FEATURES_DB_NAME = "{}/detected-features-no-recal.sqlite".format(FEATURES_BASE_DIR)
if os.path.isfile(FEATURES_DB_NAME):
    os.remove(FEATURES_DB_NAME)

if args.run_name is None:
    r_l = glob.glob('{}/{}_*'.format(FEATURES_BASE_DIR, args.experiment_name))
    run_names = []
    for r in r_l:
        run_names.append(r.split('/')[-1])
else:
    run_names.append(args.run_name)

print('found {} runs in {}'.format(len(run_names), FEATURES_BASE_DIR))
db_conn = sqlite3.connect(FEATURES_DB_NAME)
for r in run_names:
    features_dir = '{}/{}'.format(FEATURES_BASE_DIR, run_name)
    run_feature_files = glob.glob("{}/exp-{}-run-{}-features-precursor-*.pkl".format(features_dir, args.experiment_name, run_name))
    print("found {} feature files for the run {}".format(len(run_feature_files), run_name))
    df_l = []
    for file in run_feature_files:
        df = pd.read_pickle(file)
        df_l.append(df)
    run_features_df = pd.concat(df_l, axis=0, sort=False)
    run_features_df.drop(['candidate_phr_error','mono_adjusted','original_phr','original_phr_error','rt_curve_fit','scan_curve_fit'], axis=1, inplace=True)
    run_features_df['envelope'] = run_features_df.apply(lambda row: json.dumps([tuple(e) for e in row.envelope]), axis=1)
    run_features_df['run_name'] = run_name
    run_features_df.to_sql(name='features', con=db_conn, if_exists='append', index=False)
db_conn.close()
