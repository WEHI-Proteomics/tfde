import pandas as pd
import glob
import sqlite3
import argparse
import os


parser = argparse.ArgumentParser(description='Gather the detected features for analysis.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
args = parser.parse_args()


# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

FEATURES_DIR = "{}/features".format(EXPERIMENT_DIR)
FEATURES_DB_NAME = "{}/detected-features-no-recal.sqlite".format(FEATURES_DIR)
if os.path.isfile(FEATURES_DB_NAME):
    os.remove(FEATURES_DB_NAME)

run_names = glob.glob('{}/exp-*'.format(FEATURES_DIR))
df_l = []
db_conn = sqlite3.connect(FEATURES_DB_NAME)
for r in run_names:
    run_name = r.split('/')[-1]
    features_dir = '{}/{}'.format(FEATURES_DIR, run_name)
    run_feature_files = glob.glob("{}/exp-{}-run-{}-features-precursor-*.pkl".format(features_dir, experiment_name, run_name))
    print("found {} feature files for the run {}".format(len(run_feature_files), run_name))
    for file in run_feature_files:
        df = pd.read_pickle(file)
        df['run_name'] = run_name
        df.to_sql(name='features', con=db_conn, if_exists='append', index=False)
db_conn.close()
