import pandas as pd
import sys
import sqlite3
import argparse
import os
import time
import shutil
import json


parser = argparse.ArgumentParser(description='Generate a file containing summary information about all the identifications and extractions in the experiment.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
args = parser.parse_args()

# print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the identifications directory exists
IDENTIFICATIONS_DIR = '{}/identifications-pasef'.format(EXPERIMENT_DIR)
if not os.path.exists(IDENTIFICATIONS_DIR):
    print("The identifications directory is required but doesn't exist: {}".format(IDENTIFICATIONS_DIR))
    sys.exit(1)

# check the identifications file exists
IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-pasef-recalibrated.feather'.format(IDENTIFICATIONS_DIR, args.experiment_name)
if not os.path.isfile(IDENTIFICATIONS_FILE):
    print("The identifications file doesn't exist: {}".format(IDENTIFICATIONS_FILE))
    sys.exit(1)

# load the experiment identifications
identifications_df = pd.read_feather(IDENTIFICATIONS_FILE)
print('loaded {} experiment identifications from {}'.format(len(identifications_df), IDENTIFICATIONS_FILE))

# remove any rubbish peptide masses
identifications_df = identifications_df[identifications_df['peptide mass'] > 0]

# summarise the identifications
print('summarising the identifications')
identifications_l = []
for group_name,group_df in identifications_df.groupby(['sequence','charge'], as_index=False):
    proteins_l = list(group_df['protein id'].unique())
    run_names_l = list(group_df.run_name.unique())
    q_value = group_df['percolator q-value'].iloc[0]
    ids_d = {'perc_q_value':q_value, 'run_names':run_names_l, 'number_of_runs':len(run_names_l), 'proteins':proteins_l, 'number_of_proteins':len(proteins_l)}
    identifications_l.append((group_name[0], group_name[1], json.dumps(ids_d)))
identifications_df = pd.DataFrame(identifications_l, columns=['sequence','charge','identifications'])

# load the extractions
EXTRACTED_FEATURES_DB_NAME = '{}/extracted-features/extracted-features.sqlite'.format(EXPERIMENT_DIR)
print('loading the extractions from {}'.format(EXTRACTED_FEATURES_DB_NAME))
db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
extracted_features_df = pd.read_sql_query("select sequence,charge,run_name,prob_target,intensity,inferred from features where classed_as == \'target\'", db_conn)
db_conn.close()

# summarise the extractions
print('summarising the extractions')
extractions_l = []
for group_name,group_df in extracted_features_df.groupby(['sequence','charge'], as_index=False):
    extracts_d = group_df[['run_name','prob_target','intensity','inferred']].to_dict('records')
    extractions_l.append((group_name[0], group_name[1], json.dumps(extracts_d)))
extractions_df = pd.DataFrame(extractions_l, columns=['sequence','charge','extractions'])

# merge them
print('merging the identifications and extractions')
results_df = pd.merge(identifications_df, extractions_df, how='outer', left_on=['sequence','charge'], right_on=['sequence','charge'])
del identifications_df
del extractions_df

RESULTS_DIR = "{}/summarised-results".format(EXPERIMENT_DIR)
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.makedirs(RESULTS_DIR)

RESULTS_DB_FILE_NAME = '{}/results.sqlite'.format(RESULTS_DIR)
print('saving the results to {}'.format(RESULTS_DB_FILE_NAME))
db_conn = sqlite3.connect(RESULTS_DB_FILE_NAME)
results_df.to_sql(name='sequences', con=db_conn, if_exists='replace', index=False)
db_conn.close()

# record the time it took
stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
