import pandas as pd
import pickle
import numpy as np
import sqlite3
import json
import os
import shutil
import time
import argparse
import sys

# This program classifies the target and decoy features extracted for the library sequences in each run, and stores the features classified as targets in a database with metrics and attributes unpacked.


##################################################

parser = argparse.ArgumentParser(description='Orchestrate the feature extraction of sequence library features from all runs.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_names', type=str, help='Comma-separated names of runs to process.', required=True)
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.', required=False)
parser.add_argument('-ssms','--small_set_mode_size', type=int, default='100', help='The number of sequences to sample for small set mode.', required=False)
args = parser.parse_args()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# set up the target decoy classifier directory
TARGET_DECOY_MODEL_DIR = "{}/target-decoy-models".format(EXPERIMENT_DIR)
CLASSIFIER_FILE_NAME = "{}/target-decoy-classifier.pkl".format(TARGET_DECOY_MODEL_DIR)
if not os.path.isfile(CLASSIFIER_FILE_NAME):
    print("The target-decoy classifier is required but doesn't exist: {}".format(CLASSIFIER_FILE_NAME))
    sys.exit(1)

# check the experiment metrics database exists
METRICS_DB_NAME = "{}/experiment-metrics-for-library-sequences.sqlite".format(TARGET_DECOY_MODEL_DIR)
if not os.path.isfile(METRICS_DB_NAME):
    print("The extracted features database is required but doesn't exist: {}".format(METRICS_DB_NAME))
    sys.exit(1)

# set up the extracted features directory
EXTRACTED_FEATURES_DIR = "{}/extracted-features".format(EXPERIMENT_DIR)
if os.path.exists(EXTRACTED_FEATURES_DIR):
    shutil.rmtree(EXTRACTED_FEATURES_DIR)
os.makedirs(EXTRACTED_FEATURES_DIR)
print("The extracted features directory was created: {}".format(EXTRACTED_FEATURES_DIR))
EXTRACTED_FEATURES_DB_NAME = "{}/extracted-features.sqlite".format(EXTRACTED_FEATURES_DIR)

##################################################

# print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

if args.small_set_mode:
    small_set_flags = "limit {}".format(args.small_set_mode_size)
else:
    small_set_flags = ""

# load the classifier
with open(CLASSIFIER_FILE_NAME, 'rb') as file:
    gbc = pickle.load(file)

# reduce memory requirements by processing the extracted features one run at a time
run_names_l = args.run_names.split(',')
for run_name in run_names_l:
    print('processing {}'.format(run_name))

    # load the metrics for the extracted features
    db_conn = sqlite3.connect(METRICS_DB_NAME)
    # create the index if it's not already there
    print("creating index in {}".format(METRICS_DB_NAME))
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_extracted_metrics_2 on extracted_metrics (run_name)")
    # read the sequences for this run
    sequences_df = pd.read_sql_query('select sequence,charge,run_name,peak_idx,target_coords,target_metrics,attributes,theoretical_mz,q_value from extracted_metrics where run_name == \'{}\' {}'.format(run_name, small_set_flags), db_conn)
    db_conn.close()
    print("loaded {} feature metrics from {}".format(len(sequences_df), METRICS_DB_NAME))

    if len(sequences_df) > 0:
        # fix up types
        sequences_df.peak_idx = sequences_df.peak_idx.astype(int)

        # unpack the target metrics from each sequence
        print("unpacking metrics")
        metrics = []
        metric_names = [key for key in sorted(json.loads(sequences_df.iloc[0].target_metrics))]

        for row in sequences_df.itertuples():
            # target metric values
            d = json.loads(row.target_metrics)
            if ((d is not None) and (isinstance(d, dict))):
                l = []
                l.append(row.sequence)
                l.append(row.charge)
                l.append(row.run_name)
                l.append(int(row.peak_idx))
                l += [d[key] for key in sorted(d)]
                metrics.append(tuple(l))

        # drop the metrics column because we no longer need it
        sequences_df.drop('target_metrics', axis=1, inplace=True)

        # create a dataframe with the expanded metrics
        columns = ['sequence','charge','run_name','peak_idx']
        columns += metric_names
        metrics_df = pd.DataFrame(metrics, columns=columns)
        # del metrics[:]  # no longer needed

        # tidy up any attributes that will upset the model training
        metrics_df.fillna(value=0.0, inplace=True)
        metrics_df.r_squared_phr.replace((-np.inf, 0), inplace=True)

        print("merging unpacked metrics")
        sequences_df = pd.merge(sequences_df, metrics_df, how='left', left_on=['sequence','charge','run_name','peak_idx'], right_on=['sequence','charge','run_name','peak_idx'])
        # metrics_df = metrics_df.iloc[0:0]  # no longer needed

        # classify the sequences
        sequences_df['classed_as'] = gbc.predict(sequences_df[metric_names].values).tolist()
        class_probabilities = gbc.predict_proba(sequences_df[metric_names].values)
        class_probabilities_df = pd.DataFrame(class_probabilities, columns=['prob_decoy','prob_target'])
        sequences_df = pd.concat([sequences_df, class_probabilities_df], axis=1)

        # filter out the features that were not classified as targets
        # sequences_df = sequences_df[(sequences_df.classed_as == 'target')]
        # print("trimmed to {} features classified as targets".format(len(sequences_df)))

        # unpack the attributes from each sequence
        print("unpacking attributes")
        attributes = []
        attribute_names = [key for key in sorted(json.loads(sequences_df.iloc[0].attributes))]

        for row in sequences_df.itertuples():
            # attribute values
            d = json.loads(row.attributes)
            if ((d is not None) and (isinstance(d, dict))):
                l = []
                l.append(row.sequence)
                l.append(row.charge)
                l.append(row.run_name)
                l.append(int(row.peak_idx))
                l += [d[key] for key in sorted(d)]
                attributes.append(tuple(l))

        # drop the attributes column because we no longer need it
        sequences_df.drop('attributes', axis=1, inplace=True)

        # create a dataframe with the expanded attributes
        columns = ['sequence','charge','run_name','peak_idx']
        columns += attribute_names
        attributes_df = pd.DataFrame(attributes, columns=columns)
        # del attributes[:]  # no longer needed

        print("merging unpacked attributes")
        sequences_df = pd.merge(sequences_df, attributes_df, how='left', left_on=['sequence','charge','run_name','peak_idx'], right_on=['sequence','charge','run_name','peak_idx'])
        # attributes_df = attributes_df.iloc[0:0]  # no longer needed

        # filter out any sequences that have no intensity
        sequences_df = sequences_df[(sequences_df.intensity > 0)]

        # convert the lists to JSON so we can store them in SQLite
        print("converting diagnostic info to JSON")
        sequences_df.mono_filtered_points_l = sequences_df.apply(lambda row: json.dumps(row.mono_filtered_points_l), axis=1)
        sequences_df.mono_rt_bounds = sequences_df.apply(lambda row: json.dumps(row.mono_rt_bounds), axis=1)
        sequences_df.mono_scan_bounds = sequences_df.apply(lambda row: json.dumps(row.mono_scan_bounds), axis=1)

        sequences_df.isotope_1_filtered_points_l = sequences_df.apply(lambda row: json.dumps(row.isotope_1_filtered_points_l), axis=1)
        sequences_df.isotope_1_rt_bounds = sequences_df.apply(lambda row: json.dumps(row.isotope_1_rt_bounds), axis=1)
        sequences_df.isotope_1_scan_bounds = sequences_df.apply(lambda row: json.dumps(row.isotope_1_scan_bounds), axis=1)

        sequences_df.isotope_2_filtered_points_l = sequences_df.apply(lambda row: json.dumps(row.isotope_2_filtered_points_l), axis=1)
        sequences_df.isotope_2_rt_bounds = sequences_df.apply(lambda row: json.dumps(row.isotope_2_rt_bounds), axis=1)
        sequences_df.isotope_2_scan_bounds = sequences_df.apply(lambda row: json.dumps(row.isotope_2_scan_bounds), axis=1)

        sequences_df.isotope_intensities_l = sequences_df.apply(lambda row: json.dumps(row.isotope_intensities_l), axis=1)

        sequences_df.peak_proportions = sequences_df.apply(lambda row: json.dumps(row.peak_proportions), axis=1)

        # fix up some types
        sequences_df.inferred = sequences_df.inferred.astype(bool)

        # write out the results for analysis
        print("writing out the extracted sequences to {}".format(EXTRACTED_FEATURES_DB_NAME))
        db_conn = sqlite3.connect(EXTRACTED_FEATURES_DB_NAME)
        sequences_df.to_sql(name='features', con=db_conn, if_exists='append', index=False)
        db_conn.close()
    else:
        print("The metrics database {} has no records for the specified run {}".format(METRICS_DB_NAME, run_name))
        sys.exit(1)

# finish up
stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
