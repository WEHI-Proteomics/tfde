import pandas as pd
import numpy as np
import sys
import pickle
import os
import shutil
import time
import argparse
import peakutils
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import configparser
from configparser import ExtendedInterpolation


def generate_estimator(X_train, X_test, y_train, y_test):
    if args.search_for_new_model_parameters:
        # do a randomised search to find the best regressor dimensions
        print('setting up randomised search')
        parameter_search_space = {
            "loss": ['ls','lad','huber'],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            'n_estimators': range(20,510,10),
            'max_depth':range(5,30,2), 
            'min_samples_split':range(100,1001,100),
            'subsample':list(np.arange(0.2,0.9,0.1)),
            'min_samples_leaf':range(10,71,10),
            'max_features':["log2", "sqrt"],
            }
        # cross-validation splitting strategy uses 'cv' folds in a (Stratified)KFold
        rsearch = RandomizedSearchCV(GradientBoostingRegressor(), parameter_search_space, n_iter=100, n_jobs=-1, random_state=10, cv=5, scoring='r2', verbose=1)  # All scorer objects follow the convention that higher return values are better than lower return values, so we want the negated version for error metrics
        print('fitting to the training set')
        # find the best fit within the parameter search space
        rsearch.fit(X_train, y_train)
        best_estimator = rsearch.best_estimator_
        print('best score from the search: {}'.format(round(rsearch.best_score_, 4)))
        best_params = rsearch.best_params_
        print(best_params)
    else:
        print('fitting the estimator to the training data')
        # use the model parameters we found previously
        best_params = {'subsample': 0.6, 'n_estimators': 280, 'min_samples_split': 400, 'min_samples_leaf': 10, 'max_features': 'log2', 'max_depth': 11, 'loss': 'lad', 'learning_rate': 0.05}
        best_estimator = GradientBoostingRegressor(**best_params)
        best_estimator.fit(X_train, y_train)  # find the best fit within the parameter search space

    # calculate the estimator's score on the train and test sets
    print('evaluating against the training and test set')
    y_train_pred = best_estimator.predict(X_train)
    y_test_pred = best_estimator.predict(X_test)
    print("mean absolute error for training set: {}, test set: {}".format(round(np.abs(y_train-y_train_pred).mean(),4), round(np.abs(y_test-y_test_pred).mean(),4)))
    return best_estimator


####################################################################

# This program uses the sequence library to build estimation models to estimate where in each run the library sequence should be.

parser = argparse.ArgumentParser(description='Using the library sequences, build run-specific coordinate estimators for the sequence-charges identified in the experiment.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./tfde/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-snmp','--search_for_new_model_parameters', action='store_true', help='Search for new model parameters.')
parser.add_argument('-pdm','--precursor_definition_method', type=str, choices=['pasef','3did','mq'], default='pasef', help='The method used to define the precursor cuboids.', required=False)
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# load the sequence library
SEQUENCE_LIBRARY_DIR = "{}/sequence-library-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.feather".format(SEQUENCE_LIBRARY_DIR)
if not os.path.isfile(SEQUENCE_LIBRARY_FILE_NAME):
    print("The sequences library file doesn't exist: {}".format(SEQUENCE_LIBRARY_FILE_NAME))
    sys.exit(1)

# load the sequence library
library_sequences_df = pd.read_feather(SEQUENCE_LIBRARY_FILE_NAME)
print('loaded {} sequences from the library {}'.format(len(library_sequences_df), SEQUENCE_LIBRARY_FILE_NAME))

# load the indentifications from each run
IDENTIFICATIONS_DIR = '{}/identifications-pasef'.format(EXPERIMENT_DIR)
IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-pasef-recalibrated.feather'.format(IDENTIFICATIONS_DIR, args.experiment_name)
if not os.path.isfile(IDENTIFICATIONS_FILE):
    print("The identifications file doesn't exist: {}".format(IDENTIFICATIONS_FILE))
    sys.exit(1)

# load the experiment identifications
identifications_df = pd.read_feather(IDENTIFICATIONS_FILE)
print('loaded {} identifications from {}'.format(len(identifications_df), IDENTIFICATIONS_FILE))
identifications_df['human'] = identifications_df['protein id'].str.contains('HUMAN')

# set up the coordinate estimators directory
COORDINATE_ESTIMATORS_DIR = "{}/coordinate-estimators".format(EXPERIMENT_DIR)
if os.path.exists(COORDINATE_ESTIMATORS_DIR):
    shutil.rmtree(COORDINATE_ESTIMATORS_DIR)
os.makedirs(COORDINATE_ESTIMATORS_DIR)
print("The coordinate estimators directory was created: {}".format(COORDINATE_ESTIMATORS_DIR))

# outputs
RUN_SEQUENCES_FILE_NAME = "{}/run-sequence-attribs.feather".format(COORDINATE_ESTIMATORS_DIR)
MERGED_RUN_LIBRARY_SEQUENCES_FILE_NAME = "{}/merged-run-library-sequence-attribs.feather".format(COORDINATE_ESTIMATORS_DIR)

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
MINIMUM_PROPORTION_OF_IDENTS_FOR_COORD_ESTIMATOR_TRAINING = cfg.getfloat('extraction','MINIMUM_PROPORTION_OF_IDENTS_FOR_COORD_ESTIMATOR_TRAINING')

start_run = time.time()

# for each run, find the mz, scan, RT, and intensity for each sequence-charge identified
run_sequences_l = []
for group_name,group_df in identifications_df.groupby(['run_name','sequence','charge'], as_index=False):
    run_name = group_name[0]
    sequence = group_name[1]
    charge = group_name[2]
    run_mz_mean = peakutils.centroid(group_df.recalibrated_monoisotopic_mz, group_df.feature_intensity)
    run_mz_std_dev = np.std(group_df.recalibrated_monoisotopic_mz)
    run_scan_mean = np.mean(group_df.scan_apex)
    run_scan_std_dev = np.std(group_df.scan_apex)
    run_rt_mean = np.mean(group_df.rt_apex)
    run_rt_std_dev = np.std(group_df.rt_apex)
    run_intensity_mean = np.mean(group_df.feature_intensity)
    run_intensity_std_dev = np.std(group_df.feature_intensity)
    run_sequences_l.append((run_name,sequence,charge,run_mz_mean,run_scan_mean,run_rt_mean,run_mz_std_dev,run_scan_std_dev,run_rt_std_dev,run_intensity_mean,run_intensity_std_dev))

run_sequences_df = pd.DataFrame(run_sequences_l, columns=['run_name','sequence','charge','run_mz','run_scan','run_rt','run_mz_std_dev','run_scan_std_dev','run_rt_std_dev','run_intensity','run_intensity_std_dev'])

# calculate the coefficients of variance
run_sequences_df['cv_mz'] = run_sequences_df.run_mz_std_dev / run_sequences_df.run_mz
run_sequences_df['cv_scan'] = run_sequences_df.run_scan_std_dev / run_sequences_df.run_scan
run_sequences_df['cv_rt'] = run_sequences_df.run_rt_std_dev / run_sequences_df.run_rt
run_sequences_df['cv_intensity'] = run_sequences_df.run_intensity_std_dev / run_sequences_df.run_intensity

run_sequences_df.to_feather(RUN_SEQUENCES_FILE_NAME)

# merge the sequence-charges for each run with their library counterparts
merged_df = pd.merge(run_sequences_df, library_sequences_df, how='left', left_on=['sequence','charge'], right_on=['sequence','charge'])

# for each run-sequence-charge, calculate the delta from the library
merged_df['delta_mz'] = merged_df.run_mz - merged_df.theoretical_mz
merged_df['delta_mz_ppm'] = (merged_df.run_mz - merged_df.theoretical_mz) / merged_df.theoretical_mz * 1e6
merged_df['delta_scan'] = (merged_df.run_scan - merged_df.experiment_scan_mean) / merged_df.experiment_scan_mean
merged_df['delta_rt'] = (merged_df.run_rt - merged_df.experiment_rt_mean) / merged_df.experiment_rt_mean

merged_df.drop(['run_mz_std_dev','run_scan_std_dev','run_rt_std_dev','run_intensity_std_dev'], axis=1, inplace=True)
print("writing {} merged run-library sequence attributes to {}".format(len(merged_df), MERGED_RUN_LIBRARY_SEQUENCES_FILE_NAME))
merged_df.to_feather(MERGED_RUN_LIBRARY_SEQUENCES_FILE_NAME)

# create an estimator for each run in the experiment
run_names_l = list(identifications_df.run_name.unique())
for run_name in run_names_l:
    print("building the coordinate estimators for run {}".format(run_name))
    estimator_training_set_df = merged_df[(merged_df.run_name == run_name) & (merged_df.number_of_runs_identified > round(len(run_names_l) * MINIMUM_PROPORTION_OF_IDENTS_FOR_COORD_ESTIMATOR_TRAINING))]

    # X is the same for all the estimators
    # filter out rows not to be used in this training set
    X = estimator_training_set_df[['theoretical_mz','experiment_rt_mean','experiment_rt_std_dev','experiment_scan_mean','experiment_scan_std_dev','experiment_intensity_mean','experiment_intensity_std_dev']].values
    y = estimator_training_set_df[['delta_mz_ppm','delta_scan','delta_rt','run_mz','run_scan','run_rt']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02)
    print('there are {} examples in the training set, {} in the test set'.format(len(X_train), len(X_test)))

    # save the test set so we can evaluate performance
    np.save('{}/run-{}-X_test.npy'.format(COORDINATE_ESTIMATORS_DIR, run_name), X_test)
    np.save('{}/run-{}-y_test.npy'.format(COORDINATE_ESTIMATORS_DIR, run_name), y_test)

    # build the m/z delta estimation model - estimate the m/z delta ppm as a proportion of the experiment-wide value
    print('training the m/z model')
    mz_estimator = generate_estimator(X_train, X_test, y_train[:,0], y_test[:,0])

    # save the trained m/z model
    ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, 'mz')
    with open(ESTIMATOR_MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(mz_estimator, file)

    # build the scan estimation model - estimate the delta scan as a proportion of the experiment-wide value
    print('training the scan model')
    scan_estimator = generate_estimator(X_train, X_test, y_train[:,1], y_test[:,1])

    # save the trained scan model
    ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, 'scan')
    with open(ESTIMATOR_MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(scan_estimator, file)

    # RT estimation model - estimate the RT delta as a proportion of the experiment-wide value
    print('training the RT model')
    rt_estimator = generate_estimator(X_train, X_test, y_train[:,2], y_test[:,2])

    # save the trained RT model
    ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, 'rt')
    with open(ESTIMATOR_MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(rt_estimator, file)
    
    print()

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
