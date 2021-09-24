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
from sklearn.model_selection import GridSearchCV,ShuffleSplit
from sklearn.model_selection import train_test_split
import configparser
from configparser import ExtendedInterpolation


def GradientBooster(param_grid, n_jobs, X_train, y_train):
    estimator = GradientBoostingRegressor()
    cv = ShuffleSplit(n_splits=10, train_size=0.8, test_size=0.2, random_state=0)
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs)
    classifier.fit(X_train, y_train)
    print('best estimator found by grid search: {}'.format(classifier.best_estimator_))
    return cv, classifier.best_estimator_

# use the coordinate estimators to estimate the target coordinates for the given sequence-charge attributes
def estimate_target_coordinates(row_as_series, mz_estimator, scan_estimator, rt_estimator):
    sequence_estimation_attribs_s = row_as_series[['theoretical_mz','experiment_rt_mean','experiment_rt_std_dev','experiment_scan_mean','experiment_scan_std_dev','experiment_intensity_mean','experiment_intensity_std_dev']]
    sequence_estimation_attribs = np.reshape(sequence_estimation_attribs_s.values, (1, -1))  # make it 2D

    # estimate the raw monoisotopic m/z
    mz_delta_ppm_estimated = mz_estimator.predict(sequence_estimation_attribs)[0]
    theoretical_mz = sequence_estimation_attribs_s.theoretical_mz
    estimated_monoisotopic_mz = (mz_delta_ppm_estimated / 1e6 * theoretical_mz) + theoretical_mz

    # estimate the raw monoisotopic scan
    estimated_scan_delta = scan_estimator.predict(sequence_estimation_attribs)[0]
    experiment_scan_mean = sequence_estimation_attribs_s.experiment_scan_mean
    estimated_scan_apex = (estimated_scan_delta * experiment_scan_mean) + experiment_scan_mean

    # estimate the raw monoisotopic RT
    estimated_rt_delta = rt_estimator.predict(sequence_estimation_attribs)[0]
    experiment_rt_mean = sequence_estimation_attribs_s.experiment_rt_mean
    estimated_rt_apex = (estimated_rt_delta * experiment_rt_mean) + experiment_rt_mean

    return (estimated_monoisotopic_mz, estimated_scan_apex, estimated_rt_apex)


####################################################################

# This program uses the sequence library to build estimation models to estimate where in each run the library sequence should be.

parser = argparse.ArgumentParser(description='Using the library sequences, build run-specific coordinate estimators for the sequence-charges identified in the experiment.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
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
SEQUENCE_LIBRARY_DIR = "{}/sequence-library".format(EXPERIMENT_DIR)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print('there are {} examples in the training set, {} in the test set'.format(len(X_train), len(X_test)))

    # save the test set so we can evaluate performance
    np.save('{}/run-{}-X_test.npy'.format(COORDINATE_ESTIMATORS_DIR, run_name), X_test)
    np.save('{}/run-{}-y_test.npy'.format(COORDINATE_ESTIMATORS_DIR, run_name), y_test)

    # build the m/z delta estimation model - estimate the m/z delta ppm as a proportion of the experiment-wide value
    y_train_delta_mz_ppm = y_train[:,0]
    y_test_delta_mz_ppm = y_test[:,0]

    mz_estimator = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                              learning_rate=0.1, loss='ls', max_depth=10,
                              max_features=1.0, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=5, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=500,
                              n_iter_no_change=None, presort='auto',
                              random_state=None, subsample=1.0, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)

    # use the best parameters to train the model
    mz_estimator.fit(X_train, y_train_delta_mz_ppm)
    y_train_pred = mz_estimator.predict(X_train)
    y_test_pred = mz_estimator.predict(X_test)
    print("m/z estimator: mean absolute error for training set: {}, test set: {}".format(round(np.abs(y_train-y_train_pred).mean(),4), round(np.abs(y_test-y_test_pred).mean(),4)))

    # save the trained model
    ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, 'mz')
    with open(ESTIMATOR_MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(mz_estimator, file)

    # build the scan estimation model - estimate the delta scan as a proportion of the experiment-wide value
    y_train_delta_scan = y_train[:,1]
    y_test_delta_scan = y_test[:,1]

    scan_estimator = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                              learning_rate=0.1, loss='ls', max_depth=10,
                              max_features=1.0, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=5, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=500,
                              n_iter_no_change=None, presort='auto',
                              random_state=None, subsample=1.0, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)

    # use the best parameters to train the model
    scan_estimator.fit(X_train, y_train_delta_scan)
    y_train_pred = scan_estimator.predict(X_train)
    y_test_pred = scan_estimator.predict(X_test)
    print("scan estimator: mean absolute error for training set: {}, test set: {}".format(round(np.abs(y_train-y_train_pred).mean(),4), round(np.abs(y_test-y_test_pred).mean(),4)))

    # save the trained model
    ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, 'scan')
    with open(ESTIMATOR_MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(scan_estimator, file)

    # RT estimation model - estimate the RT delta as a proportion of the experiment-wide value
    y_train_delta_rt = y_train[:,2]
    y_test_delta_rt = y_test[:,2]

    rt_estimator = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                              learning_rate=0.1, loss='ls', max_depth=10,
                              max_features=1.0, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=5, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=500,
                              n_iter_no_change=None, presort='auto',
                              random_state=None, subsample=1.0, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)

    # use the best parameters to train the model
    rt_estimator.fit(X_train, y_train_delta_rt)
    y_train_pred = rt_estimator.predict(X_train)
    y_test_pred = rt_estimator.predict(X_test)
    print("RT estimator: mean absolute error for training set: {}, test set: {}".format(round(np.abs(y_train-y_train_pred).mean(),4), round(np.abs(y_test-y_test_pred).mean(),4)))

    # save the trained model
    ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, 'rt')
    with open(ESTIMATOR_MODEL_FILE_NAME, 'wb') as file:
        pickle.dump(rt_estimator, file)
    
    print()

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
