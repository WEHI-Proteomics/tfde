import glob
import os
import shutil
import time
import argparse
import sys
import pandas as pd
import pickle
import configparser
from configparser import ExtendedInterpolation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,ShuffleSplit
import ray
import multiprocessing as mp

# convert the monoisotopic mass to the monoisotopic m/z
def mono_mass_to_mono_mz(monoisotopic_mass, charge):
    return (monoisotopic_mass / charge) + PROTON_MASS

# train a model on the features that gave the best identifications to predict the mass error, so we can predict the mass error for all the features 
# detected (not just those with high quality identifications), and adjust their calculated mass to give zero mass error.
@ray.remote
def adjust_features(run_name, idents_for_training_df, run_features_df):
    print("Processing file {}".format(run_name))

    X_train = idents_for_training_df[['mono_mz_without_saturated_points','scan_apex','rt_apex','feature_intensity']].to_numpy()
    y_train = idents_for_training_df[['mass_error']].to_numpy()[:,0]

    # search for the best model in the specified hyperparameter space
    param_grid = {'n_estimators':[100], 'learning_rate': [0.1, 0.05, 0.02, 0.01], 'max_depth':[20, 10, 6, 4], 'min_samples_leaf':[3, 5, 9, 17], 'max_features':[1.0, 0.3, 0.1] }
    n_jobs = 2
    cv, best_estimator = GradientBooster(param_grid, n_jobs, X_train, y_train)

    # use the best parameters to train the model
    best_estimator.fit(X_train, y_train)
    print("R-squared for training set (best model found): {}".format(best_estimator.score(X_train, y_train)))

    # use the trained model to predict the mass error for all the detected features
    X = run_features_df[['mono_mz_without_saturated_points','scan_apex','rt_apex','feature_intensity']].to_numpy()
    y = best_estimator.predict(X)

    # collate the recalibrated feature attributes
    run_features_df['predicted_mass_error'] = y
    run_features_df['recalibrated_monoisotopic_mass'] = run_features_df.monoisotopic_mass - run_features_df.predicted_mass_error
    run_features_df['recalibrated_monoisotopic_mz'] = run_features_df.apply(lambda row: mono_mass_to_mono_mz(row.recalibrated_monoisotopic_mass, row.charge), axis=1)

    return {'run_name':run_name, 'adjusted_features_df':run_features_df}

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = round(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers


################################
parser = argparse.ArgumentParser(description='Use high-quality identifications to recalibrate the mass of detected features.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-fdm','--feature_detection_method', type=str, choices=['pasef','3did'], help='Which feature detection method.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
args = parser.parse_args()

# Print the arguments for the log
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

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
PROTON_MASS = cfg.getfloat('common','PROTON_MASS')
ADD_C_CYSTEINE_DA = cfg.getfloat('common','ADD_C_CYSTEINE_DA')
MAXIMUM_Q_VALUE_FOR_RECAL_TRAINING_SET = cfg.getfloat('common','MAXIMUM_Q_VALUE_FOR_RECAL_TRAINING_SET')

# check the identifications directory
IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, args.feature_detection_method)
if not os.path.exists(IDENTIFICATIONS_DIR):
    print("The identifications directory is required but doesn't exist: {}".format(IDENTIFICATIONS_DIR))
    sys.exit(1)

# check the identifications file
IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.pkl'.format(IDENTIFICATIONS_DIR, args.experiment_name, args.feature_detection_method)
if not os.path.isfile(IDENTIFICATIONS_FILE):
    print("The identifications file doesn't exist: {}".format(IDENTIFICATIONS_FILE))
    sys.exit(1)

# load the identifications to use for the training set
with open(IDENTIFICATIONS_FILE, 'rb') as handle:
    idents_df = pickle.load(handle)['identifications_df']
idents_df = idents_df[(idents_df['percolator q-value'] <= MAXIMUM_Q_VALUE_FOR_RECAL_TRAINING_SET)]
print('loaded {} identifications with q-value lower than {} from {}'.format(len(idents_df), MAXIMUM_Q_VALUE_FOR_RECAL_TRAINING_SET, IDENTIFICATIONS_FILE))

# load the features for recalibration
FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, args.feature_detection_method)
feature_files = glob.glob("{}/exp-{}-run-*-features-{}-dedup.pkl".format(FEATURES_DIR, args.experiment_name, args.feature_detection_method))
features_l = []
for f in feature_files:
    with open(f, 'rb') as handle:
        features_l.append(pickle.load(handle)['features_df'])
print('loaded {} features from {} files for recalibration'.format(len(features_df), len(feature_files)))

# set up Ray
print("setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

# for each run, produce a model that estimates the mass error from a feature's characteristics, and generate a revised feature file with adjusted mass, 
# to get a smaller mass error on a second Comet search with tighter mass tolerance.
print("training models and adjusting monoisotopic mass for each feature")
adjusted_features_l = ray.get([adjust_features.remote(run_name, idents_for_training_df=group_df, run_features_df=features_df[features_df.run_name == run_name]) for file_idx,group_df in idents_df.groupby('run_name')])

# write out the recalibrated features
for adj in adjusted_features_l:
    RECAL_FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.pkl'.format(FEATURES_DIR, args.experiment_name, adj['run_name'], args.feature_detection_method)
    print("writing {} recalibrated features to {}".format(len(adjustments_df), RECAL_FEATURES_FILE))
    info.append(('total_running_time',round(time.time()-start_run,1)))
    info.append(('processor',parser.prog))
    info.append(('processed', time.ctime()))
    content_d = {'features_df':adj['adjusted_features_df'], 'metadata':info}
    with open(RECAL_FEATURES_FILE, 'wb') as handle:
        pickle.dump(content_d, handle)

# finish up
stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
