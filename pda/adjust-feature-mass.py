import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,ShuffleSplit
import pickle
import time
import argparse
import ray

MAXIMUM_Q_VALUE = 0.005
PROTON_MASS = 1.007276
ADD_C_CYSTEINE_DA = 57.021464  # from Unimod.org

parser = argparse.ArgumentParser(description='Adjust the monoisotopic m/z for each feature detected.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster','join'], help='The Ray mode to use.', required=True)
parser.add_argument('-ra','--redis_address', type=str, help='Address of the cluster to join.', required=False)
args = parser.parse_args()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

PERCOLATOR_OUTPUT_DIR = "{}/percolator-output".format(EXPERIMENT_DIR)
if not os.path.exists(PERCOLATOR_OUTPUT_DIR):
    print("The percolator output directory is required but doesn't exist: {}".format(PERCOLATOR_OUTPUT_DIR))
    sys.exit(1)
PERCOLATOR_OUTPUT_FILE_NAME = "{}/{}.percolator.target.psms.txt".format(PERCOLATOR_OUTPUT_DIR, args.experiment_name)
PERCOLATOR_STDOUT_FILE_NAME = "{}/percolator-stdout.log".format(PERCOLATOR_OUTPUT_DIR)
PERCOLATOR_MAPPING_FILE_NAME = "{}/percolator-idx-mapping.csv".format(PERCOLATOR_OUTPUT_DIR)

FEATURES_DIR = "{}/features".format(EXPERIMENT_DIR)
if not os.path.exists(FEATURES_DIR):
    print("The features directory is required but doesn't exist: {}".format(FEATURES_DIR))
    sys.exit(1)

RECALIBRATED_FEATURES_DIR = "{}/recalibrated-features".format(EXPERIMENT_DIR)
if not os.path.exists(RECALIBRATED_FEATURES_DIR):
    os.makedirs(RECALIBRATED_FEATURES_DIR)
    print("The recalibrated features directory was created: {}".format(RECALIBRATED_FEATURES_DIR))

MASS_ERROR_ESTIMATORS_DIR = "{}/mass-correction-models".format(EXPERIMENT_DIR)
if not os.path.exists(MASS_ERROR_ESTIMATORS_DIR):
    os.makedirs(MASS_ERROR_ESTIMATORS_DIR)
    print("The mass error estimators directory was created: {}".format(MASS_ERROR_ESTIMATORS_DIR))

# source: https://shankarmsy.github.io/stories/gbrt-sklearn.html
def GradientBooster(param_grid, n_jobs):
    estimator = GradientBoostingRegressor()
    cv = ShuffleSplit(n_splits=10, train_size=0.8, test_size=.2, random_state=0)
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs)
    classifier.fit(X_train, y_train)
    print('best estimator found by grid search: {}'.format(classifier.best_estimator_))
    return cv, classifier.best_estimator_

def mono_mass_to_mono_mz(row):
    return (row.recalibrated_monoisotopic_mass / row.charge) + PROTON_MASS

@ray.remote
def adjust_features(file_idx, X_train, y_train, features_df):
    # train a model on the features that gave the best identifications to predict the mass error, so we can predict
    # the mass error for all the features detected (not just those with high quality identifications), and adjust
    # their calculated mass to give zero mass error.

    # search for the best model in the specified hyperparameter space
    param_grid = {'n_estimators':[100], 'learning_rate': [0.1, 0.05, 0.02, 0.01], 'max_depth':[20, 10, 6, 4], 'min_samples_leaf':[3, 5, 9, 17], 'max_features':[1.0, 0.3, 0.1] }
    n_jobs = 4
    cv, best_estimator = GradientBooster(param_grid, n_jobs)

    # use the best parameters to train the model
    best_estimator.fit(X_train, y_train)
    print("R-squared for training set (best model found): {}".format(best_estimator.score(X_train, y_train)))

    # use the trained model to predict the mass error for all the detected features
    X_df = features_df[['monoisotopic_mz','scan_apex','rt_apex','intensity']]
    X = X_df.to_numpy()
    y = best_estimator.predict(X)

    # calculate the recalibrated mass attributes
    feature_recal_attributes_df = pd.DataFrame()
    feature_recal_attributes_df['feature_id'] = feature_df.feature_id
    feature_recal_attributes_df['predicted_mass_error'] = y
    feature_recal_attributes_df['recalibrated_monoisotopic_mass'] = feature_df.monoisotopic_mass - feature_recal_attributes_df.predicted_mass_error
    feature_recal_attributes_df['recalibrated_monoisotopic_mz'] = feature_recal_attributes_df.apply(lambda row: mono_mass_to_mono_mz(row), axis=1)

    # add the minimal set of attributes required for MGF generation
    feature_recal_attributes_df['charge'] = feature_df.charge
    feature_recal_attributes_df['rt_apex'] = feature_df.rt_apex
    feature_recal_attributes_df['scan_apex'] = feature_df.scan_apex
    feature_recal_attributes_df['intensity'] = feature_df.intensity
    feature_recal_attributes_df['precursor_id'] = feature_df.precursor_id

    return file_idx, feature_recal_attributes_df, best_estimator

###########################################

# initialise Ray
if not ray.is_initialized():
    if args.ray_mode == "join":
        if args.redis_address is not None:
            ray.init(redis_address=args.redis_address)
        else:
            print("Argument error: a redis_address is needed for join mode")
            sys.exit(1)
    elif args.ray_mode == "cluster":
        ray.init(object_store_memory=40000000000,
                    redis_max_memory=25000000000)
    else:
        ray.init(local_mode=True)

start_run = time.time()

# load the percolator output
psms_df = pd.read_csv(PERCOLATOR_OUTPUT_FILE_NAME, sep='\t')
# we only want to use the high quality identifications for our training set
psms_df = psms_df[psms_df['percolator q-value'] <= MAXIMUM_Q_VALUE]

# determine the mapping between the percolator index and the run file name - this is only
# available by parsing percolator's stdout redirected to a text file.
mapping = []
with open(PERCOLATOR_STDOUT_FILE_NAME) as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('INFO: Assigning index'):
            splits = line.split(' ')
            percolator_index = int(splits[3])
            comet_filename = splits[5]
            run_name = comet_filename.split('/')[-1].split('.')[0]  # e.g. 190719_Hela_Ecoli_1to3_06
            mapping.append((percolator_index, run_name))
mapping_df = pd.DataFrame(mapping, columns=['percolator_idx','run_name'])
mapping_df.to_csv(PERCOLATOR_MAPPING_FILE_NAME, index=False)

# load the feature pkl files from the whole experiment analysed by percolator, and associate them
# with their percolator index
df_l = []
for m in mapping:
    idx = m[0]
    run_name = m[1]
    df = pd.read_pickle('{}/{}-features.pkl'.format(FEATURES_DIR, run_name))
    df['percolator_idx'] = idx
    df_l.append(df)
# make a single df from the list of dfs
features_df = pd.concat(df_l, axis=0)
features_df.drop(['candidate_phr_error','envelope','mono_adjusted','original_phr','original_phr_error','precursor_id','rt_curve_fit','rt_lower','rt_upper','scan_curve_fit','scan_lower','scan_upper'], axis=1, inplace=True)
del df_l[:]

# merge the features with their precolator identifications, based on the file index and the
# feature id (embedded as the 'scan' in the percolator output)
percolator_df = pd.merge(psms_df, features_df, how='left', left_on=['file_idx','scan'], right_on=['percolator_idx','feature_id'])
percolator_df['human'] = percolator_df['protein id'].str.contains('HUMAN')
percolator_df = percolator_df[percolator_df['peptide mass'] > 0]

# add the mass of cysteine carbamidomethylation to the theoretical peptide mass from percolator,
# for the fixed modification of carbamidomethyl
percolator_df['monoisotopic_mass'] = (percolator_df.monoisotopic_mz * percolator_df.charge_y) - (PROTON_MASS * percolator_df.charge_y)
percolator_df['peptide_mass_mod'] = percolator_df['peptide mass'] + (percolator_df.sequence.str.count('C') * ADD_C_CYSTEINE_DA)

# now we can calculate the difference between the feature's monoisotopic mass and the
# theoretical peptide mass that is calculated from the sequence's molecular formula and its
# modifications
percolator_df['mass_accuracy_ppm'] = (percolator_df['monoisotopic_mass'] - percolator_df['peptide_mass_mod']) / percolator_df['peptide mass'] * 10**6
percolator_df['mass_error'] = percolator_df['monoisotopic_mass'] - percolator_df['peptide_mass_mod']
# filter out the identifications with high ppm error
percolator_df = percolator_df[(percolator_df.mass_accuracy_ppm >= -10) & (percolator_df.mass_accuracy_ppm <= 10)]

# for each feature file, produce a model that estimates the mass error from a feature's characteristics,
# and generate a revised feature file with adjusted mass, to get a smaller mass error on a second Comet search.
adjustments_l = ray.get([adjust_features.remote(file_idx, X_train=group_df[['monoisotopic_mz','scan_apex','rt_apex','intensity']].to_numpy(), y_train=group_df[['mass_error']].to_numpy()[:,0]) for file_idx,group_df in percolator_df.groupby('file_idx')])

for adjustment in adjustments_l:
    file_idx = adjustment[0]
    feature_recal_attributes_df = adjustment[1]
    best_estimator = adjustment[2]
    run_name = mapping_df[mapping_df.file_idx == file_idx].iloc[0].run_name

    # write out the recalibrated feature attributes so we can regenerate the MGF and search again
    RECALIBRATED_FEATURES_FILE_NAME = "{}/{}-recalibrated-features.pkl".format(RECALIBRATED_FEATURES_DIR, run_name)
    feature_recal_attributes_df.to_pickle(RECALIBRATED_FEATURES_FILE_NAME)
    print("Wrote the recalibrated features to {}".format(RECALIBRATED_FEATURES_FILE_NAME))

    # save the estimator
    with open(MASS_ERROR_ESTIMATOR_FILE_NAME, 'wb') as handle:
        pickle.dump(best_estimator, handle)
    print("Saved the estimator to {}".format(MASS_ERROR_ESTIMATOR_FILE_NAME))

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
