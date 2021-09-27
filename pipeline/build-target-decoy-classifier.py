import pandas as pd
import numpy as np
import sys
import pickle
import sqlite3
import argparse
import os
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import math


def generate_estimator(X_train, X_test, y_train, y_test):
    if args.search_for_new_model_parameters:
        # do a randomised search to find the best classifier
        print('setting up randomised search')
        parameters = {
            "loss":["deviance"],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            # "min_samples_split": np.linspace(0.1, 0.5, 6),
            # "min_samples_leaf": np.linspace(0.1, 0.5, 6),
            "max_depth":[3, 5, 8, 20, 100],
            "max_features":["log2","sqrt"],
            "criterion": ["friedman_mse",  "mae"],
            "subsample":[0.6, 0.8, 1.0],
            "n_estimators":[50, 100, 1000, 2000]
            }
        # cross-validation splitting strategy uses 'cv' folds in a (Stratified)KFold
        rsearch = RandomizedSearchCV(GradientBoostingClassifier(), parameters, n_iter=100, n_jobs=-1, random_state=10, cv=5, scoring='accuracy', verbose=1)  # All scorer objects follow the convention that higher return values are better than lower return values, so we want the negated version for error metrics
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
        best_estimator = GradientBoostingClassifier(**best_params)
        best_estimator.fit(X_train, y_train)  # find the best fit within the parameter search space

    # calculate the estimator's score on the train and test sets
    print('evaluating against the training and test set')
    train_score = best_estimator.score(X_train, y_train)
    test_score = best_estimator.score(X_test, y_test)
    print("R-squared for training set: {}, test set: {}".format(round(train_score,2), round(test_score,2)))
    return best_estimator


####################################################################

# nohup python ./open-path/pda/build-target-decoy-classifier.py -en dwm-test > build-target-decoy-classifier.log 2>&1 &

parser = argparse.ArgumentParser(description='With metrics from each of the library sequences, from each run, build a feature targets and decoys training set.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-min','--minimum_number_files', type=int, default=10, help='For inclusion in the training set, the minimum number of files in which the sequence was identified.', required=False)
parser.add_argument('-tsm','--training_set_multiplier', type=int, default=10, help='Make the target training set this many times bigger than the decoy set.', required=False)
parser.add_argument('-snmp','--search_for_new_model_parameters', action='store_true', help='Search for new model parameters.')
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

# check the experiment metrics file exists
TARGET_DECOY_MODEL_DIR = "{}/target-decoy-models".format(EXPERIMENT_DIR)
METRICS_DB_NAME = "{}/experiment-metrics-for-library-sequences.sqlite".format(TARGET_DECOY_MODEL_DIR)
if not os.path.isfile(METRICS_DB_NAME):
    print("The experiment sequence metrics file doesn't exist: {}".format(METRICS_DB_NAME))
    sys.exit(1)

# create the index if it's not already there
db_conn = sqlite3.connect(METRICS_DB_NAME)
print("creating index in {}".format(METRICS_DB_NAME))
src_c = db_conn.cursor()
src_c.execute("create index if not exists idx_extracted_metrics_1 on extracted_metrics (number_of_runs_identified)")
# load the sequences
print("loading metrics from {}".format(METRICS_DB_NAME))
metrics_df = pd.read_sql_query('select target_metrics,decoy_metrics from extracted_metrics where number_of_runs_identified >= {}'.format(args.minimum_number_files), db_conn)
db_conn.close()
print("loaded {} metrics for library sequences that satisfy the criteria for inclusion in the training set from {}".format(len(metrics_df), METRICS_DB_NAME))

# now we can build the training set
print("building the training set")

if len(metrics_df) > 0:
    # unpack the metrics from each sequence
    metrics = []
    metrics_names = None
    for row in metrics_df.itertuples():
        # target metrics
        target_metrics = json.loads(row.target_metrics)
        if isinstance(target_metrics, dict):
            l = [target_metrics[key] for key in sorted(target_metrics)]
            l.append('target')
            metrics.append(tuple(l))
            if metrics_names == None:
                metrics_names = [key for key in sorted(target_metrics)]

            # decoy metrics
            decoy_metrics = json.loads(row.decoy_metrics)
            if isinstance(decoy_metrics, dict):
                l = [decoy_metrics[key] for key in sorted(decoy_metrics)]
                l.append('decoy')
                metrics.append(tuple(l))

    columns = metrics_names.copy()
    columns.append('class_name')

    metrics_df = pd.DataFrame(metrics, columns=columns)

    # tidy up any attributes that will upset the model training
    metrics_df.fillna(value=0.0, inplace=True)
    metrics_df.replace(to_replace=-math.inf, value=0, inplace=True)

    # down-sample the target class to balance the classes
    target_class_df = metrics_df[(metrics_df.class_name == 'target')]
    decoy_class_df = metrics_df[(metrics_df.class_name == 'decoy')]
    print('prior to down-sampling, targets {}, decoys {}'.format(len(target_class_df), len(decoy_class_df)))
    number_of_targets_for_training_set = args.training_set_multiplier * len(decoy_class_df)
    if len(target_class_df) > number_of_targets_for_training_set:
        target_class_df = target_class_df.sample(n=number_of_targets_for_training_set)  # even them up somewhat, sort-of
    metrics_df = pd.concat([target_class_df, decoy_class_df], ignore_index=True)

    # set up the train and test sets
    X = metrics_df[metrics_names].values
    y = metrics_df[['class_name']].values[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print('training set targets: {}, decoys: {}'.format(np.count_nonzero(y_train == 'target'), np.count_nonzero(y_train == 'decoy')))
    print('test set targets: {}, decoys: {}'.format(np.count_nonzero(y_test == 'target'), np.count_nonzero(y_test == 'decoy')))

    # save the training set alongside the model
    np.save('{}/X_train.npy'.format(TARGET_DECOY_MODEL_DIR), X_train)
    np.save('{}/y_train.npy'.format(TARGET_DECOY_MODEL_DIR), y_train)
    np.save('{}/X_test.npy'.format(TARGET_DECOY_MODEL_DIR), X_test)
    np.save('{}/y_test.npy'.format(TARGET_DECOY_MODEL_DIR), y_test)
    np.save('{}/feature_names.npy'.format(TARGET_DECOY_MODEL_DIR), np.array(metrics_names))

    best_estimator = generate_estimator(X_train, X_test, y_train, y_test)

    # save the classifier
    CLASSIFIER_FILE_NAME = "{}/target-decoy-classifier.pkl".format(TARGET_DECOY_MODEL_DIR)
    print("saving the classifier to {}".format(CLASSIFIER_FILE_NAME))
    with open(CLASSIFIER_FILE_NAME, 'wb') as file:
        pickle.dump(best_estimator, file)

    print("make predictions on the test set")
    predictions = best_estimator.predict(X_test)
    class_probabilities = best_estimator.predict_proba(X_test)
    np.save('{}/class_probabilities.npy'.format(TARGET_DECOY_MODEL_DIR), class_probabilities, allow_pickle=False)

    # display some interesting model attributes
    cm = confusion_matrix(y_test, predictions, labels=["target", "decoy"])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix:")
    print(cm)
    print('false discovery rate (i.e. decoy was identified as a target): {}'.format(cm[1,0]))
    print()
    print("Classification Report")
    print(classification_report(y_test, predictions))
else:
    print("there are no sequences that meet the criteria for use in the training set.")

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
