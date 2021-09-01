import sys
import time
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import json
import pandas as pd


#######################
parser = argparse.ArgumentParser(description='Use the trained model to classify the 3DID features detected for their identifiability.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
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

# check the detected features
FEATURES_DIR = "{}/features-3did".format(EXPERIMENT_DIR)
if not os.path.exists(FEATURES_DIR):
    print("The features directory is required but doesn't exist: {}".format(FEATURES_DIR))
    sys.exit(1)

FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.feather'.format(FEATURES_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(FEATURES_FILE):
    print("The detected features file is required but doesn't exist: {}".format(FEATURES_FILE))
    sys.exit(1)

FEATURES_METADATA_FILE = '{}/exp-{}-run-{}-features-3did.json'.format(FEATURES_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(FEATURES_METADATA_FILE):
    print("The detected features metadata file is required but doesn't exist: {}".format(FEATURES_METADATA_FILE))
    sys.exit(1)

# output files
FEATURES_IDENT_FILE = '{}/exp-{}-run-{}-features-3did-ident.feather'.format(FEATURES_DIR, args.experiment_name, args.run_name)
FEATURES_IDENT_METADATA_FILE = '{}/exp-{}-run-{}-features-3did-ident-metadata.json'.format(FEATURES_DIR, args.experiment_name, args.run_name)

# check the trained model
MODEL_DIR = '{}/features-3did-classifier'.format(EXPERIMENT_DIR)
if not os.path.exists(MODEL_DIR):
    print("The trained model is required but doesn't exist: {}".format(MODEL_DIR))
    sys.exit(1)

# load the trained model
print('loading the trained model from {}'.format(MODEL_DIR))
model = keras.models.load_model(MODEL_DIR)

# load the features detected
features_df = pd.read_feather(FEATURES_FILE)
features_df.fillna(0, inplace=True)
# ... and the features metadata
with open(FEATURES_METADATA_FILE) as handle:
    features_metadata = json.load(handle)

# use the model to predict their identifiability
input_names = ['deconvolution_score','coelution_coefficient','mobility_coefficient','isotope_count']
predictions = model.predict(features_df[input_names].to_numpy())
features_df['prediction'] = predictions
features_df['identification_predicted'] = features_df.apply(lambda row: row.prediction >= 0.5, axis=1)

# update the original detected features file with the predictions for later analysis
print('updating {} features with predictions: {}'.format(len(features_df), FEATURES_FILE))
# save the features
features_df.reset_index(drop=True).to_feather(FEATURES_FILE)

# strip out previous predictions entry in the features metadata
features_metadata = [x for x in features_metadata if 'predictions' != x[0]]
# add predictions entry to metadata
l = []
l.append(('processor', parser.prog))
l.append(('processed', time.ctime()))
l.append(('model', MODEL_DIR))
features_metadata.append(('predictions',l))
# save the metadata file
with open(FEATURES_METADATA_FILE, 'w') as handle:
    json.dump(features_metadata, handle)

# filter out the features unlikely to be identified
features_df = features_df[(features_df.identification_predicted == True)]

# ... and write them to the output file
print()
print('saving {} features classified as identifiable to {}'.format(len(features_df), FEATURES_IDENT_FILE))
features_df.reset_index(drop=True).to_feather(FEATURES_IDENT_FILE)

info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor', parser.prog))
info.append(('processed', time.ctime()))
info.append(('model', MODEL_DIR))
# save the metadata file
with open(FEATURES_IDENT_METADATA_FILE, 'w') as handle:
    json.dump(features_metadata, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
