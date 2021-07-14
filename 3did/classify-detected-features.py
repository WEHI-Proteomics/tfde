import sys
import pickle
import time
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

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

FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(FEATURES_FILE):
    print("The detected features file is required but doesn't exist: {}".format(FEATURES_FILE))
    sys.exit(1)

FEATURES_IDENT_FILE = '{}/exp-{}-run-{}-features-3did-ident.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name)

# check the trained model
MODEL_DIR = '{}/classifier'.format(FEATURES_DIR)
if not os.path.exists(MODEL_DIR):
    print("The trained model is required but doesn't exist: {}".format(MODEL_DIR))
    sys.exit(1)

# load the trained model
print('loading the trained model from {}'.format(MODEL_DIR))
model = keras.models.load_model(MODEL_DIR)

# load the features detected
with open(FEATURES_FILE, 'rb') as handle:
    d = pickle.load(handle)
features_df = d['features_df']
features_metadata = d['metadata']
features_df.fillna(0, inplace=True)

# use the model to predict their identifiability
input_names = ['deconvolution_score','coelution_coefficient','mobility_coefficient','isotope_count']
predictions = model.predict(features_df[input_names].to_numpy())
features_df['prediction'] = predictions
features_df['identification_predicted'] = features_df.apply(lambda row: row.prediction >= 0.5, axis=1)

# update the detected features with the predictions
print('updating {} features with predictions: {}'.format(len(features_df), FEATURES_FILE))
l = []
l.append(('processor',parser.prog))
l.append(('processed', time.ctime()))
l.append(('model', MODEL_DIR))
features_metadata.append({'predictions':l})
content_d = {'features_df':features_df, 'metadata':features_metadata}
with open(FEATURES_FILE, 'wb') as handle:
    pickle.dump(content_d, handle)

# filter out the features unlikely to be identified
features_df = features_df[(features_df.identification_predicted == True)]

print()
print('saving {} features classified as identifiable to {}'.format(len(features_df), FEATURES_IDENT_FILE))
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
info.append(('model', MODEL_DIR))
content_d = {'features_df':features_df, 'metadata':info}
with open(FEATURES_IDENT_FILE, 'wb') as handle:
    pickle.dump(content_d, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
