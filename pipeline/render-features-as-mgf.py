import pandas as pd
import numpy as np
import sqlite3
import json
import time
import argparse
import os
import pickle
from pyteomics import mgf
import shutil

# collate the feature attributes for MGF rendering
def collate_spectra_for_feature(feature_d, run_name):
    # sort the fragment ions by increasing m/z
    fragment_ions_df = pd.DataFrame(json.loads(feature_d['fragment_ions_l']))
    fragment_ions_df.sort_values(by=['decharged_mass'], ascending=True, inplace=True)

    spectrum = {}
    spectrum["m/z array"] = fragment_ions_df['decharged_mass'].to_numpy(dtype='float')
    spectrum["intensity array"] = fragment_ions_df['intensity'].to_numpy(dtype='uint')

    params = {}
    params["TITLE"] = "RawFile: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {} Precursor: {}".format(run_name, int(feature_d['charge']), int(feature_d['feature_intensity']), int(feature_d['feature_id']), round(feature_d['rt_apex'],2), int(feature_d['precursor_cuboid_id']))
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(feature_d['monoisotopic_mz'],6), int(feature_d['feature_intensity']))
    params["CHARGE"] = "{}+".format(int(feature_d['charge']))
    params["RTINSECONDS"] = "{}".format(round(feature_d['rt_apex'],2))
    params["SCANS"] = "{}".format(int(feature_d['feature_id']))

    spectrum["params"] = params
    return spectrum


###################################
parser = argparse.ArgumentParser(description='Render the detected features as an MGF.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-fdm','--feature_detection_method', type=str, choices=['pasef','3did'], help='Which feature detection method.', required=True)
parser.add_argument('-pid', '--precursor_id', type=int, help='Only process this precursor ID.', required=False)
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

EXPERIMENT_DIR = '{}/{}'.format(args.experiment_base_dir, args.experiment_name)
FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, args.feature_detection_method)
FEATURES_DEDUP_FILE = '{}/exp-{}-run-{}-features-{}-dedup.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name, args.feature_detection_method)

if not os.path.isfile(FEATURES_DEDUP_FILE):
    print("The features file is required but doesn't exist: {}".format(FEATURES_DEDUP_FILE))
    sys.exit(1)

# load the features
with open(FEATURES_DEDUP_FILE, 'rb') as handle:
    d = pickle.load(handle)
features_df = d['features_dedup_df']

# trim down the features to just those from the specified precursor_id
if args.precursor_id is not None:
    features_df = features_df[(features_df.precursor_cuboid_id == args.precursor_id)]
print('loaded {} features from {}'.format(len(features_df), FEATURES_DEDUP_FILE))

# output MGF
MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, args.feature_detection_method)
MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, args.experiment_name, args.run_name, args.feature_detection_method)

# set up the output directory
if os.path.exists(MGF_DIR):
    shutil.rmtree(MGF_DIR)
os.makedirs(MGF_DIR)

# associate the spectra with each feature found for this precursor
associations = []
for row in features_df.itertuples():
    # collate them for the MGF
    spectrum = collate_spectra_for_feature(feature_d=row._asdict(), run_name=args.run_name)
    associations.append(spectrum)

# generate the MGF for all the features
print("writing {} entries to {}".format(len(associations), MGF_FILE))
if os.path.isfile(MGF_FILE):
    os.remove(MGF_FILE)
_ = mgf.write(output=MGF_FILE, spectra=associations)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
