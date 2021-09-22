import pandas as pd
import numpy as np
import sys
import pickle
import glob
import os
import shutil
import random
import time
import argparse
from packaged.utils import get_run_names, get_percolator_run_mapping

def percolator_index_for_run_name(mapping_l, run_name):
    return [m[0] for m in mapping_l if m[1] == run_name][0]

def calculate_mono_mz(peptide_mass, charge):
    mono_mz = (peptide_mass + (PROTON_MASS * charge)) / charge
    return mono_mz

####################################################################

# This program builds a sequence attributes library based on all the runs in an experiment. For now it records the average attribute values across the experiment.
# nohup python -u ./open-path/pda/build-sequence-library.py -en dwm-test > build-sequence-library.log 2>&1 &

parser = argparse.ArgumentParser(description='Build run-specific coordinate estimators for the sequence-charges identified in the experiment.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
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

PERCOLATOR_OUTPUT_DIR = "{}/recalibrated-percolator-output".format(EXPERIMENT_DIR)
if not os.path.exists(PERCOLATOR_OUTPUT_DIR):
    print("The percolator output directory is required but doesn't exist: {}".format(PERCOLATOR_OUTPUT_DIR))
    sys.exit(1)

PERCOLATOR_OUTPUT_FILE_NAME = "{}/{}.percolator.target.psms.txt".format(PERCOLATOR_OUTPUT_DIR, args.experiment_name)
PERCOLATOR_STDOUT_FILE_NAME = "{}/{}.percolator.log.txt".format(PERCOLATOR_OUTPUT_DIR, args.experiment_name)
PERCOLATOR_MAPPING_FILE_NAME = "{}/percolator-idx-mapping.csv".format(PERCOLATOR_OUTPUT_DIR)

SEQUENCE_LIBRARY_DIR = "{}/sequence-library".format(EXPERIMENT_DIR)
if os.path.exists(SEQUENCE_LIBRARY_DIR):
    shutil.rmtree(SEQUENCE_LIBRARY_DIR)
os.makedirs(SEQUENCE_LIBRARY_DIR)
print("The sequence library directory was created: {}".format(SEQUENCE_LIBRARY_DIR))

PERCOLATOR_FEATURES_FILE_NAME = "{}/percolator-id-feature-mapping.pkl".format(SEQUENCE_LIBRARY_DIR)
SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.pkl".format(SEQUENCE_LIBRARY_DIR)

MAXIMUM_Q_VALUE = 0.01  # set the FDR to the standard 1% for the sequence library
ADD_C_CYSTEINE_DA = 57.021464  # from Unimod.org
PROTON_MASS = 1.007276

RECALIBRATED_FEATURES_DIR = "{}/recalibrated-features".format(EXPERIMENT_DIR)
if not os.path.exists(RECALIBRATED_FEATURES_DIR):
    print("The recalibrated features directory is required but doesn't exist: {}".format(RECALIBRATED_FEATURES_DIR))
    sys.exit(1)

start_run = time.time()

# get the run names for the experiment
run_names = get_run_names(EXPERIMENT_DIR)
print("found {} runs for this experiment: {}".format(len(run_names), run_names))

NUMBER_OF_RUNS_IN_EXPERIMENT = len(run_names)
MINIMUM_NUMBER_OF_FILES_SEQUENCE_IDENTIFIED = round(NUMBER_OF_RUNS_IN_EXPERIMENT / 2)

# get the mapping between percolator index and the run names
mapping_l = get_percolator_run_mapping(PERCOLATOR_STDOUT_FILE_NAME)
mapping_df = pd.DataFrame([(l[0],l[1]) for l in mapping_l], columns=['file_idx','run_name'])
mapping_df.to_csv(PERCOLATOR_MAPPING_FILE_NAME, index=False)
print("determined mapping between percolator index and run: {}; saved to {}".format(mapping_l, PERCOLATOR_MAPPING_FILE_NAME))

# go through all the runs in the experiment and gather the features
experiment_features_l = []
for run_name in run_names:
    print("consolidating the features found in run {}".format(run_name))
    recalibrated_features_dir = '{}/recalibrated-features/{}'.format(EXPERIMENT_DIR, run_name)

    # consolidate the features found in this run
    run_feature_files = glob.glob("{}/exp-{}-run-{}-recalibrated-features-precursor-*.pkl".format(recalibrated_features_dir, args.experiment_name, run_name))
    run_features_l = []
    print("found {} feature files for the run {}".format(len(run_feature_files), run_name))
    for file in run_feature_files:
        df = pd.read_pickle(file)
        run_features_l.append(df)
    # make a single df from the list of dfs
    run_features_df = pd.concat(run_features_l, axis=0, sort=False)
    run_features_df['percolator_idx'] = percolator_index_for_run_name(mapping_l, run_name)
    run_features_df['run_name'] = run_name
    del run_features_l[:]

    experiment_features_l.append(run_features_df)

# consolidate the features found across the experiment
EXPERIMENT_FEATURES_NAME = '{}/{}'.format(RECALIBRATED_FEATURES_DIR, 'experiment-features.pkl')
experiment_features_df = pd.concat(experiment_features_l, axis=0, sort=False)
print("saving {} experiment features to {}".format(len(experiment_features_df), EXPERIMENT_FEATURES_NAME))
experiment_features_df.to_pickle(EXPERIMENT_FEATURES_NAME)

# read the percolator identifications and throw away the poor quality ones
psms_df = pd.read_csv(PERCOLATOR_OUTPUT_FILE_NAME, sep='\t')
psms_df = psms_df[psms_df['percolator q-value'] <= MAXIMUM_Q_VALUE]

# merge the features with the experiment-wide percolator identifications
percolator_features_df = pd.merge(psms_df, experiment_features_df, how='left', left_on=['file_idx','scan'], right_on=['percolator_idx','feature_id'])
percolator_features_df['human'] = percolator_features_df['protein id'].str.contains('HUMAN')

# percolator_features_df contains all the sequence-charge identifications across the experiment and the detected feature for each
print("writing {} percolator-feature mappings to {}".format(len(percolator_features_df), PERCOLATOR_FEATURES_FILE_NAME))
percolator_features_df.to_pickle(PERCOLATOR_FEATURES_FILE_NAME)

# remove the rubbish peptide masses
percolator_features_df = percolator_features_df[percolator_features_df['peptide mass'] > 0]

# find the experiment-average for each sequence-charge identified
experiment_sequences_l = []
for group_name,group_df in percolator_features_df.groupby(['sequence','charge_x'], as_index=False):
    sequence = group_name[0]
    charge = group_name[1]
    peptide_mass = group_df.iloc[0]['peptide mass']
    peptide_mass_modification = peptide_mass + (sequence.count('C') * ADD_C_CYSTEINE_DA)
    theoretical_mz = calculate_mono_mz(peptide_mass=peptide_mass_modification, charge=charge)  # where the mono m/z should be, from the theoretical peptide mass
    experiment_scan_mean = np.mean(group_df.scan_apex)
    experiment_scan_std_dev = np.std(group_df.scan_apex)
    experiment_scan_peak_width = np.mean(group_df.scan_peak_width)
    experiment_rt_mean = np.mean(group_df.rt_apex)
    experiment_rt_std_dev = np.std(group_df.rt_apex)
    experiment_rt_peak_width = np.mean(group_df.rt_peak_width)
    experiment_intensity_mean = np.mean(group_df.intensity)
    experiment_intensity_std_dev = np.std(group_df.intensity)
    number_of_runs_identified = len(group_df.file_idx.unique())
    q_value = group_df.iloc[0]['percolator q-value']
    experiment_sequences_l.append((sequence, charge, theoretical_mz, experiment_scan_mean, experiment_scan_std_dev, experiment_scan_peak_width, experiment_rt_mean, experiment_rt_std_dev, experiment_rt_peak_width, experiment_intensity_mean, experiment_intensity_std_dev, number_of_runs_identified, q_value))

experiment_sequences_df = pd.DataFrame(experiment_sequences_l, columns=['sequence','charge','theoretical_mz', 'experiment_scan_mean', 'experiment_scan_std_dev', 'experiment_scan_peak_width', 'experiment_rt_mean', 'experiment_rt_std_dev', 'experiment_rt_peak_width', 'experiment_intensity_mean', 'experiment_intensity_std_dev', 'number_of_runs_identified', 'q_value'])
print("writing {} experiment-wide sequence attributes to {}".format(len(experiment_sequences_df), SEQUENCE_LIBRARY_FILE_NAME))
experiment_sequences_df.to_pickle(SEQUENCE_LIBRARY_FILE_NAME)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
