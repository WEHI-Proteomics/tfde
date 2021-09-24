import pandas as pd
import numpy as np
import sys
import os
import shutil
import time
import argparse
import configparser
from configparser import ExtendedInterpolation


def calculate_mono_mz(peptide_mass, charge):
    mono_mz = (peptide_mass + (PROTON_MASS * charge)) / charge
    return mono_mz


####################################################################

# This program builds a sequence attributes library based on all the runs in an experiment. For now it records the average attribute values across the experiment.

parser = argparse.ArgumentParser(description='Build a library of sequence attributes based on all the identifications in all the runs in an experiment.')
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

SEQUENCE_LIBRARY_DIR = "{}/sequence-library".format(EXPERIMENT_DIR)
if os.path.exists(SEQUENCE_LIBRARY_DIR):
    shutil.rmtree(SEQUENCE_LIBRARY_DIR)
os.makedirs(SEQUENCE_LIBRARY_DIR)
print("The sequence library directory was created: {}".format(SEQUENCE_LIBRARY_DIR))

SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.feather".format(SEQUENCE_LIBRARY_DIR)

IDENTIFICATIONS_DIR = '{}/identifications-pasef'.format(EXPERIMENT_DIR)
if not os.path.exists(IDENTIFICATIONS_DIR):
    print("The identifications directory is required but doesn't exist: {}".format(IDENTIFICATIONS_DIR))
    sys.exit(1)

IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-pasef-recalibrated.feather'.format(IDENTIFICATIONS_DIR, args.experiment_name)
if not os.path.isfile(IDENTIFICATIONS_FILE):
    print("The identifications file doesn't exist: {}".format(IDENTIFICATIONS_FILE))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
PROTON_MASS = cfg.getfloat('common','PROTON_MASS')

start_run = time.time()

# load the experiment identifications
identifications_df = pd.read_feather(IDENTIFICATIONS_FILE)
print('loaded {} identifications from {}'.format(len(identifications_df), IDENTIFICATIONS_FILE))

# find the experiment-average for each sequence-charge identified
experiment_sequences_l = []
for group_name,group_df in identifications_df.groupby(['sequence','charge'], as_index=False):
    sequence = group_name[0]
    charge = group_name[1]
    theoretical_peptide_mass = group_df.iloc[0].theoretical_peptide_mass
    theoretical_mz = calculate_mono_mz(peptide_mass=theoretical_peptide_mass, charge=charge)  # where the mono m/z should be, from the theoretical peptide mass
    experiment_scan_mean = np.mean(group_df.scan_apex)
    experiment_scan_std_dev = np.std(group_df.scan_apex)
    experiment_scan_peak_width = np.mean(group_df.scan_upper - group_df.scan_lower)
    experiment_rt_mean = np.mean(group_df.rt_apex)
    experiment_rt_std_dev = np.std(group_df.rt_apex)
    experiment_rt_peak_width = np.mean(group_df.rt_upper - group_df.rt_lower)
    experiment_intensity_mean = np.mean(group_df.feature_intensity)
    experiment_intensity_std_dev = np.std(group_df.feature_intensity)
    number_of_runs_identified = len(group_df.run_name.unique())
    q_value = group_df.iloc[0]['percolator q-value']
    experiment_sequences_l.append((sequence, charge, theoretical_mz, experiment_scan_mean, experiment_scan_std_dev, experiment_scan_peak_width, experiment_rt_mean, experiment_rt_std_dev, experiment_rt_peak_width, experiment_intensity_mean, experiment_intensity_std_dev, number_of_runs_identified, q_value))

experiment_sequences_df = pd.DataFrame(experiment_sequences_l, columns=['sequence','charge','theoretical_mz', 'experiment_scan_mean', 'experiment_scan_std_dev', 'experiment_scan_peak_width', 'experiment_rt_mean', 'experiment_rt_std_dev', 'experiment_rt_peak_width', 'experiment_intensity_mean', 'experiment_intensity_std_dev', 'number_of_runs_identified', 'q_value'])
print("writing {} experiment-wide sequence attributes to {}".format(len(experiment_sequences_df), SEQUENCE_LIBRARY_FILE_NAME))
experiment_sequences_df.to_feather(SEQUENCE_LIBRARY_FILE_NAME)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
