# Associates ms2 spectra with features detected in ms1.

import sqlite3
import pandas as pd
import numpy as np
import sys
import os.path
from pyteomics import mgf
import os
import multiprocessing as mp
from multiprocessing import Pool
import configparser
from configparser import ExtendedInterpolation
import os.path
import argparse
import ray
import time
import pickle
import glob

parser = argparse.ArgumentParser(description='Manage the ms1 and ms2 processing, and generate the MGF.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-ini','--ini_file', type=str, help='Path to the config file.', required=True)
parser.add_argument('-os','--operating_system', type=str, choices=['linux','macos'], help='Operating system name.', required=True)
parser.add_argument('-di','--drop_indexes', action='store_true', help='Drop converted database indexes, if they exist.')
parser.add_argument('-ao','--association_only', action='store_true', help='Bypass the ms1 and ms2 processing; only do the association step.')
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.')
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use the recalibrated features.')
args = parser.parse_args()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the raw database exists
raw_files_l = glob.glob("{}/raw-databases/{}*.d".format(EXPERIMENT_DIR, args.run_name))
if (len(raw_files_l) > 0):
    RAW_DATABASE_FILE_NAME = "{}/analysis.tdf".format(raw_files_l[0])
    if not os.path.isfile(RAW_DATABASE_FILE_NAME):
        print("The raw database is required but doesn't exist: {}".format(RAW_DATABASE_FILE_NAME))
        sys.exit(1)
else:
    print("The raw database is required but doesn't exist: {}".format(RAW_DATABASE_FILE_NAME))
    sys.exit(1)

if args.recalibration_mode:
    if not args.association_only:
        print("The association_only argument should be set but is not.")
        sys.exit(1)

    MS1_PEAK_PKL = "{}/recalibrated-features/{}-recalibrated-features.pkl".format(EXPERIMENT_DIR, args.run_name)
    monoisotopic_column_name = 'recalibrated_monoisotopic_mz'
    MGF_DIR = "{}/recalibrated-mgfs"
    MGF_NAME = "{}/{}-recalibrated-search.mgf".format(MGF_DIR, args.run_name)
else:
    MS1_PEAK_PKL = "{}/features/{}-features.pkl".format(EXPERIMENT_DIR, args.run_name)
    monoisotopic_column_name = 'monoisotopic_mz'
    MGF_DIR = "{}/mgfs"
    MGF_NAME = "{}/{}-search.mgf".format(MGF_DIR, args.run_name)

DECONVOLUTED_MS2_PKL = "{}/ms2-spectra/{}-ms2-spectra.pkl".format(EXPERIMENT_DIR, args.run_name)
ASSOCIATIONS_PKL = "{}/associations/{}-associations.pkl".format(EXPERIMENT_DIR, args.run_name)
CONVERTED_DATABASE_NAME = "{}/converted-databases/{}-converted.sqlite".format(EXPERIMENT_DIR, args.run_name)

# check the converted database exists
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

# check the configuration file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file is required but doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# read the configuration items
config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.ini_file)

SOURCE_BASE = config.get(args.operating_system, 'SOURCE_BASE')

start_run = time.time()

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

# feature_a is a numpy array [feature_id,charge,monoisotopic_mz,rt_apex,intensity,precursor_id]
# ms2_a is a numpy array [precursor_id,mz,intensity]
# return is a dictionary containing the feature information and spectra
def collate_spectra_for_feature(feature_a, ms2_a):
    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    ms2_sorted_a = ms2_a[ms2_a[:,1].argsort()] # sort by m/z increasing
    spectrum = {}
    spectrum["m/z array"] = ms2_sorted_a[:,1]
    spectrum["intensity array"] = ms2_sorted_a[:,2].astype(int)
    params = {}
    params["TITLE"] = "RawFile: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {} Precursor: {}".format(args.processing_name, int(feature_a[1]), int(feature_a[4]), int(feature_a[0]), round(feature_a[3],2), int(feature_a[5]))
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(feature_a[2],6), int(feature_a[4]))
    params["CHARGE"] = "{}+".format(int(feature_a[1]))
    params["RTINSECONDS"] = "{}".format(round(feature_a[3],2))
    params["SCANS"] = "{}".format(int(feature_a[0]))
    spectrum["params"] = params
    return spectrum

# features_a is a numpy array [feature_id,charge,monoisotopic_mz,rt_apex,intensity,precursor_id]
def associate_feature_spectra(features_a, spectra_a):
    associations = []
    # associate the spectra with each feature found for this precursor
    for i in range(len(features_a)):
        feature_a = features_a[i]
        # collate them for the MGF
        spectrum = collate_spectra_for_feature(feature_a=feature_a, ms2_a=spectra_a)
        associations.append(spectrum)
    return associations

##################################################
ray_cluster = ray.init()
# get the address for the sub-processes to join
redis_address = ray_cluster['redis_address']

# make sure the right indexes are created in the source database
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
src_c = db_conn.cursor()

if args.drop_indexes:
    print("Dropping indexes on {}".format(CONVERTED_DATABASE_NAME))
    src_c.execute("drop index if exists idx_pasef_frames_1")
    src_c.execute("drop index if exists idx_pasef_frame_properties_1")

print("Setting up indexes on {}".format(CONVERTED_DATABASE_NAME))
src_c.execute("create index if not exists idx_pasef_frames_1 on frames (frame_id, mz, scan, intensity, retention_time_secs)")
src_c.execute("create index if not exists idx_pasef_frame_properties_1 on frame_properties (retention_time_secs, collision_energy)")

db_conn.close()

if not args.association_only:
    # Set up the processing pool
    pool = Pool(processes=2)

    # create a list of commands to start the ms1 and ms2 sub-processes and get them to join the cluster
    processes = []
    ms1_command = "python -u {}/pda/pasef-process-ms1.py -rdb {} -bpd {} -pn {} -ini {} -os {} -rm join -ra {}".format(SOURCE_BASE, args.raw_database_dir, args.base_processing_dir, args.processing_name, args.ini_file, args.operating_system, redis_address)
    ms2_command = "python -u {}/pda/pasef-process-ms2.py -rdb {} -bpd {} -pn {} -ini {} -os {} -rm join -ra {}".format(SOURCE_BASE, args.raw_database_dir, args.base_processing_dir, args.processing_name, args.ini_file, args.operating_system, redis_address)
    if args.small_set_mode:
        ms1_command += " -ssm"
        ms2_command += " -ssm"
    processes.append(ms1_command)
    processes.append(ms2_command)

    # run the processes and wait for them to finish
    pool.map(run_process, processes)
    print("Finished ms1 and ms2 processing")

if not os.path.isfile(MS1_PEAK_PKL):
    print("The ms1 output file is required but doesn't exist: {}".format(MS1_PEAK_PKL))
    sys.exit(1)
else:
    # get the features detected in ms1
    features_df = pd.read_pickle(MS1_PEAK_PKL)
    print("Loaded {} features from {}".format(len(features_df), MS1_PEAK_PKL))
    features_a = features_df[['feature_id','charge',monoisotopic_column_name,'rt_apex','intensity','precursor_id']].to_numpy()

if not os.path.isfile(DECONVOLUTED_MS2_PKL):
    print("The ms2 output file is required doesn't exist: {}".format(DECONVOLUTED_MS2_PKL))
    sys.exit(1)
else:
    # get the ms2 spectra
    ms2_peaks_df = pd.read_pickle(DECONVOLUTED_MS2_PKL)
    print("Loaded {} peaks from {}".format(len(ms2_peaks_df), DECONVOLUTED_MS2_PKL))
    ms2_peaks_a = ms2_peaks_df[['precursor','mz','intensity']].to_numpy()

print("Associating ms2 spectra with ms1 features")
start_time = time.time()
unique_precursor_ids_a = features_df.precursor_id.unique()
print("there are {} unique precursors to use for association".format(len(unique_precursor_ids_a)))
associations = []
for precursor_id in unique_precursor_ids_a:
    association = associate_feature_spectra(features_a[np.where(features_a[:,5] == precursor_id)], ms2_peaks_a[np.where(ms2_peaks_a[:,0] == precursor_id)])
    associations.append(association)
associations = [item for sublist in associations for item in sublist]  # associations is a list of the associated ms1 features and ms2 spectra
stop_time = time.time()
print("association time: {} seconds".format(round(stop_time-start_time,1)))

print("writing associations to {}".format(ASSOCIATIONS_PKL))
with open(ASSOCIATIONS_PKL, 'wb') as f:
    pickle.dump(associations, f)

# generate the MGF for all the features
print("Writing {} entries to {}".format(len(associations), MGF_NAME))
if os.path.isfile(MGF_NAME):
    os.remove(MGF_NAME)
_ = mgf.write(output=MGF_NAME, spectra=associations)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
