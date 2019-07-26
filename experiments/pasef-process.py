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

parser = argparse.ArgumentParser(description='Manage the ms1 and ms2 processing, and generate the MGF.')
parser.add_argument('-ini','--ini_file', type=str, help='Path to the config file.', required=True)
parser.add_argument('-os','--operating_system', type=str, choices=['linux','macos'], help='Operating system name.', required=True)
parser.add_argument('-di','--drop_indexes', action='store_true', help='Drop converted database indexes, if they exist.')
args = parser.parse_args()

if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.ini_file)

SOURCE_BASE = config.get(args.operating_system, 'SOURCE_BASE')
MS1_PEAK_PKL = config.get(args.operating_system, 'MS1_PEAK_PKL')
DECONVOLUTED_MS2_PKL = config.get(args.operating_system, 'DECONVOLUTED_MS2_PKL')
MGF_NAME = config.get(args.operating_system, 'MGF_NAME')
CONVERTED_DATABASE_NAME = config.get(args.operating_system, 'CONVERTED_DATABASE_NAME')

# clean up from last time
if os.path.isfile(MS1_PEAK_PKL):
    os.remove(MS1_PEAK_PKL)

if os.path.isfile(DECONVOLUTED_MS2_PKL):
    os.remove(DECONVOLUTED_MS2_PKL)

start_run = time.time()

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

# ms2_a is a numpy array [precursor_id,mz,intensity]
# return is a dictionary containing the feature information and spectra
def collate_spectra_for_feature(feature_df, ms2_a):
    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    ms2_sorted_a = ms2_a[ms2_a[:,1].argsort()] # sort by m/z increasing
    spectrum = {}
    spectrum["m/z array"] = ms2_sorted_a[:,1]
    spectrum["intensity array"] = ms2_sorted_a[:,2].astype(int)
    params = {}
    params["TITLE"] = "RawFile: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {} Precursor: {}".format(os.path.basename(CONVERTED_DATABASE_NAME).split('.')[0], feature_df.charge, round(feature_df.intensity), feature_df.feature_id, round(feature_df.rt_apex,2), feature_df.precursor_id)
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(feature_df.monoisotopic_mz,6), round(feature_df.intensity))
    params["CHARGE"] = "{}+".format(feature_df.charge)
    params["RTINSECONDS"] = "{}".format(round(feature_df.rt_apex,2))
    spectrum["params"] = params
    return spectrum

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

# Set up the processing pool
pool = Pool(processes=2)

# create a list of commands to start the ms1 and ms2 sub-processes and get them to join the cluster
processes = []
processes.append("python -W once -u {}/experiments/pasef-process-ms1.py -ini {} -os {} -rm join -ra {}".format(SOURCE_BASE, args.ini_file, args.operating_system, redis_address))
processes.append("python -W once -u {}/experiments/pasef-process-ms2.py -ini {} -os {} -rm join -ra {}".format(SOURCE_BASE, args.ini_file, args.operating_system, redis_address))

# run the processes and wait for them to finish
pool.map(run_process, processes)
print("Finished ms1 and ms2 processing")

if not os.path.isfile(MS1_PEAK_PKL):
    print("The ms1 output file doesn't exist: {}".format(MS1_PEAK_PKL))
    sys.exit(1)
else:
    # get the features detected in ms1
    print("Reading the ms1 output file: {}".format(MS1_PEAK_PKL))
    ms1_features_df = pd.read_pickle(MS1_PEAK_PKL)

if not os.path.isfile(DECONVOLUTED_MS2_PKL):
    print("The ms2 output file doesn't exist: {}".format(DECONVOLUTED_MS2_PKL))
    sys.exit(1)
else:
    # get the ms2 spectra
    print("Reading the ms2 output file: {}".format(DECONVOLUTED_MS2_PKL))
    ms2_deconvoluted_peaks_df = pd.read_pickle(DECONVOLUTED_MS2_PKL)

print("Associating ms2 spectra with ms1 features")
start_time = time.time()
feature_results = []
for i in range(len(features_df)):
    feature_df = features_df.iloc[i]
    # package the feature and its fragment ions for writing out to the MGF
    ms2_a = ms2_peaks_a[np.where(ms2_peaks_a[:,0] == feature_df.precursor_id)]
    result = collate_spectra_for_feature(feature_df, ms2_a)
    feature_results.append(result)
stop_time = time.time()
print("association time: {} seconds".format(round(stop_time-start_time,1)))

# generate the MGF for all the features
print("Writing {} entries to {}".format(len(feature_results), MGF_NAME))
if os.path.isfile(MGF_NAME):
    os.remove(MGF_NAME)
_ = mgf.write(output=MGF_NAME, spectra=feature_results)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
