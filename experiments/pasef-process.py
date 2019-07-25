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

start_run = time.time()

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

def collate_spectra_for_feature(feature_df, ms2_deconvoluted_df):
    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    pairs_df = ms2_deconvoluted_df[['mz', 'intensity']].copy().sort_values(by=['mz'], ascending=True)
    spectrum = {}
    spectrum["m/z array"] = pairs_df.mz.values
    spectrum["intensity array"] = pairs_df.intensity.values.astype(int)
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

# Set up the processing pool
pool = Pool(processes=2)

# create a list of commands to start the ms1 and ms2 sub-processes and get them to join the cluster
processes = []
processes.append("python -W once -u {}/experiments/pasef-process-ms1.py -ini {} -os {} -rm join -ra {}".format(SOURCE_BASE, args.ini_file, args.operating_system, redis_address))
processes.append("python -u {}/experiments/pasef-process-ms2.py -ini {} -os {} -rm join -ra {}".format(SOURCE_BASE, args.ini_file, args.operating_system, redis_address))

# run the processes and wait for them to finish
pool.map(run_process, processes)

# get the features detected in ms1
ms1_features_df = pd.read_pickle(MS1_PEAK_PKL)

# get the ms2 spectra
ms2_deconvoluted_peaks_df = pd.read_pickle(DECONVOLUTED_MS2_PKL)

feature_results = []
for idx,feature_df in ms1_features_df.iterrows():
    # package the feature and its fragment ions for writing out to the MGF
    ms2_df = ms2_deconvoluted_peaks_df[ms2_deconvoluted_peaks_df.precursor == feature_df.precursor_id]
    result = collate_spectra_for_feature(feature_df, ms2_df)
    feature_results.append(result)

# generate the MGF for all the features
print("writing {} entries to {}".format(len(feature_results), MGF_NAME))
if os.path.isfile(MGF_NAME):
    os.remove(MGF_NAME)
_ = mgf.write(output=MGF_NAME, spectra=feature_results)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
