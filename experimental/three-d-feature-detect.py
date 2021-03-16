import pandas as pd
import numpy as np
import sys
import os.path
import argparse
import time
import json
import multiprocessing as mp
import ray
import sqlite3
import shutil
from ms_deisotope import deconvolute_peaks, averagine, scoring
from ms_deisotope.deconvolution import peak_retention_strategy

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# include points from either side of the intense point, in Da
MS1_PEAK_DELTA = 0.1

PROTON_MASS = 1.007276

# takes a numpy array of intensity, and another of mz
def mz_centroid(_int_f, _mz_f):
    return ((_int_f/_int_f.sum()) * _mz_f).sum()

# ms1_peaks_a is a numpy array of [mz,intensity]
# returns a numpy array of [mz_centroid,summed_intensity]
def ms1_intensity_descent(ms1_peaks_a, ms1_peak_delta):
    # intensity descent
    ms1_peaks_l = []
    while len(ms1_peaks_a) > 0:
        # find the most intense point
        max_intensity_index = np.argmax(ms1_peaks_a[:,1])
        peak_mz = ms1_peaks_a[max_intensity_index,0]
        peak_mz_lower = peak_mz - ms1_peak_delta
        peak_mz_upper = peak_mz + ms1_peak_delta

        # get all the raw points within this m/z region
        peak_indexes = np.where((ms1_peaks_a[:,0] >= peak_mz_lower) & (ms1_peaks_a[:,0] <= peak_mz_upper))[0]
        if len(peak_indexes) > 0:
            mz_cent = mz_centroid(ms1_peaks_a[peak_indexes,1], ms1_peaks_a[peak_indexes,0])
            summed_intensity = ms1_peaks_a[peak_indexes,1].sum()
            ms1_peaks_l.append((mz_cent, summed_intensity))
            # remove the raw points assigned to this peak
            ms1_peaks_a = np.delete(ms1_peaks_a, peak_indexes, axis=0)
    return np.array(ms1_peaks_l)

# process a precursor cuboid to detect ms1 features
def ms1(precursor_metadata, ms1_points_df, args):
    # find features in the cuboid
    features_df = find_features(precursor_metadata, ms1_points_df, args)
    features_df['monoisotopic_mass'] = (features_df.monoisotopic_mz * features_df.charge) - (args.PROTON_MASS * features_df.charge)
    print("found {} features for precursor {}".format(len(features_df), precursor_metadata['precursor_id']))
    return checked_features_df

# prepare the metadata and raw points for the feature detection
@ray.remote
def detect_ms1_features(precursor_cuboid_row, converted_db_name):
    # load the raw points for this cuboid
    db_conn = sqlite3.connect(converted_db_name)
    ms1_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {}".format(FRAME_TYPE_MS1, precursor_cuboid_row.mz_lower, precursor_cuboid_row.mz_upper, precursor_cuboid_row.scan_lower, precursor_cuboid_row.scan_upper, precursor_cuboid_row.rt_lower, precursor_cuboid_row.rt_upper), db_conn)
    db_conn.close()

    # intensity descent
    raw_points_a = ms1_points_df[['mz','intensity']].to_numpy()
    peaks_a = ms1_intensity_descent(raw_points_a, MS1_PEAK_DELTA)

    # deconvolution
    ms1_peaks_l = list(map(tuple, peaks_a))
    deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, use_quick_charge=True, averagine=averagine.peptide, charge_range=(1,5), error_tolerance=5.0, scorer=scoring.MSDeconVFitter(100.0), truncate_after=0.95)

    # collect features from deconvolution
    ms1_deconvoluted_peaks_l = []
    for peak_idx,peak in enumerate(deconvoluted_peaks):
        # discard a monoisotopic peak that has either of the first two peaks as placeholders (indicated by intensity of 1)
        if ((len(peak.envelope) >= 3) and (peak.envelope[0][1] > 1) and (peak.envelope[1][1] > 1)):
            mono_peak_mz = peak.mz
            mono_intensity = peak.intensity
            second_peak_mz = peak.envelope[1][0]
            ms1_deconvoluted_peaks_l.append((mono_peak_mz, second_peak_mz, mono_intensity, peak.score, peak.signal_to_noise, peak.charge, peak.envelope, peak.neutral_mass))
    deconvolution_features_df = pd.DataFrame(ms1_deconvoluted_peaks_l, columns=['mono_mz','second_peak_mz','intensity','score','SN','charge','envelope','neutral_mass'])

    # determine the feature attributes
    feature_l = []
    for idx,row in enumerate(deconvolution_features_df.itertuples()):
        feature_d = {}
        # from deconvolution for this feature
        feature_d['monoisotopic_mz'] = row.mono_mz
        feature_d['charge'] = row.charge
        feature_d['monoisotopic_mass'] = (feature_d['monoisotopic_mz'] * feature_d['charge']) - (PROTON_MASS * feature_d['charge'])
        feature_d['intensity'] = row.intensity
        feature_d['envelope'] = json.dumps([tuple(e) for e in row.envelope])
        feature_d['isotope_count'] = len(row.envelope)
        feature_d['deconvolution_score'] = row.score
        # from the precursor cuboid
        feature_d['precursor_id'] = precursor_cuboid_row.precursor_cuboid_id
        feature_d['scan_apex'] = precursor_cuboid_row.anchor_point_scan
        feature_d['scan_lower'] = precursor_cuboid_row.scan_lower
        feature_d['scan_upper'] = precursor_cuboid_row.scan_upper
        feature_d['rt_apex'] = precursor_cuboid_row.anchor_point_retention_time_secs
        feature_d['rt_lower'] = precursor_cuboid_row.rt_lower
        feature_d['rt_upper'] = precursor_cuboid_row.rt_upper
        feature_d['feature_id'] = generate_feature_id(precursor_cuboid_row.precursor_cuboid_id, idx+1)
        # add it to the list
        feature_l.append(feature_d)
    features_df = pd.DataFrame(feature_l)

    # assign each feature an identifier based on the precursor it was found in
    features_df['feature_id'] = np.arange(start=1, stop=len(features_df)+1)
    features_df['feature_id'] = features_df['feature_id'].apply(lambda x: generate_feature_id(precursor_cuboid_row.precursor_cuboid_id, x))

    print("found {} features for precursor {}".format(len(features_df), precursor_cuboid_row.precursor_cuboid_id))
    return features_df

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers

# generate a unique feature_id from the precursor id and the feature sequence number found for that precursor
def generate_feature_id(precursor_id, feature_sequence_number):
    feature_id = (precursor_id * 100) + feature_sequence_number  # assumes there will not be more than 99 features found for a precursor
    return feature_id

###################################
parser = argparse.ArgumentParser(description='Detect the features precursor cuboids found in a run with 3D intensity descent.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-ml','--mz_lower', type=int, default='100', help='Lower limit for m/z.', required=False)
parser.add_argument('-mu','--mz_upper', type=int, default='1700', help='Upper limit for m/z.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./open-path/pda/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-pid', '--precursor_id', type=int, help='Only process this precursor ID.', required=False)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], help='The Ray mode to use.', required=True)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.9, help='Proportion of the machine\'s cores to use for this program.', required=False)
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

# check the converted databases directory exists
CONVERTED_DATABASE_NAME = "{}/converted-databases/exp-{}-run-{}-converted.sqlite".format(EXPERIMENT_DIR, args.experiment_name, args.run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

CUBOIDS_DIR = "{}/precursor-cuboids-3did".format(EXPERIMENT_DIR)
CUBOIDS_FILE = '{}/exp-{}-run-{}-mz-{}-{}-precursor-cuboids.pkl'.format(CUBOIDS_DIR, args.experiment_name, args.run_name, args.mz_lower, args.mz_upper)

# check the cuboids file
if not os.path.isfile(CUBOIDS_FILE):
    print("The cuboids file is required but doesn't exist: {}".format(CUBOIDS_FILE))
    sys.exit(1)

# load the precursor cuboids
precursor_cuboids_df = pd.read_pickle(CUBOIDS_FILE)
print('loaded {} precursor cuboids from {}'.format(len(precursor_cuboids_df), CUBOIDS_FILE))

# limit the cuboids to just the selected one
if args.precursor_id is not None:
    precursor_cuboids_df = precursor_cuboids_df[(precursor_cuboids_df.precursor_cuboid_id == args.precursor_id)]

FEATURES_DIR = "{}/features-3did".format(EXPERIMENT_DIR)
FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name)

if args.precursor_id is None:
    # set up the output directory
    if os.path.exists(FEATURES_DIR):
        shutil.rmtree(FEATURES_DIR)
    os.makedirs(FEATURES_DIR)

# set up Ray
print("setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

# find the features in each precursor cuboid
features_l = ray.get([detect_ms1_features.remote(precursor_cuboid_row=row, converted_db_name=CONVERTED_DATABASE_NAME) for row in precursor_cuboids_df.itertuples()])

# join the list of dataframes into a single dataframe
features_df = pd.concat(features_l, axis=0, sort=False)

# write out all the features
print("writing {} features to {}".format(len(features_df), FEATURES_FILE))
features_df.to_pickle(FEATURES_FILE)
