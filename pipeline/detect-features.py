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

# find 3sigma for a specified m/z
def calculate_ms1_peak_delta(mz):
    instrument_resolution = 40000
    delta_m = mz / instrument_resolution  # FWHM of the peak
    sigma = delta_m / 2.35482  # std dev is FWHM / 2.35482. See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    ms1_peak_delta = 3 * sigma  # 99.7% of values fall within +/- 3 sigma
    return ms1_peak_delta
    
# calculate the sum of the raw points in the mono m/z
def calculate_cuboid_intensity_at_mz(centre_mz, raw_points):
    mz_delta = calculate_ms1_peak_delta(centre_mz)
    mz_lower = centre_mz - mz_delta
    mz_upper = centre_mz + mz_delta

    # extract the raw points for this peak
    mono_points_df = raw_points[(raw_points.mz >= mz_lower) & (raw_points.mz <= mz_upper)]
    mono_intensity = mono_points_df.intensity.sum()
    return (mono_intensity,mono_points_df)

# determine the mono peak apex and extent in CCS and RT
def determine_peak_characteristics(mono_mz, ms1_raw_points_df):
    # determine the raw that belong to the mono peak
    mono_intensity,mono_points_df = calculate_cuboid_intensity_at_mz(mono_mz, ms1_raw_points_df)
    if len(mono_points_df) > 0:
        # collapsing the monoisotopic's summed points onto the mobility dimension
        scan_df = mono_points_df.groupby(['scan'], as_index=False).intensity.sum()
        mobility_curve_fit = False
        try:
            guassian_params = peakutils.peak.gaussian_fit(scan_df.scan, scan_df.intensity, center_only=False)
            scan_apex = guassian_params[1]
            scan_side_width = min(2 * abs(guassian_params[2]), SCAN_BASE_PEAK_WIDTH)  # number of standard deviations either side of the apex
            scan_lower = scan_apex - scan_side_width
            scan_upper = scan_apex + scan_side_width
            if (scan_apex >= wide_scan_lower) and (scan_apex <= wide_scan_upper):
                mobility_curve_fit = True
        except:
            pass

        # if we couldn't fit a curve to the mobility dimension, take the intensity-weighted centroid
        if not mobility_curve_fit:
            scan_apex = mz_centroid(scan_df.intensity.to_numpy(), scan_df.scan.to_numpy())
            scan_lower = scan_apex - SCAN_BASE_PEAK_WIDTH
            scan_upper = scan_apex + SCAN_BASE_PEAK_WIDTH

        # In the RT dimension, look wider to find the apex
        rt_df = mono_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
        rt_curve_fit = False
        try:
            guassian_params = peakutils.peak.gaussian_fit(rt_df.retention_time_secs, rt_df.intensity, center_only=False)
            rt_apex = guassian_params[1]
            rt_side_width = min(3 * abs(guassian_params[2]), RT_BASE_PEAK_WIDTH_SECS)  # number of standard deviations either side of the apex
            rt_lower = rt_apex - rt_side_width
            rt_upper = rt_apex + rt_side_width
            if (rt_apex >= wide_rt_lower) and (rt_apex <= wide_rt_upper):
                rt_curve_fit = True
        except:
            pass

        # if we couldn't fit a curve to the RT dimension, take the intensity-weighted centroid
        if not rt_curve_fit:
            rt_apex = mz_centroid(rt_df.intensity.to_numpy(), rt_df.retention_time_secs.to_numpy())
            rt_lower = rt_apex - RT_BASE_PEAK_WIDTH_SECS
            rt_upper = rt_apex + RT_BASE_PEAK_WIDTH_SECS

        # package the result
        result_d = {}
        result_d['scan_apex'] = scan_apex
        result_d['scan_lower'] = scan_lower
        result_d['scan_upper'] = scan_upper
        result_d['rt_apex'] = rt_apex
        result_d['rt_lower'] = rt_lower
        result_d['rt_upper'] = rt_upper
    else:
        print('found no raw points where the mono peak should be: {}'.format(round(mono_mz,4)))
        result = None
    return result

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
    deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, use_quick_charge=True, averagine=averagine.peptide, charge_range=(1,5), error_tolerance=5.0, scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.95)

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
        feature_d['feature_intensity'] = row.intensity
        feature_d['envelope_mono_peak_intensity'] = row.envelope[0][1]
        feature_d['envelope_mono_peak_three_sigma_intensity'],_ = calculate_cuboid_intensity_at_mz(centre_mz=row.envelope[0][0], raw_points=ms1_points_df)
        feature_d['envelope'] = json.dumps([tuple(e) for e in row.envelope])
        feature_d['isotope_count'] = len(row.envelope)
        feature_d['deconvolution_score'] = row.score
        # from the precursor cuboid
        feature_d['precursor_id'] = precursor_cuboid_row.precursor_cuboid_id
        if args.precursor_definition_method == 'pasef':
            mono_mz = row.envelope[0][0]
            peak_d = determine_peak_characteristics(mono_mz, ms1_points_df)  # could also do this for 3did, but that method defines much tighter cuboids so it shouldn't be necessary
            if peak_d is not None:
                feature_d['scan_apex'] = peak_d['scan_apex']
                feature_d['scan_lower'] = peak_d['scan_lower']
                feature_d['scan_upper'] = peak_d['scan_upper']
                feature_d['rt_apex'] = peak_d['rt_apex']
                feature_d['rt_lower'] = peak_d['rt_lower']
                feature_d['rt_upper'] = peak_d['rt_upper']
        elif args.precursor_definition_method == '3did':
            feature_d['scan_apex'] = precursor_cuboid_row.anchor_point_scan
            feature_d['scan_lower'] = precursor_cuboid_row.scan_lower
            feature_d['scan_upper'] = precursor_cuboid_row.scan_upper
            feature_d['rt_apex'] = precursor_cuboid_row.anchor_point_retention_time_secs
            feature_d['rt_lower'] = precursor_cuboid_row.rt_lower
            feature_d['rt_upper'] = precursor_cuboid_row.rt_upper
        # assign a unique identifier to this feature
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

def extract_cuboid_attributes(precursor_cuboid_row):
    cuboid_d = {}
    if args.precursor_definition_method == 'pasef':
        cuboid_d['precursor_id'] = precursor_cuboid_row.precursor_cuboid_id
        cuboid_d['scan_apex'] = precursor_cuboid_row.anchor_point_scan
        cuboid_d['scan_lower'] = precursor_cuboid_row.scan_lower
        cuboid_d['scan_upper'] = precursor_cuboid_row.scan_upper
        cuboid_d['rt_apex'] = precursor_cuboid_row.anchor_point_retention_time_secs
        cuboid_d['rt_lower'] = precursor_cuboid_row.rt_lower
        cuboid_d['rt_upper'] = precursor_cuboid_row.rt_upper
    elif args.precursor_definition_method == '3did':
        cuboid_d['precursor_id'] = precursor_cuboid_row.precursor_cuboid_id
        cuboid_d['scan_apex'] = precursor_cuboid_row.anchor_point_scan
        cuboid_d['scan_lower'] = precursor_cuboid_row.scan_lower
        cuboid_d['scan_upper'] = precursor_cuboid_row.scan_upper
        cuboid_d['rt_apex'] = precursor_cuboid_row.anchor_point_retention_time_secs
        cuboid_d['rt_lower'] = precursor_cuboid_row.rt_lower
        cuboid_d['rt_upper'] = precursor_cuboid_row.rt_upper
    return cuboid_d

###################################
parser = argparse.ArgumentParser(description='Detect the features in a run\'s precursor cuboids.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-pdm','--precursor_definition_method', type=str, choices=['pasef','3did'], help='The method used to define the precursor cuboids.', required=True)
parser.add_argument('-ml','--mz_lower', type=int, default='100', help='Lower limit for m/z.', required=False)
parser.add_argument('-mu','--mz_upper', type=int, default='1700', help='Upper limit for m/z.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
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

# input cuboids
CUBOIDS_DIR = "{}/precursor-cuboids-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
CUBOIDS_FILE = '{}/exp-{}-run-{}-mz-{}-{}-rt-{}-{}-precursor-cuboids-{}.pkl'.format(CUBOIDS_DIR, args.experiment_name, args.run_name, round(args.mz_lower), round(args.mz_upper), round(args.rt_lower), round(args.rt_upper), args.precursor_definition_method)

# output features
FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
FEATURES_FILE = '{}/exp-{}-run-{}-mz-{}-{}-rt-{}-{}-features-{}.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name, round(args.mz_lower), round(args.mz_upper), round(args.rt_lower), round(args.rt_upper), args.precursor_definition_method)

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
