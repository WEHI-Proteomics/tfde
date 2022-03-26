import pandas as pd
import numpy as np
import sys
import pickle
import peakutils
from scipy import signal
import random
import argparse
import os
import time
import math
import multiprocessing as mp
import json
import alphatims.bruker
import sqlite3
import configparser
from configparser import ExtendedInterpolation


class FixedDict(object):
    def __init__(self, dictionary):
        self._dictionary = dictionary
    def __setitem__(self, key, item):
            if key not in self._dictionary:
                raise KeyError("The key {} is not defined.".format(key))
            self._dictionary[key] = item
    def __getitem__(self, key):
        return self._dictionary[key]
    def get_dict(self):
        return self._dictionary

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# load the ms1 frame ids
def load_ms1_frame_ids(raw_db_name):
    db_conn = sqlite3.connect('{}/analysis.tdf'.format(raw_db_name))
    ms1_frame_properties_df = pd.read_sql_query("select Id,Time from Frames where MsMsType == {} order by Time".format(FRAME_TYPE_MS1), db_conn)
    db_conn.close()
    return ms1_frame_properties_df

# get the ms1 frame ids with a range of RT as a tuple
def get_ms1_frame_ids(rt_lower, rt_upper):
    df = ms1_frame_properties_df[(ms1_frame_properties_df.Time >= rt_lower) & (ms1_frame_properties_df.Time <= rt_upper)]
    ms1_frame_ids = tuple(df.Id)
    return ms1_frame_ids

def calculate_monoisotopic_mass_from_mz(monoisotopic_mz, charge):
    monoisotopic_mass = (monoisotopic_mz * charge) - (PROTON_MASS * charge)
    return monoisotopic_mass

# takes a numpy array of intensity, and another of mz
def mz_centroid(_int_f, _mz_f):
    try:
        return ((_int_f/_int_f.sum()) * _mz_f).sum()
    except:
        print("exception in mz_centroid")
        return None

# calculate the r-squared value of series_2 against series_1, where series_1 is the original data (source: https://stackoverflow.com/a/37899817/1184799)
def calculate_r_squared(series_1, series_2):
    residuals = series_1 - series_2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((series_1 - np.mean(series_1))**2)
    if ss_tot != 0:
        r_squared = 1 - (ss_res / ss_tot)
    else:
        r_squared = None
    return r_squared

def estimate_target_coordinates(row_as_series, mz_estimator, scan_estimator, rt_estimator):
    sequence_estimation_attribs_s = row_as_series[['theoretical_mz','experiment_rt_mean','experiment_rt_std_dev','experiment_scan_mean','experiment_scan_std_dev','experiment_intensity_mean','experiment_intensity_std_dev']]
    sequence_estimation_attribs = np.reshape(sequence_estimation_attribs_s.values, (1, -1))  # make it 2D

    # estimate the raw monoisotopic m/z
    mz_delta_ppm_estimated = mz_estimator.predict(sequence_estimation_attribs)[0]
    theoretical_mz = sequence_estimation_attribs_s.theoretical_mz
    estimated_monoisotopic_mz = (mz_delta_ppm_estimated / 1e6 * theoretical_mz) + theoretical_mz

    # estimate the raw monoisotopic scan
    estimated_scan_delta = scan_estimator.predict(sequence_estimation_attribs)[0]
    experiment_scan_mean = sequence_estimation_attribs_s.experiment_scan_mean
    estimated_scan_apex = (estimated_scan_delta * experiment_scan_mean) + experiment_scan_mean

    # estimate the raw monoisotopic RT
    estimated_rt_delta = rt_estimator.predict(sequence_estimation_attribs)[0]
    experiment_rt_mean = sequence_estimation_attribs_s.experiment_rt_mean
    estimated_rt_apex = (estimated_rt_delta * experiment_rt_mean) + experiment_rt_mean

    return {"mono_mz":estimated_monoisotopic_mz, "scan_apex":estimated_scan_apex, "rt_apex":estimated_rt_apex}

def get_decoy_coordinates(target_mz, target_scan, peak_width_scan, target_rt, peak_width_rt):
    # calculate decoy mz
    mz_base_offset_ppm = (1 if random.random() < 0.5 else -1) * 100  # +/- offset of 100 ppm
    mz_random_delta_ppm = random.randint(-20, +20)  # random delta ppm between -20 and +20
    mz_offset_ppm = mz_base_offset_ppm + mz_random_delta_ppm
    decoy_mz = (mz_offset_ppm / 1e6 * target_mz) + target_mz
    # calculate decoy scan
    scan_base_offset = (1 if random.random() < 0.5 else -1) * 2 * peak_width_scan  # +/- 2 peak widths
    scan_random_delta = random.randint(-10, +10)
    scan_offset = scan_base_offset + scan_random_delta
    decoy_scan = target_scan + scan_offset
    # calculate decoy RT
    rt_base_offset = (1 if random.random() < 0.5 else -1) * 2 * peak_width_rt  # +/- 2 peak widths
    rt_random_delta = random.randint(-10, +10)
    rt_offset = rt_base_offset + rt_random_delta
    decoy_rt = target_rt + rt_offset
    return (decoy_mz, decoy_scan, decoy_rt)

def calculate_decoy_coordinates(row_as_series):
    peak_width_scan = row_as_series.experiment_scan_peak_width
    peak_width_rt = row_as_series.experiment_rt_peak_width
    estimated_monoisotopic_mz = row_as_series.target_coords['mono_mz']
    estimated_scan_apex = row_as_series.target_coords['scan_apex']
    estimated_rt_apex = row_as_series.target_coords['rt_apex']

    (decoy_mz, decoy_scan, decoy_rt) = get_decoy_coordinates(estimated_monoisotopic_mz, estimated_scan_apex, peak_width_scan, estimated_rt_apex, peak_width_rt)
    return {"mono_mz":decoy_mz, "scan_apex":decoy_scan, "rt_apex":decoy_rt}

# Find the ratio of H(peak_number)/H(peak_number-1) for peak_number=1..6
# peak_number = 0 refers to the monoisotopic peak
# number_of_sulphur = number of sulphur atoms in the molecule
#
# source: Valkenborg et al, "A Model-Based Method for the Prediction of the Isotopic Distribution of Peptides", https://core.ac.uk/download/pdf/82021511.pdf
def peak_ratio(monoisotopic_mass, peak_number, number_of_sulphur):
    MAX_NUMBER_OF_SULPHUR_ATOMS = 3
    MAX_NUMBER_OF_PREDICTED_RATIOS = 6

    S0_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=np.ndarray)
    S0_r[1] = np.array([-0.00142320578040, 0.53158267080224, 0.00572776591574, -0.00040226083326, -0.00007968737684])
    S0_r[2] = np.array([0.06258138406507, 0.24252967352808, 0.01729736525102, -0.00427641490976, 0.00038011211412])
    S0_r[3] = np.array([0.03092092306220, 0.22353930450345, -0.02630395501009, 0.00728183023772, -0.00073155573939])
    S0_r[4] = np.array([-0.02490747037406, 0.26363266501679, -0.07330346656184, 0.01876886839392, -0.00176688757979])
    S0_r[5] = np.array([-0.19423148776489, 0.45952477474223, -0.18163820209523, 0.04173579115885, -0.00355426505742])
    S0_r[6] = np.array([0.04574408690798, -0.05092121193598, 0.13874539944789, -0.04344815868749, 0.00449747222180])

    S1_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=np.ndarray)
    S1_r[1] = np.array([-0.01040584267474, 0.53121149663696, 0.00576913817747, -0.00039325152252, -0.00007954180489])
    S1_r[2] = np.array([0.37339166598255, -0.15814640001919, 0.24085046064819, -0.06068695741919, 0.00563606634601])
    S1_r[3] = np.array([0.06969331604484, 0.28154425636993, -0.08121643989151, 0.02372741957255, -0.00238998426027])
    S1_r[4] = np.array([0.04462649178239, 0.23204790123388, -0.06083969521863, 0.01564282892512, -0.00145145206815])
    S1_r[5] = np.array([-0.20727547407753, 0.53536509500863, -0.22521649838170, 0.05180965157326, -0.00439750995163])
    S1_r[6] = np.array([0.27169670700251, -0.37192045082925, 0.31939855191976, -0.08668833166842, 0.00822975581940])

    S2_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=np.ndarray)
    S2_r[1] = np.array([-0.01937823810470, 0.53084210514216, 0.00580573751882, -0.00038281138203, -0.00007958217070])
    S2_r[2] = np.array([0.68496829280011, -0.54558176102022, 0.44926662609767, -0.11154849560657, 0.01023294598884])
    S2_r[3] = np.array([0.04215807391059, 0.40434195078925, -0.15884974959493, 0.04319968814535, -0.00413693825139])
    S2_r[4] = np.array([0.14015578207913, 0.14407679007180, -0.01310480312503, 0.00362292256563, -0.00034189078786])
    S2_r[5] = np.array([-0.02549241716294, 0.32153542852101, -0.11409513283836, 0.02617210469576, -0.00221816103608])
    S2_r[6] = np.array([-0.14490868030324, 0.33629928307361, -0.08223564735018, 0.01023410734015, -0.00027717589598])

    model_params = np.empty(MAX_NUMBER_OF_SULPHUR_ATOMS, dtype=np.ndarray)
    model_params[0] = S0_r
    model_params[1] = S1_r
    model_params[2] = S2_r

    ratio = None
    if (((1 <= peak_number <= 3) & (((number_of_sulphur == 0) & (498 <= monoisotopic_mass <= 3915)) |
                                    ((number_of_sulphur == 1) & (530 <= monoisotopic_mass <= 3947)) |
                                    ((number_of_sulphur == 2) & (562 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 4) & (((number_of_sulphur == 0) & (907 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (939 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (971 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 5) & (((number_of_sulphur == 0) & (1219 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (1251 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (1283 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 6) & (((number_of_sulphur == 0) & (1559 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (1591 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (1623 <= monoisotopic_mass <= 3978))))):
        beta0 = model_params[number_of_sulphur][peak_number][0]
        beta1 = model_params[number_of_sulphur][peak_number][1]
        beta2 = model_params[number_of_sulphur][peak_number][2]
        beta3 = model_params[number_of_sulphur][peak_number][3]
        beta4 = model_params[number_of_sulphur][peak_number][4]
        scaled_m = monoisotopic_mass / 1000.0
        ratio = beta0 + (beta1*scaled_m) + beta2*(scaled_m**2) + beta3*(scaled_m**3) + beta4*(scaled_m**4)
    return ratio

# assumes the isotope's raw points have already been flattened to a particular dimension (e.g. scan, RT, m/z) and
# sorted by ascending order in that dimension
def fit_curve_to_flattened_isotope(flattened_points_df, estimated_apex, estimated_peak_width, maximum_number_of_peaks, isotope_dimension, isotope_number, sequence, charge, run_name):
    peaks_l = []
    filtered_points_d = None
    if len(flattened_points_df) > 0:
        # apply a filter to make curve fitting easier, if there are enough points
        flattened_points_df['filtered_intensity'] = flattened_points_df.intensity  # set the default
        window_length = 11
        if len(flattened_points_df) > window_length:
            try:
                flattened_points_df['filtered_intensity'] = signal.savgol_filter(flattened_points_df.intensity, window_length=window_length, polyorder=3)
                filtered = True
            except:
                # print("Filter failed for the flattened isotope {}, dimension {}, sequence {}, charge {}, run {}".format(isotope_number, isotope_dimension, sequence, charge, run_name))
                filtered = False
        else:
            # print("Not enough points to filter in the flattened isotope {}, dimension {}, sequence {}, charge {}, run {}".format(isotope_number, isotope_dimension, sequence, charge, run_name))
            filtered = False
        if filtered:
            filtered_points_d = flattened_points_df[['x','filtered_intensity']].to_dict('records')
        else:
            filtered_points_d = None

        # find the peak(s)
        # the minimum distance between peaks gives the minimum mount of feature overlap we will tolerate
        peak_x_l = []
        try:
            peak_idxs = peakutils.indexes(flattened_points_df.filtered_intensity.values, thres=0.05, min_dist=estimated_peak_width/2, thres_abs=False)
            peak_x_l = flattened_points_df.iloc[peak_idxs].x.to_list()
        except:
            pass
        if len(peak_x_l) == 0:
            # get the maximum intensity point
            peak_x_l = [flattened_points_df.loc[flattened_points_df.filtered_intensity.idxmax()].x]
            # print('could not find any peaks - taking the maximum point - peak_x_l: {}'.format(peak_x_l))
        peaks_df = flattened_points_df[flattened_points_df.x.isin(peak_x_l)]

        # peaks_df should now contain the rows from flattened_points_df that represent the peaks

        # find the valleys
        # the minimum distance between valleys gives the minimum peak width
        valley_x_l = []
        try:
            valley_idxs = peakutils.indexes(-flattened_points_df.filtered_intensity.values, thres=0.05, min_dist=estimated_peak_width/8, thres_abs=False)
            valley_x_l = flattened_points_df.iloc[valley_idxs].x.to_list()
        except:
            pass
        if len(valley_x_l) == 0:
            # get the minimum and maximum x
            # print('could not find any valleys - taking the widest points')
            valley_x_l = [flattened_points_df.x.min(), flattened_points_df.x.max()]
        valleys_df = flattened_points_df[flattened_points_df.x.isin(valley_x_l)]

        # valleys_df should now contain the rows from flattened_points_df that represent the valleys

        # print('sequence {}, charge {}, number of points {}\npoints {}\npeaks {}\nvalleys {}\n'.format(sequence, charge, len(flattened_points_df), flattened_points_df, peaks_df, valleys_df))

        # isolate each peak and extract its attributes
        for peak_idx,peak in enumerate(peaks_df.itertuples()):
            # find the x bounds
            upper_x = valleys_df[valleys_df.x > peak.x].x.min()
            if math.isnan(upper_x):
                upper_x = flattened_points_df.x.max()
            lower_x = valleys_df[valleys_df.x < peak.x].x.max()
            if math.isnan(lower_x):
                lower_x = flattened_points_df.x.min()
            peak_points_df = flattened_points_df[(flattened_points_df.x >= lower_x) & (flattened_points_df.x <= upper_x)]
            peak_points_left_df = peak_points_df[peak_points_df.x <= peak.x]
            peak_points_right_df = peak_points_df[peak_points_df.x > peak.x]
            # find the standard deviation and FWHM, assuming a gaussian distribution
            std_dev = np.mean([abs(peak.x-upper_x), abs(peak.x-lower_x)]) / 3  # each side is three std devs, so we take the mean of the two
            fwhm = 2.355 * std_dev
            base_width = upper_x - lower_x
            # calculate the area under the curve by summing the intensities
            area_under_curve = peak_points_df.intensity.sum()
            # calculate the r-squared value of the fitted curve
            if filtered:
                r_squared = calculate_r_squared(series_1=peak_points_df.intensity, series_2=peak_points_df.filtered_intensity)
            else:
                r_squared = None
            # calculate the peak symmetry
            lhs_auc = peak_points_left_df.intensity.sum()
            rhs_auc = peak_points_right_df.intensity.sum()
            if rhs_auc == 0:
                symmetry = 0
            else:
                symmetry = lhs_auc / rhs_auc
            # assemble all the peak attributes
            peak_attributes = (peak.x, peak.intensity, lower_x, upper_x, std_dev, fwhm, base_width, area_under_curve, r_squared, symmetry)
            peaks_l.append(peak_attributes)
    else:
        if args.small_set_mode:
            print("No points were found in the flattened isotope {}, dimension {}, sequence {}, charge {}, run {}".format(isotope_number, isotope_dimension, sequence, charge, run_name))

    # form a dataframe to process the peaks we detected
    isolated_peaks_df = pd.DataFrame(peaks_l, columns=['apex_x','intensity','lower_x','upper_x','std_dev','full_width_half_max','base_width','area_under_curve','r_squared','peak_symmetry'])
    isolated_peaks_found = len(isolated_peaks_df)

    # sort the detected peaks by their proximity to the estimated apex, and return the maximum_number_of_peaks
    isolated_peaks_df['apex_x_delta'] = abs(isolated_peaks_df.apex_x - estimated_apex)
    isolated_peaks_df.sort_values(by=['apex_x_delta'], ascending=True, inplace=True)
    isolated_peaks_df.reset_index(drop=True, inplace=True)
    isolated_peaks_df = isolated_peaks_df.loc[:maximum_number_of_peaks-1]
    isolated_peaks_returned = len(isolated_peaks_df)

    # print('fit_curve_to_flattened_isotope: found {} isolated peaks; returning {}, sequence {}, charge {}, run {}'.format(isolated_peaks_found, isolated_peaks_returned, sequence, charge, run_name))

    # collate the results
    results_d = {}
    if len(isolated_peaks_df) > 0:
        results_d['peaks'] = isolated_peaks_df.to_dict('records')
    else:
        results_d['peaks'] = None
    results_d['filtered_points'] = filtered_points_d
    return results_d

# this function assumes the x-axis is a column labelled 'x', and the y-axis is a column labelled 'intensity'
def calculate_isotope_correlation(isotope_0_df, isotope_1_df, isotope_2_df):
    # scale the x axis so we can join them
    isotope_0_df['x_scaled'] = (isotope_0_df.x * 100).astype(int)
    isotope_1_df['x_scaled'] = (isotope_1_df.x * 100).astype(int)
    isotope_2_df['x_scaled'] = (isotope_2_df.x * 100).astype(int)

    # combine the isotopes by aligning the x-dimension points they have in common
    combined_df = pd.merge(isotope_0_df, isotope_1_df, on='x_scaled', how='inner', suffixes=('_0', '_1')).sort_values(by='x_scaled')
    combined_df = pd.merge(combined_df, isotope_2_df, on='x_scaled', how='inner', suffixes=('_0', '_2')).sort_values(by='x_scaled')
    combined_df.rename(columns={'intensity': 'intensity_2'}, inplace=True)
    combined_df = combined_df[['x_scaled','intensity_0','intensity_1','intensity_2']]

    # calculate the correlation coefficient
    if len(combined_df) >=3:  # let's have at least three points in common between the isotopes to get a meaningful correlation
        correlation_coefficient_a = np.corrcoef(combined_df[['intensity_0','intensity_1','intensity_2']].values, rowvar=False)
        if isinstance(correlation_coefficient_a, np.ndarray):
            isotope_correlation = correlation_coefficient_a[1,0]
        else:
            isotope_correlation = correlation_coefficient_a
            print("np.corrcoef return a scalar {} for isotopes {}".format(isotope_correlation, combined_df))
    else:
        isotope_correlation = 0
    return isotope_correlation

# for a given set of isotopes, calculate the feature metrics
def calculate_feature_metrics(isotope_peaks_df, isotope_raw_points_l, estimated_mono_mz, estimated_scan_apex, estimated_rt_apex, rt_metrics_l, scan_metrics_l, expected_spacing_mz, charge):
    # ensure we have a full set of metrics, even if some are None
    metrics_names = \
        ['delta_mz_ppm',
         'delta_rt',
         'delta_scan',
         'fwhm_rt_0',
         'fwhm_scan_0',
         'geometric_mean_0_1',
         'geometric_mean_0_1_2',
         'isotope_0_1_mz_delta_ppm',
         'isotope_0_1_rt_delta',
         'isotope_0_1_scan_delta',
         'isotope_0_2_mz_delta_ppm',
         'isotope_0_2_rt_delta',
         'isotope_0_2_scan_delta',
         'monoisotope_auc_over_isotope_peak_auc_sum',
         'monoisotope_int_over_isotope_peak_int_sum',
         'mz_delta_ppm_std_dev_0',
         'mz_delta_ppm_std_dev_1',
         'number_of_frames_0',
         'number_of_frames_1',
         'number_of_frames_2',
         'number_of_missing_frames_0',
         'number_of_missing_frames_1',
         'number_of_missing_frames_2',
         'peak_base_width_rt_0',
         'peak_base_width_scan_0',
         'r_squared_phr',
         'rt_isotope_correlation',
         'rt_isotope_cv',
         'rt_peak_symmetry_0',
         'rt_peak_symmetry_1',
         'rt_peak_symmetry_2',
         'scan_isotope_correlation',
         'scan_isotope_cv',
         'scan_peak_symmetry_0',
         'scan_peak_symmetry_1',
         'scan_peak_symmetry_2']
    d = {}
    for n in metrics_names:
        d[n] = None
    feature_metrics = FixedDict(d)

    # Calculate the feature metrics
    calculated_monoisotopic_mz = isotope_peaks_df.iloc[0].mz_centroid
    if calculated_monoisotopic_mz is not None:
        feature_metrics['delta_mz_ppm'] = (calculated_monoisotopic_mz - estimated_mono_mz) / estimated_mono_mz * 1e6
    else:
        feature_metrics['delta_mz_ppm'] = None
    if (scan_metrics_l[0] is not None) and (scan_metrics_l[0]['apex_x'] is not None) and (estimated_scan_apex is not None):
        feature_metrics['delta_scan'] = (scan_metrics_l[0]['apex_x'] - estimated_scan_apex) / estimated_scan_apex
    else:
        feature_metrics['delta_scan'] = None
    if (rt_metrics_l[0] is not None) and (rt_metrics_l[0]['apex_x'] is not None) and (estimated_rt_apex is not None):
        feature_metrics['delta_rt'] = (rt_metrics_l[0]['apex_x'] - estimated_rt_apex) / estimated_rt_apex
    else:
        feature_metrics['delta_rt'] = None

    # Calculate the delta ppm of the de-isotoped first and second isotopic peaks
    monoisotopic_mz_centroid = isotope_peaks_df.iloc[0].mz_centroid  # monoisotopic
    isotope_1_mz_centroid = isotope_peaks_df.iloc[1].mz_centroid  # first isotope
    isotope_2_mz_centroid = isotope_peaks_df.iloc[2].mz_centroid  # second isotope

    # get the raw points for each isotope
    mono_raw_points_df = isotope_raw_points_l[0].copy()
    isotope_1_raw_points_df = isotope_raw_points_l[1].copy()
    isotope_2_raw_points_df = isotope_raw_points_l[2].copy()

    # delta from where isotope 1 is detected and where it's predicted
    if (monoisotopic_mz_centroid is not None) and (isotope_1_mz_centroid is not None):
        isotope_0_1_mz_delta_ppm = (monoisotopic_mz_centroid - (isotope_1_mz_centroid - (1 * expected_spacing_mz))) / monoisotopic_mz_centroid * 1e6
        feature_metrics['isotope_0_1_mz_delta_ppm'] = isotope_0_1_mz_delta_ppm
    else:
        feature_metrics['isotope_0_1_mz_delta_ppm'] = None

    # delta from where isotope 2 is detected and where it's predicted
    if (monoisotopic_mz_centroid is not None) and (isotope_2_mz_centroid is not None):
        isotope_0_2_mz_delta_ppm = (monoisotopic_mz_centroid - (isotope_2_mz_centroid - (2 * expected_spacing_mz))) / monoisotopic_mz_centroid * 1e6
        feature_metrics['isotope_0_2_mz_delta_ppm'] = isotope_0_2_mz_delta_ppm
    else:
        feature_metrics['isotope_0_2_mz_delta_ppm'] = None

    # calculate the RT apex deltas from the monoisotopic peak for the first and second isotopes
    if (rt_metrics_l[1] is not None) and (rt_metrics_l[0] is not None) and (rt_metrics_l[1]['apex_x'] is not None) and (rt_metrics_l[0]['apex_x'] is not None):
        isotope_0_1_rt_delta = (rt_metrics_l[1]['apex_x'] - rt_metrics_l[0]['apex_x']) / rt_metrics_l[0]['apex_x']
        feature_metrics['isotope_0_1_rt_delta'] = isotope_0_1_rt_delta
    else:
        feature_metrics['isotope_0_1_rt_delta'] = None

    if (rt_metrics_l[2] is not None) and (rt_metrics_l[0] is not None) and (rt_metrics_l[2]['apex_x'] is not None) and (rt_metrics_l[0]['apex_x'] is not None):
        isotope_0_2_rt_delta = (rt_metrics_l[2]['apex_x'] - rt_metrics_l[0]['apex_x']) / rt_metrics_l[0]['apex_x']
        feature_metrics['isotope_0_2_rt_delta'] = isotope_0_2_rt_delta
    else:
        feature_metrics['isotope_0_2_rt_delta'] = None

    # calculate the scan apex deltas from the monoisotopic peak for the first and second isotopes
    if (scan_metrics_l[1] is not None) and (scan_metrics_l[0] is not None) and (scan_metrics_l[1]['apex_x'] is not None) and (scan_metrics_l[0]['apex_x'] is not None):
        isotope_0_1_scan_delta = (scan_metrics_l[1]['apex_x'] - scan_metrics_l[0]['apex_x']) / scan_metrics_l[0]['apex_x']
        feature_metrics['isotope_0_1_scan_delta'] = isotope_0_1_scan_delta
    else:
        feature_metrics['isotope_0_1_scan_delta'] = None

    if (scan_metrics_l[2] is not None) and (scan_metrics_l[0] is not None) and (scan_metrics_l[2]['apex_x'] is not None) and (scan_metrics_l[0]['apex_x'] is not None):
        isotope_0_2_scan_delta = (scan_metrics_l[2]['apex_x'] - scan_metrics_l[0]['apex_x']) / scan_metrics_l[0]['apex_x']
        feature_metrics['isotope_0_2_scan_delta'] = isotope_0_2_scan_delta
    else:
        feature_metrics['isotope_0_2_scan_delta'] = None

    # calculate the monoisotopic peak intensity divided by the sum of the isotope peaks
    if (isotope_peaks_df.iloc[:3].summed_intensity.sum() != 0):
        monoisotope_int_over_isotope_peak_int_sum = isotope_peaks_df.iloc[0].summed_intensity / isotope_peaks_df.iloc[:3].summed_intensity.sum()
        feature_metrics['monoisotope_int_over_isotope_peak_int_sum'] = monoisotope_int_over_isotope_peak_int_sum
    else:
        feature_metrics['monoisotope_int_over_isotope_peak_int_sum'] = None

    # calculate the monoisotopic peak AUC divided by the sum of the isotope peak AUCs
    if (rt_metrics_l[0] is not None) and (rt_metrics_l[1] is not None) and (rt_metrics_l[2] is not None) and (rt_metrics_l[0]['area_under_curve'] is not None) and (rt_metrics_l[1]['area_under_curve'] is not None) and (rt_metrics_l[2]['area_under_curve'] is not None):
        monoisotope_auc_over_isotope_peak_auc_sum = rt_metrics_l[0]['area_under_curve'] / (rt_metrics_l[0]['area_under_curve'] + rt_metrics_l[1]['area_under_curve'] + rt_metrics_l[2]['area_under_curve'])
        feature_metrics['monoisotope_auc_over_isotope_peak_auc_sum'] = monoisotope_auc_over_isotope_peak_auc_sum
    else:
        feature_metrics['monoisotope_auc_over_isotope_peak_auc_sum'] = None

    # calculate the theoretical and observed isotopic peak height ratios
    monoisotopic_mass = calculate_monoisotopic_mass_from_mz(monoisotopic_mz_centroid, charge)
    ratios = []
    for isotope in [1,2]:  # ratio of isotopes 1:0, 2:1
        expected_ratio = peak_ratio(monoisotopic_mass=monoisotopic_mass, peak_number=isotope, number_of_sulphur=0)
        observed_ratio = isotope_peaks_df.iloc[isotope].summed_intensity / isotope_peaks_df.iloc[isotope-1].summed_intensity
        ratios.append((expected_ratio, observed_ratio))

    ratios_a = np.array(ratios)
    if ((None not in ratios_a[:,0]) and (None not in ratios_a[:,1])):
        r_squared_phr = calculate_r_squared(ratios_a[:,0], ratios_a[:,1])
    else:
        r_squared_phr = None
    feature_metrics['r_squared_phr'] = r_squared_phr

    # calculate the geometric mean of the isotope peak intensities
    if (isotope_peaks_df.iloc[0].summed_intensity > 0) and (isotope_peaks_df.iloc[1].summed_intensity > 0):
        geometric_mean_0_1 = np.log(isotope_peaks_df.iloc[0].summed_intensity * isotope_peaks_df.iloc[1].summed_intensity) / 2
        feature_metrics['geometric_mean_0_1'] = geometric_mean_0_1
    else:
        feature_metrics['geometric_mean_0_1'] = None
    if (isotope_peaks_df.iloc[0].summed_intensity > 0) and (isotope_peaks_df.iloc[1].summed_intensity > 0) and (isotope_peaks_df.iloc[2].summed_intensity > 0):
        geometric_mean_0_1_2 = np.log(isotope_peaks_df.iloc[0].summed_intensity * isotope_peaks_df.iloc[1].summed_intensity * isotope_peaks_df.iloc[2].summed_intensity) / 3
        feature_metrics['geometric_mean_0_1_2'] = geometric_mean_0_1_2
    else:
        feature_metrics['geometric_mean_0_1_2'] = None

    # calculate the m/z ppm standard deviation for isotopes 0 and 1
    mz_centroid_0 = isotope_peaks_df.iloc[0].mz_centroid
    mono_raw_points_df['mz_ppm_delta'] = (mono_raw_points_df.mz - mz_centroid_0) / mz_centroid_0 * 1e6
    mz_delta_ppm_std_dev_0 = np.std(mono_raw_points_df.mz_ppm_delta)
    feature_metrics['mz_delta_ppm_std_dev_0'] = mz_delta_ppm_std_dev_0

    mz_centroid_1 = isotope_peaks_df.iloc[1].mz_centroid
    isotope_1_raw_points_df['mz_ppm_delta'] = (isotope_1_raw_points_df.mz - mz_centroid_1) / mz_centroid_1 * 1e6
    mz_delta_ppm_std_dev_1 = np.std(isotope_1_raw_points_df.mz_ppm_delta)
    feature_metrics['mz_delta_ppm_std_dev_1'] = mz_delta_ppm_std_dev_1

    # calculate the symmetry of the isotopes in RT and CCS
    if (rt_metrics_l[0] is not None):
        feature_metrics['rt_peak_symmetry_0'] = rt_metrics_l[0]['peak_symmetry']
    if (rt_metrics_l[1] is not None):
        feature_metrics['rt_peak_symmetry_1'] = rt_metrics_l[1]['peak_symmetry']
    if (rt_metrics_l[2] is not None):
        feature_metrics['rt_peak_symmetry_2'] = rt_metrics_l[2]['peak_symmetry']

    if (scan_metrics_l[0] is not None):
        feature_metrics['scan_peak_symmetry_0'] = scan_metrics_l[0]['peak_symmetry']
    if (scan_metrics_l[1] is not None):
        feature_metrics['scan_peak_symmetry_1'] = scan_metrics_l[1]['peak_symmetry']
    if (scan_metrics_l[2] is not None):
        feature_metrics['scan_peak_symmetry_2'] = scan_metrics_l[2]['peak_symmetry']

    # calculate the isotopic peak correlation with each other in RT and CCS
    if ((mono_raw_points_df is not None) and (len(mono_raw_points_df) > 0) and (isotope_1_raw_points_df is not None) and (len(isotope_1_raw_points_df) > 0) and (isotope_2_raw_points_df is not None) and (len(isotope_2_raw_points_df) > 0)):
        # correlation in RT
        rt_0_df = mono_raw_points_df.groupby(['frame_id'], as_index=False).intensity.sum()
        rt_1_df = isotope_1_raw_points_df.groupby(['frame_id'], as_index=False).intensity.sum()
        rt_2_df = isotope_2_raw_points_df.groupby(['frame_id'], as_index=False).intensity.sum()

        rt_0_df['x'] = rt_0_df.frame_id
        rt_1_df['x'] = rt_1_df.frame_id
        rt_2_df['x'] = rt_2_df.frame_id

        feature_metrics['rt_isotope_correlation'] = calculate_isotope_correlation(rt_0_df, rt_1_df, rt_2_df)

        # correlation in CCS
        scan_0_df = mono_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()
        scan_1_df = isotope_1_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()
        scan_2_df = isotope_2_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()

        scan_0_df['x'] = scan_0_df.scan
        scan_1_df['x'] = scan_1_df.scan
        scan_2_df['x'] = scan_2_df.scan

        feature_metrics['scan_isotope_correlation'] = calculate_isotope_correlation(scan_0_df, scan_1_df, scan_2_df)
    else:
        feature_metrics['rt_isotope_correlation'] = None
        feature_metrics['scan_isotope_correlation'] = None

    # calculate the CV for isotope apexes in RT
    if (rt_metrics_l[0] is not None) and (rt_metrics_l[1] is not None) and (rt_metrics_l[2] is not None) and (rt_metrics_l[0]['apex_x'] is not None) and (rt_metrics_l[1]['apex_x'] is not None) and (rt_metrics_l[2]['apex_x'] is not None) and (np.mean([rt_metrics_l[0]['apex_x'], rt_metrics_l[1]['apex_x'], rt_metrics_l[2]['apex_x']]) != 0):
        n = [rt_metrics_l[0]['apex_x'], rt_metrics_l[1]['apex_x'], rt_metrics_l[2]['apex_x']]
        feature_metrics['rt_isotope_cv'] = np.std(n) / np.mean(n)
    else:
        feature_metrics['rt_isotope_cv'] = None

    # calculate the CV for isotope apexes in CCS
    if (scan_metrics_l[0] is not None) and (scan_metrics_l[1] is not None) and (scan_metrics_l[2] is not None) and (scan_metrics_l[0]['apex_x'] is not None) and (scan_metrics_l[1]['apex_x'] is not None) and (scan_metrics_l[2]['apex_x'] is not None) and (np.mean([scan_metrics_l[0]['apex_x'], scan_metrics_l[1]['apex_x'], scan_metrics_l[2]['apex_x']]) != 0):
        n = [scan_metrics_l[0]['apex_x'], scan_metrics_l[1]['apex_x'], scan_metrics_l[2]['apex_x']]
        feature_metrics['scan_isotope_cv'] = np.std(n) / np.mean(n)
    else:
        feature_metrics['scan_isotope_cv'] = None

    # calculate the FWHM and peak base width of the monoisotopic peak in RT and CCS dimensions
    if (rt_metrics_l[0] is not None):
        feature_metrics['fwhm_rt_0'] = rt_metrics_l[0]['full_width_half_max']
        feature_metrics['peak_base_width_rt_0'] = rt_metrics_l[0]['base_width']
    else:
        feature_metrics['fwhm_rt_0'] = None
        feature_metrics['peak_base_width_rt_0'] = None

    if (scan_metrics_l[0] is not None):
        feature_metrics['fwhm_scan_0'] = scan_metrics_l[0]['full_width_half_max']
        feature_metrics['peak_base_width_scan_0'] = scan_metrics_l[0]['base_width']
    else:
        feature_metrics['fwhm_scan_0'] = None
        feature_metrics['peak_base_width_scan_0'] = None

    # calculate the number of points and missing points in consecutive frames
    if (rt_metrics_l[0] is not None):
        rt_lower_0 = rt_metrics_l[0]['lower_x']
        rt_upper_0 = rt_metrics_l[0]['upper_x']
        ms1_frame_ids_0 = get_ms1_frame_ids(rt_lower_0, rt_upper_0)
        ms1_frame_ids_0_df = pd.DataFrame(ms1_frame_ids_0, columns=['frame_id'])
        ms1_frame_ids_0_df['intensity'] = 0
        merged_df = pd.merge(ms1_frame_ids_0_df, isotope_raw_points_l[0], on='frame_id', how='left', suffixes=('_0', '_1')).sort_values(by='frame_id')
        number_of_missing_frames_0 = merged_df.intensity_1.isna().sum()
        feature_metrics['number_of_missing_frames_0'] = number_of_missing_frames_0
        feature_metrics['number_of_frames_0'] = len(ms1_frame_ids_0_df)
    else:
        feature_metrics['number_of_missing_frames_0'] = None
        feature_metrics['number_of_frames_0'] = None

    if (rt_metrics_l[1] is not None):
        rt_lower_1 = rt_metrics_l[1]['lower_x']
        rt_upper_1 = rt_metrics_l[1]['upper_x']
        ms1_frame_ids_1 = get_ms1_frame_ids(rt_lower_1, rt_upper_1)
        ms1_frame_ids_1_df = pd.DataFrame(ms1_frame_ids_1, columns=['frame_id'])
        ms1_frame_ids_1_df['intensity'] = 0
        merged_df = pd.merge(ms1_frame_ids_1_df, isotope_raw_points_l[1], on='frame_id', how='left', suffixes=('_0', '_1')).sort_values(by='frame_id')
        number_of_missing_frames_1 = merged_df.intensity_1.isna().sum()
        feature_metrics['number_of_missing_frames_1'] = number_of_missing_frames_1
        feature_metrics['number_of_frames_1'] = len(ms1_frame_ids_1_df)
    else:
        feature_metrics['number_of_missing_frames_1'] = None
        feature_metrics['number_of_frames_1'] = None

    if (rt_metrics_l[2] is not None):
        rt_lower_2 = rt_metrics_l[2]['lower_x']
        rt_upper_2 = rt_metrics_l[2]['upper_x']
        ms1_frame_ids_2 = get_ms1_frame_ids(rt_lower_2, rt_upper_2)
        ms1_frame_ids_2_df = pd.DataFrame(ms1_frame_ids_2, columns=['frame_id'])
        ms1_frame_ids_2_df['intensity'] = 0
        merged_df = pd.merge(ms1_frame_ids_2_df, isotope_raw_points_l[2], on='frame_id', how='left', suffixes=('_0', '_1')).sort_values(by='frame_id')
        number_of_missing_frames_2 = merged_df.intensity_1.isna().sum()
        feature_metrics['number_of_missing_frames_2'] = number_of_missing_frames_2
        feature_metrics['number_of_frames_2'] = len(ms1_frame_ids_2_df)
    else:
        feature_metrics['number_of_missing_frames_2'] = None
        feature_metrics['number_of_frames_2'] = None

    # gather the metrics
    if feature_metrics is not None and isinstance(feature_metrics, FixedDict):
        feature_metrics = feature_metrics.get_dict()

    return feature_metrics

def calculate_feature_attributes(isotope_raw_points_l, rt_0_metrics, scan_0_metrics, sequence, charge, run_name, estimated_mono_mz):
    intensity = 0
    inferred = False
    isotope_idx_not_in_saturation = -1  # this means there are _no_ isotopes not in saturation
    isotope_intensities_l = None
    monoisotopic_mz = None
    monoisotopic_mass = None
    number_of_isotopes = 0

    # join the isotope dataframes together
    isotope_raw_points_df = pd.concat(isotope_raw_points_l, axis=0, sort=False)

    # re-calculate the intensity of each isotope by summing its point closest to the monoisotope apex and one point either side
    isotope_intensity_l = []
    isotope_idx_not_in_saturation = -1
    for isotope_idx in range(NUMBER_OF_ISOTOPES):
        isotope_df = isotope_raw_points_df[isotope_raw_points_df.isotope_idx == isotope_idx].copy()
        if len(isotope_df) > 0:
            # find the intensity by summing the maximum point in the frame closest to the RT apex, and the frame maximums either side
            frame_maximums_l = []
            for frame_id,group_df in isotope_df.groupby('frame_id'):
                frame_maximums_l.append(group_df.loc[group_df.intensity.idxmax()])
            frame_maximums_df = pd.DataFrame(frame_maximums_l)
            frame_maximums_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)
            frame_maximums_df.reset_index(drop=True, inplace=True)
            # find the index closest to the RT apex and the index either side
            if (rt_0_metrics is not None) and (rt_0_metrics['apex_x'] is not None):
                frame_maximums_df['rt_delta'] = np.abs(frame_maximums_df.retention_time_secs - rt_0_metrics['apex_x'])
                apex_idx = frame_maximums_df.rt_delta.idxmin()
            else:
                apex_idx = frame_maximums_df.intensity.idxmax()
            apex_idx_minus_one = max(0, apex_idx-1)
            apex_idx_plus_one = min(len(frame_maximums_df)-1, apex_idx+1)
            # sum the maximum intensity and the max intensity of the frame either side in RT
            summed_intensity = frame_maximums_df.loc[apex_idx_minus_one:apex_idx_plus_one].intensity.sum()
            # are any of the three points in saturation?
            isotope_in_saturation = (frame_maximums_df.loc[apex_idx_minus_one:apex_idx_plus_one].intensity.max() > SATURATION_INTENSITY)
            # keep the points used at the apex for calculating the intensity
            isotope_apex_points_l = [tuple(x) for x in frame_maximums_df[['mz','scan','frame_id','retention_time_secs','intensity']].loc[apex_idx_minus_one:apex_idx_plus_one].to_numpy()]
            # keep the raw points for each isotope, and those used at the apex for calculating the intensity
            isotope_points_l = [tuple(x) for x in isotope_df[['mz','scan','frame_id','retention_time_secs','intensity']].to_numpy()]
            # add the isotope to the list
            isotope_intensity_l.append((summed_intensity, isotope_in_saturation, isotope_points_l, isotope_apex_points_l))
            if (isotope_in_saturation == False) and (isotope_idx_not_in_saturation == -1):
                isotope_idx_not_in_saturation = isotope_idx
        else:
            # the isotope doesn't have any points so there's no point continuing
            if args.small_set_mode:
                print('calculate_feature_attributes: isotope {} doesn\'t have any points - stopping'.format(isotope_idx))
            break

    # calculate the monoisotopic m/z and mass
    monoisotopic_points_a = isotope_raw_points_df[isotope_raw_points_df.isotope_idx == 0][['mz','intensity']].to_numpy()
    monoisotopic_mz = mz_centroid(monoisotopic_points_a[:,1], monoisotopic_points_a[:,0])
    monoisotopic_mass = calculate_monoisotopic_mass_from_mz(monoisotopic_mz, charge)
    monoisotopic_mz_delta_ppm = (monoisotopic_mz - estimated_mono_mz) / estimated_mono_mz * 1e6

    # infer the intensity of peaks made up of points in saturation
    if len(isotope_intensity_l) > 0:
        isotope_intensities_df = pd.DataFrame(isotope_intensity_l, columns=['summed_intensity','saturated','isotope_points','isotope_apex_points'])
        # set up the default values
        isotope_intensities_df['inferred_intensity'] = isotope_intensities_df.summed_intensity  # set the summed intensity to be the default adjusted intensity for all isotopes
        isotope_intensities_df['inferred'] = False

        # adjust the monoisotopic intensity if it has points in saturation. We can use an isotope that's
        # not in saturation as a reference, as long as there is one
        if isotope_idx_not_in_saturation > 0:
            # using as a reference the most intense isotope that is not in saturation, derive the isotope intensities back to the monoisotopic
            Hpn = isotope_intensities_df.iloc[isotope_idx_not_in_saturation].summed_intensity
            for peak_number in reversed(range(1,isotope_idx_not_in_saturation+1)):
                phr = peak_ratio(monoisotopic_mass, peak_number, number_of_sulphur=0)
                if phr is not None:
                    Hpn_minus_1 = Hpn / phr
                    isotope_intensities_df.at[peak_number-1, 'inferred_intensity'] = int(Hpn_minus_1)
                    isotope_intensities_df.at[peak_number-1, 'inferred'] = True
                    Hpn = Hpn_minus_1
                else:
                    break

        intensity = int(isotope_intensities_df.iloc[0].inferred_intensity)    # the inferred saturation
        inferred = int(isotope_intensities_df.iloc[0].inferred)               # whether the monoisotope intensity was inferred

        isotope_intensities_l = [tuple(x) for x in isotope_intensities_df[['summed_intensity','saturated','inferred_intensity','inferred','isotope_points','isotope_apex_points']].to_numpy()]
        number_of_isotopes = len(isotope_intensities_df)
    else:
        isotope_intensities_l = None

    # calculate with the top-proportion method
    monoisotope_df = isotope_raw_points_df[isotope_raw_points_df.isotope_idx == 0]
    # find the maximum intensity by scan in each frame
    frame_ccs_cutoffs = []
    for group_name,group_df in monoisotope_df.groupby(['frame_id']):
        max_intensity = group_df.intensity.max()
        intensity_cutoff = (1.0 - TOP_CCS_PROPORTION_TO_INCLUDE) * max_intensity
        frame_ccs_cutoffs.append((group_name, intensity_cutoff))
    # trim the monoisotope according to the CCS cutoffs
    frames_l = []
    for ccs_cutoff in frame_ccs_cutoffs:
        frame_df = monoisotope_df[(monoisotope_df.frame_id == ccs_cutoff[0]) & (monoisotope_df.intensity >= ccs_cutoff[1])]
        frames_l.append(frame_df)
    monoisotope_trimmed_by_ccs_cutoff_df = pd.concat(frames_l, axis=0, sort=False)  # the monoisotope points trimmed by CCS cutoff
    # find the RT cutoff
    rt_flattened_df = monoisotope_trimmed_by_ccs_cutoff_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
    max_rt_intensity = rt_flattened_df.intensity.max()
    rt_intensity_cutoff = (1.0 - TOP_RT_PROPORTION_TO_INCLUDE) * max_rt_intensity
    # trim the RT-flattened monoisotope accordingly
    rt_flattened_with_cutoff_df = rt_flattened_df[rt_flattened_df.intensity >= rt_intensity_cutoff]
    # now sum the remaining points to calculate the intensity
    peak_proportion_intensity = rt_flattened_with_cutoff_df.intensity.sum()

    # package the feature attributes
    feature_attributes = {}
    feature_attributes['intensity'] = intensity                                         # sum of the maximum intensity and the max intensity of the frame either side in RT
    feature_attributes['inferred'] = inferred                                           # whether the mono intensity was inferred from peak height ratios
    feature_attributes['isotope_idx_not_in_saturation'] = isotope_idx_not_in_saturation # index of the first isotope that is not in saturation
    if rt_0_metrics is not None:
        feature_attributes['rt_apex'] = rt_0_metrics['apex_x']                          # the RT apex of the isolated mono peak
    else:
        feature_attributes['rt_apex'] = None
    if scan_0_metrics is not None:
        feature_attributes['scan_apex'] = scan_0_metrics['apex_x']                      # the CCS apex of the isolated mono peak
    else:
        feature_attributes['scan_apex'] = None
    feature_attributes['isotope_intensities_l'] = isotope_intensities_l                 # information about each isotope
    feature_attributes['monoisotopic_mz_centroid'] = monoisotopic_mz                    # mono m/z for the isolated peak
    feature_attributes['monoisotopic_mz_delta_ppm'] = monoisotopic_mz_delta_ppm         # delta m/z ppm from the estimated monoisotopic m/z
    feature_attributes['monoisotopic_mass'] = monoisotopic_mass                         # monoisotopic mass
    feature_attributes['number_of_isotopes'] = number_of_isotopes                       # the number of isotopes we found
    feature_attributes['peak_proportion_intensity'] = peak_proportion_intensity         # intensity calculated by taking a proportion of the top of the peak
    feature_attributes['peak_proportions'] = {'ccs_proportion':TOP_CCS_PROPORTION_TO_INCLUDE, 'rt_proportion':TOP_RT_PROPORTION_TO_INCLUDE}

    return feature_attributes

def extract_feature_metrics_at_coords(coordinates_d, data_obj, run_name, sequence, charge, target_mode):
    feature_metrics_attributes_l = []

    estimated_mono_mz = coordinates_d['mono_mz']
    estimated_scan_apex = coordinates_d['scan_apex']
    estimated_rt_apex = coordinates_d['rt_apex']

    # distance for looking either side of the scan and RT apex, based on the other times this sequence has been seen in this experiment
    SCAN_WIDTH = args.max_peak_width_ccs
    RT_WIDTH = args.max_peak_width_rt

    # the width to use for isotopic width, in Da
    MZ_TOLERANCE_PPM = 5  # +/- this amount
    MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4
    MS1_PEAK_DELTA = estimated_mono_mz * MZ_TOLERANCE_PERCENT / 100

    # the expected spacing between isotopes in the m/z dimension
    expected_spacing_mz = CARBON_MASS_DIFFERENCE / charge

    # define the region we will look in for the feature
    feature_region_mz_lower = estimated_mono_mz - expected_spacing_mz
    feature_region_mz_upper = estimated_mono_mz + (NUMBER_OF_ISOTOPES * expected_spacing_mz) + expected_spacing_mz
    scan_lower = estimated_scan_apex - SCAN_WIDTH
    scan_upper = estimated_scan_apex + SCAN_WIDTH
    rt_lower = estimated_rt_apex - RT_WIDTH
    rt_upper = estimated_rt_apex + RT_WIDTH

    if args.small_set_mode:
        print('coordinates for sequence {} charge {} in {} mode:\nestimated_mono_mz = {}\nestimated_scan_apex = {}\nestimated_rt_apex = {}\n\nfeature_region_mz_lower = {}\nfeature_region_mz_upper={}\nscan_lower={}\nscan_upper={}\nrt_lower={}\nrt_upper={}\n'.format(sequence, charge, "target" if target_mode else "decoy", estimated_mono_mz, estimated_scan_apex, estimated_rt_apex, feature_region_mz_lower, feature_region_mz_upper, scan_lower, scan_upper, rt_lower, rt_upper))

    isotope_peaks_l = []
    isotope_raw_points_l = []

    # load the ms1 points for this feature region
    feature_region_raw_points_df = data_obj[
        {
            "rt_values": slice(float(rt_lower), float(rt_upper)),
            "mz_values": slice(float(feature_region_mz_lower), float(feature_region_mz_upper)),
            "scan_indices": slice(int(scan_lower), int(scan_upper+1)),
            "precursor_indices": 0,  # ms1 frames only
        }
    ][['mz_values','scan_indices','frame_indices','rt_values','intensity_values']]
    feature_region_raw_points_df.rename(columns={'mz_values':'mz', 'scan_indices':'scan', 'frame_indices':'frame_id', 'rt_values':'retention_time_secs', 'intensity_values':'intensity'}, inplace=True)
    # downcast the data types to minimise the memory used
    int_columns = ['frame_id','scan','intensity']
    feature_region_raw_points_df[int_columns] = feature_region_raw_points_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
    float_columns = ['retention_time_secs']
    feature_region_raw_points_df[float_columns] = feature_region_raw_points_df[float_columns].apply(pd.to_numeric, downcast="float")

    MAXIMUM_NUMBER_OF_MONO_RT_PEAKS_FOR_TARGET_MODE = 10

    if len(feature_region_raw_points_df) > 0:
        # derive peaks for the monoisotopic and the isotopes

        # this section makes two lists for the isotopes:
        # - isotope_peaks_l is a list of tuples of the m/z centroid and the summed intensity for each isotope
        # - isotope_raw_points_l is a list of dataframes of the raw points for each isotope (that may be empty) with the raw points as (frame_id,mz,scan,intensity,retention_time_secs)
        for isotope_idx in range(NUMBER_OF_ISOTOPES):
            estimated_isotope_midpoint = estimated_mono_mz + (isotope_idx * expected_spacing_mz)
            # initial m/z isotope width
            isotope_mz_lower = estimated_isotope_midpoint - MS1_PEAK_DELTA
            isotope_mz_upper = estimated_isotope_midpoint + MS1_PEAK_DELTA
            isotope_raw_points_df = feature_region_raw_points_df[(feature_region_raw_points_df.mz >= isotope_mz_lower) & (feature_region_raw_points_df.mz <= isotope_mz_upper)].copy()
            if len(isotope_raw_points_df) == 0:
                # second attempt m/z isotope width
                MZ_TOLERANCE_PPM = 20  # +/- this amount
                MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4
                MS1_PEAK_DELTA = estimated_mono_mz * MZ_TOLERANCE_PERCENT / 100

                isotope_mz_lower = estimated_isotope_midpoint - MS1_PEAK_DELTA
                isotope_mz_upper = estimated_isotope_midpoint + MS1_PEAK_DELTA
                isotope_raw_points_df = feature_region_raw_points_df[(feature_region_raw_points_df.mz >= isotope_mz_lower) & (feature_region_raw_points_df.mz <= isotope_mz_upper)].copy()
            if len(isotope_raw_points_df) > 0:
                # print('found {} points for isotope {}'.format(len(isotope_raw_points_df), isotope_idx))
                isotope_raw_points_df['isotope_idx'] = isotope_idx
                # add the isotope's raw points to the list
                isotope_raw_points_l.append(isotope_raw_points_df)
                # centroid the raw points to get the peak for the isotope
                isotope_raw_points_a = isotope_raw_points_df[['mz','intensity']].values
                mz_cent = mz_centroid(isotope_raw_points_a[:,1], isotope_raw_points_a[:,0])
                # calculate the intensity
                summed_intensity = isotope_raw_points_a[:,1].sum()
                # add the peak to the list of isotopic peaks
                isotope_peaks_l.append((mz_cent, summed_intensity))
            else:
                break  # we couldn't find any points where this isotope should be, so let's stop
        # print('found {} isotopes'.format(len(isotope_peaks_l)))
        isotope_peaks_df = pd.DataFrame(isotope_peaks_l, columns=['mz_centroid','summed_intensity'])

        # clean up
        feature_region_raw_points_df = None

        if len(isotope_peaks_df) >= 3:  # we need at least three isotopes for this to work
            # update the monoisotopic m/z with the one we just calculated
            updated_mz_apex = isotope_peaks_df.iloc[0].mz_centroid

            # We have confidence in the accuracy of each dimension in decreasing order: m/z, RT, scan. Therefore we constrain
            # the cuboid by m/z first to find the peak in RT, then constrain the points to the RT peak's FWHM, then find
            # the peak in the scan dimension.

            # monoisotopic peak
            mono_raw_points_df = isotope_raw_points_l[0]

            # collapse the points onto the RT dimension
            rt_0_df = mono_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
            rt_0_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)
            rt_0_df['x'] = rt_0_df.retention_time_secs
            rt_0_metrics_d = fit_curve_to_flattened_isotope(flattened_points_df=rt_0_df, estimated_apex=estimated_rt_apex, estimated_peak_width=RT_WIDTH,
                                                            maximum_number_of_peaks=(MAXIMUM_NUMBER_OF_MONO_RT_PEAKS_FOR_TARGET_MODE if target_mode else 1),
                                                            isotope_dimension='RT', isotope_number=0, sequence=sequence, charge=charge, run_name=run_name)
            if rt_0_metrics_d['peaks'] is not None:
                # we may have detected multiple peaks in RT, and rather than pick one, we will gather the attributes and metrics for them all
                candidate_peaks_l = []
                for rt_peak_idx, rt_0_metrics in enumerate(rt_0_metrics_d['peaks']):
                    # monoisotopic peak
                    mono_raw_points_df = isotope_raw_points_l[0]
                    # update the RT apex from the estimator with the apex we determined by fitting a curve to the isolated monoisotopic peak
                    if (rt_0_metrics is not None) and (rt_0_metrics['apex_x'] is not None):
                        updated_rt_apex = rt_0_metrics['apex_x']
                    else:
                        updated_rt_apex = estimated_rt_apex
                    # collapse the points onto the mobility dimension, constraining the points to the FWHM of the peak in RT
                    if (rt_0_metrics['apex_x'] is not None) and (rt_0_metrics['lower_x'] is not None) and (rt_0_metrics['upper_x'] is not None):
                        mono_raw_points_df = mono_raw_points_df[(mono_raw_points_df.retention_time_secs >= rt_0_metrics['lower_x']) & (mono_raw_points_df.retention_time_secs <= rt_0_metrics['upper_x'])]
                    scan_0_df = mono_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()
                    scan_0_df.sort_values(by=['scan'], ascending=True, inplace=True)
                    scan_0_df['x'] = scan_0_df.scan
                    scan_0_metrics_d = fit_curve_to_flattened_isotope(flattened_points_df=scan_0_df, estimated_apex=estimated_scan_apex, estimated_peak_width=SCAN_WIDTH, maximum_number_of_peaks=1, isotope_dimension='CCS', isotope_number=0, sequence=sequence, charge=charge, run_name=run_name)
                    if scan_0_metrics_d['peaks'] is not None:
                        scan_0_metrics = scan_0_metrics_d['peaks'][0]
                    else:
                        scan_0_metrics = None
                    # update the CCS apex from the estimator with the apex we determined by fitting a curve to the isolated monoisotopic peak
                    if (scan_0_metrics is not None) and (scan_0_metrics['apex_x'] is not None):
                        updated_scan_apex = scan_0_metrics['apex_x']
                    else:
                        updated_scan_apex = estimated_scan_apex

                    # Isotope 1 peak
                    isotope_1_raw_points_df = isotope_raw_points_l[1]

                    # Collapse the points onto the RT dimension
                    rt_1_df = isotope_1_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
                    rt_1_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)
                    rt_1_df['x'] = rt_1_df.retention_time_secs
                    # we want the isolated peak nearest the monoisotopic peak apex, so we use the updated apex
                    rt_1_metrics_d = fit_curve_to_flattened_isotope(flattened_points_df=rt_1_df, estimated_apex=updated_rt_apex, estimated_peak_width=RT_WIDTH, maximum_number_of_peaks=1, isotope_dimension='RT', isotope_number=1, sequence=sequence, charge=charge, run_name=run_name)
                    if rt_1_metrics_d['peaks'] is not None:
                        rt_1_metrics = rt_1_metrics_d['peaks'][0]
                    else:
                        rt_1_metrics = None

                    # Collapse the points onto the mobility dimension
                    if (rt_1_metrics is not None) and (rt_1_metrics['apex_x'] is not None) and (rt_1_metrics['lower_x'] is not None) and (rt_1_metrics['upper_x'] is not None):
                        isotope_1_raw_points_df = isotope_1_raw_points_df[(isotope_1_raw_points_df.retention_time_secs >= rt_1_metrics['lower_x']) & (isotope_1_raw_points_df.retention_time_secs <= rt_1_metrics['upper_x'])]
                    scan_1_df = isotope_1_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()
                    scan_1_df.sort_values(by=['scan'], ascending=True, inplace=True)
                    scan_1_df['x'] = scan_1_df.scan
                    # we want the isolated peak nearest the monoisotopic peak apex, so we use the updated apex
                    scan_1_metrics_d = fit_curve_to_flattened_isotope(flattened_points_df=scan_1_df, estimated_apex=updated_scan_apex, estimated_peak_width=SCAN_WIDTH, maximum_number_of_peaks=1, isotope_dimension='CCS', isotope_number=1, sequence=sequence, charge=charge, run_name=run_name)
                    if scan_1_metrics_d['peaks'] is not None:
                        scan_1_metrics = scan_1_metrics_d['peaks'][0]
                    else:
                        scan_1_metrics = None

                    # Isotope 2 peak
                    isotope_2_raw_points_df = isotope_raw_points_l[2]

                    # Collapse the points onto the RT dimension
                    rt_2_df = isotope_2_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
                    rt_2_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)
                    rt_2_df['x'] = rt_2_df.retention_time_secs
                    # we want the isolated peak nearest the monoisotopic peak apex, so we use the updated apex
                    rt_2_metrics_d = fit_curve_to_flattened_isotope(flattened_points_df=rt_2_df, estimated_apex=updated_rt_apex, estimated_peak_width=RT_WIDTH, maximum_number_of_peaks=1, isotope_dimension='RT', isotope_number=2, sequence=sequence, charge=charge, run_name=run_name)
                    if rt_2_metrics_d['peaks'] is not None:
                        rt_2_metrics = rt_2_metrics_d['peaks'][0]
                    else:
                        rt_2_metrics = None

                    # Collapse the points onto the mobility dimension
                    if (rt_2_metrics is not None) and (rt_2_metrics['apex_x'] is not None) and (rt_2_metrics['lower_x'] is not None) and (rt_2_metrics['upper_x'] is not None):
                        isotope_2_raw_points_df = isotope_2_raw_points_df[(isotope_2_raw_points_df.retention_time_secs >= rt_2_metrics['lower_x']) & (isotope_2_raw_points_df.retention_time_secs <= rt_2_metrics['upper_x'])]
                    scan_2_df = isotope_2_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()
                    scan_2_df.sort_values(by=['scan'], ascending=True, inplace=True)
                    scan_2_df['x'] = scan_2_df.scan
                    # we want the isolated peak nearest the monoisotopic peak apex, so we use the updated apex
                    scan_2_metrics_d = fit_curve_to_flattened_isotope(flattened_points_df=scan_2_df, estimated_apex=updated_scan_apex, estimated_peak_width=SCAN_WIDTH, maximum_number_of_peaks=1, isotope_dimension='CCS', isotope_number=2, sequence=sequence, charge=charge, run_name=run_name)
                    if scan_2_metrics_d['peaks'] is not None:
                        scan_2_metrics = scan_2_metrics_d['peaks'][0]
                    else:
                        scan_2_metrics = None

                    # bundle up the data for the metrics calculation, which uses only the first three isotopes
                    trimmed_isotopes_0_2_raw_points_l = [mono_raw_points_df, isotope_1_raw_points_df, isotope_2_raw_points_df]  # the raw points for the isotopes, trimmed by the flattened curve fitting
                    rt_metrics_l = [rt_0_metrics, rt_1_metrics, rt_2_metrics]
                    scan_metrics_l = [scan_0_metrics, scan_1_metrics, scan_2_metrics]

                    # calculate the feature metrics
                    # - in this step we use the estimated apexes because we are measuring how close it was to the derived apex
                    # - isotope_peaks_df is the m/z centroid and summed intensity of each 'raw' isotope
                    # - trimmed_isotopes_0_2_raw_points_l is a list of dataframes containing the raw points for each isotope trimmed by the curve fitting
                    feature_metrics = calculate_feature_metrics(isotope_peaks_df=isotope_peaks_df[:3], isotope_raw_points_l=trimmed_isotopes_0_2_raw_points_l, estimated_mono_mz=estimated_mono_mz, estimated_scan_apex=estimated_scan_apex, estimated_rt_apex=estimated_rt_apex, rt_metrics_l=rt_metrics_l, scan_metrics_l=scan_metrics_l, expected_spacing_mz=expected_spacing_mz, charge=charge)

                    # bundle up the data for the attributes calculation - this uses all the isotopes we found
                    # these points are trimmed to the extent of the monoisotopic
                    isotope_raw_points_for_attributes_l = []
                    for isotope_idx in range(len(isotope_raw_points_l)):
                        isotope_raw_points_df = isotope_raw_points_l[isotope_idx]
                        # # trim the isotope points by the RT extent of the monoisotopic
                        # if (rt_0_metrics is not None) and (rt_0_metrics['lower_x'] is not None) and (rt_0_metrics['upper_x'] is not None):
                        #     isotope_raw_points_df = isotope_raw_points_df[(isotope_raw_points_df.retention_time_secs >= rt_0_metrics['lower_x']) & (isotope_raw_points_df.retention_time_secs <= rt_0_metrics['upper_x'])]
                        # # trim the isotope points by the CCS extent of the monoisotopic
                        # if (scan_0_metrics is not None) and (scan_0_metrics['lower_x'] is not None) and (scan_0_metrics['upper_x'] is not None):
                        #     isotope_raw_points_df = isotope_raw_points_df[(isotope_raw_points_df.scan >= scan_0_metrics['lower_x']) & (isotope_raw_points_df.scan <= scan_0_metrics['upper_x'])]
                        isotope_raw_points_for_attributes_l.append(isotope_raw_points_df)

                    # calculate the feature attributes
                    # - isotope_raw_points_for_attributes_l is the raw points for each isotope trimmed to the extent of the monoisotopic
                    feature_attributes = calculate_feature_attributes(isotope_raw_points_l=isotope_raw_points_for_attributes_l, rt_0_metrics=rt_0_metrics, scan_0_metrics=scan_0_metrics, sequence=sequence, charge=charge, run_name=run_name, estimated_mono_mz=estimated_mono_mz)

                    # we need to make sure there's a viable number of isotopes
                    if args.small_set_mode:
                        print('sequence {}, charge {}, number of isotopes {}, rt apex {}, scan apex {}'.format(sequence, charge, feature_attributes['number_of_isotopes'], feature_attributes['rt_apex'], feature_attributes['scan_apex']))

                    if (feature_attributes['number_of_isotopes'] >= MINIMUM_NUMBER_OF_ISOTOPES_FOR_VIABLE_FEATURE) and (feature_attributes['rt_apex'] is not None) and (feature_attributes['scan_apex'] is not None):
                        # -------------------------
                        # add in some monoisotopic peak RT attributes for visualisation and debugging
                        # -------------------------
                        # the isotope points flattened to the RT dimension and filtered with a Savgol filter - useful for visualisation
                        if (rt_0_metrics_d is not None) and (rt_0_metrics_d['filtered_points'] is not None):
                            feature_attributes["mono_filtered_points_l"] = list(rt_0_metrics_d['filtered_points'])
                        else:
                            feature_attributes["mono_filtered_points_l"] = None
                        if (rt_1_metrics_d is not None) and (rt_1_metrics_d['filtered_points'] is not None):
                            feature_attributes["isotope_1_filtered_points_l"] = list(rt_1_metrics_d['filtered_points'])
                        else:
                            feature_attributes["isotope_1_filtered_points_l"] = None
                        if (rt_2_metrics_d is not None) and (rt_2_metrics_d['filtered_points'] is not None):
                            feature_attributes["isotope_2_filtered_points_l"] = list(rt_2_metrics_d['filtered_points'])
                        else:
                            feature_attributes["isotope_2_filtered_points_l"] = None
                        # bounds of the feature in RT
                        if (rt_0_metrics is not None) and (rt_0_metrics['lower_x'] is not None) and (rt_0_metrics['upper_x'] is not None):
                            feature_attributes["mono_rt_bounds"] = (rt_0_metrics['lower_x'], rt_0_metrics['upper_x'])
                        else:
                            feature_attributes["mono_rt_bounds"] = None
                        if (rt_1_metrics is not None) and (rt_1_metrics['lower_x'] is not None) and (rt_1_metrics['upper_x'] is not None):
                            feature_attributes["isotope_1_rt_bounds"] = (rt_1_metrics['lower_x'], rt_1_metrics['upper_x'])
                        else:
                            feature_attributes["isotope_1_rt_bounds"] = None
                        if (rt_2_metrics is not None) and (rt_2_metrics['lower_x'] is not None) and (rt_2_metrics['upper_x'] is not None):
                            feature_attributes["isotope_2_rt_bounds"] = (rt_2_metrics['lower_x'], rt_2_metrics['upper_x'])
                        else:
                            feature_attributes["isotope_2_rt_bounds"] = None
                        # bounds of the feature in CCS
                        if (scan_0_metrics is not None) and (scan_0_metrics['lower_x'] is not None) and (scan_0_metrics['upper_x'] is not None):
                            feature_attributes["mono_scan_bounds"] = (scan_0_metrics['lower_x'], scan_0_metrics['upper_x'])
                        else:
                            feature_attributes["mono_scan_bounds"] = None
                        if (scan_1_metrics is not None) and (scan_1_metrics['lower_x'] is not None) and (scan_1_metrics['upper_x'] is not None):
                            feature_attributes["isotope_1_scan_bounds"] = (scan_1_metrics['lower_x'], scan_1_metrics['upper_x'])
                        else:
                            feature_attributes["isotope_1_scan_bounds"] = None
                        if (scan_2_metrics is not None) and (scan_2_metrics['lower_x'] is not None) and (scan_2_metrics['upper_x'] is not None):
                            feature_attributes["isotope_2_scan_bounds"] = (scan_2_metrics['lower_x'], scan_2_metrics['upper_x'])
                        else:
                            feature_attributes["isotope_2_scan_bounds"] = None
                        # add this to the list of candidate peaks
                        peak_selection_attributes = (feature_metrics['rt_isotope_correlation'],feature_metrics['rt_isotope_cv'],feature_metrics['scan_isotope_correlation'],feature_metrics['scan_isotope_cv'],feature_attributes['monoisotopic_mz_delta_ppm'],feature_metrics['delta_rt'],feature_metrics['delta_scan'])
                        if args.small_set_mode:
                            print('peak_selection_attributes: {}'.format(peak_selection_attributes))
                        if (feature_metrics is not None) and (feature_attributes is not None):
                            candidate_peak_d = {}
                            candidate_peak_d['feature_metrics_attributes'] = (sequence, charge, int(rt_peak_idx), feature_metrics, feature_attributes)
                            candidate_peak_d['peak_selection_attributes'] = peak_selection_attributes
                            candidate_peaks_l.append(candidate_peak_d)
                        else:
                            print('feature_metrics is None: {}, feature_attributes is None: {}'.format((feature_metrics is None), (feature_attributes is None)))
                    else:
                        if args.small_set_mode:
                            if not (feature_attributes['number_of_isotopes'] >= MINIMUM_NUMBER_OF_ISOTOPES_FOR_VIABLE_FEATURE):
                                print('feature attributes has {} isotopes'.format(feature_attributes['number_of_isotopes']))
                            if not (feature_attributes['rt_apex'] is not None):
                                print('feature attributes has None rt_apex')
                            if not (feature_attributes['scan_apex'] is not None):
                                print('feature attributes has None scan_apex')

                if len(candidate_peaks_l) > 0:
                    # decide which candidate peak to add for this sequence
                    candidate_peaks_df = pd.DataFrame([item['peak_selection_attributes'] for item in candidate_peaks_l], columns=['rt_isotope_correlation','rt_isotope_cv','scan_isotope_correlation','scan_isotope_cv','monoisotopic_mz_delta_ppm','delta_rt','delta_scan'])
                    candidate_peaks_df.dropna(inplace=True)  # drop any rows that contain None
                    if len(candidate_peaks_df) > 0:
                        # find the candidate peak with the best isotope correlation in RT
                        candidate_peak_idx = candidate_peaks_df.rt_isotope_correlation.idxmax()
                        # select the peak corresponding to this index
                        feature_metrics_attributes_l.append(candidate_peaks_l[candidate_peak_idx]['feature_metrics_attributes'])
                        print("extraction success for mode {}, sequence {}, charge {}".format("target" if target_mode else "decoy", sequence, charge))
                        if args.small_set_mode:
                            print('sequence {}, charge {}, run {}:'.format(sequence, charge, run_name))
                            print(candidate_peaks_df.to_string())
                            print("chose index {}".format(candidate_peak_idx))
                            print()
                    else:
                        if args.small_set_mode:
                            print("extract_feature_metrics_at_coords failed for mode {}: no L2 candidate peaks were found, sequence {}, charge {}, run {}.".format("target" if target_mode else "decoy", sequence, charge, run_name))
                        feature_metrics_attributes_l = []
                else:
                    if args.small_set_mode:
                        print("extract_feature_metrics_at_coords failed for mode {}: no L1 candidate peaks were found, sequence {}, charge {}, run {}.".format("target" if target_mode else "decoy", sequence, charge, run_name))
                    feature_metrics_attributes_l = []
            else:
                if args.small_set_mode:
                    print("extract_feature_metrics_at_coords failed for mode {}: no peaks were found in the monoisotopic peak flattened to the RT dimension, sequence {}, charge {}, run {}.".format("target" if target_mode else "decoy", sequence, charge, run_name))
                feature_metrics_attributes_l = []
        else:
            if args.small_set_mode:
                print("extract_feature_metrics_at_coords failed for mode {}: only found {} isotopes, sequence {}, charge {}, run {}.".format("target" if target_mode else "decoy", len(isotope_peaks_df), sequence, charge, run_name))
            feature_metrics_attributes_l = []
    else:
        if args.small_set_mode:
            print("extract_feature_metrics_at_coords failed for mode {}: no points were found in the feature extraction region, sequence {}, charge {}, run {}.".format("target" if target_mode else "decoy", sequence, charge, run_name))
        feature_metrics_attributes_l = []

    # time_stop = time.time()
    # print("sequence {}, charge {}: {} seconds".format(sequence, charge, round(time_stop-time_start,1)))

    # keep some debug information
    if args.small_set_mode and target_mode:
        info_d = {}
        info_d['sequence'] = sequence
        info_d['charge'] = charge
        info_d['run_name'] = run_name
        info_d['estimated_coordinates_d'] = coordinates_d
        info_d['rt_flattened_l'] = [rt_0_df.to_dict('records'), rt_1_df.to_dict('records'), rt_2_df.to_dict('records')]
        info_d['scan_flattened_l'] = [scan_0_df.to_dict('records'), scan_1_df.to_dict('records'), scan_2_df.to_dict('records')]
        info_d['feature_metrics_attributes_l'] = feature_metrics_attributes_l
        info_d['selected_peak_index'] = candidate_peak_idx
        DEBUG_DIR = "{}/debug".format(EXPERIMENT_DIR)
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)
        with open('{}/run-{}-sequence-{}-metrics.json'.format(DEBUG_DIR, run_name, sequence), 'w') as f: 
            json.dump(info_d, fp=f, cls=NpEncoder)

    return feature_metrics_attributes_l


####################################################################

parser = argparse.ArgumentParser(description='Using the run-specific coordinate estimators, for each of the library sequences, from each run, extract metrics for feature targets and decoys to collate a training set.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.', required=False)
parser.add_argument('-ssms','--small_set_mode_size', type=int, default='100', help='The number of identifications to sample for small set mode.', required=False)
parser.add_argument('-ssseq','--small_set_sequence', type=str, help='Only extract this sequence.', required=False)
parser.add_argument('-sschr','--small_set_charge', type=int, help='The charge for the selected sequence.', required=False)
parser.add_argument('-mpwrt','--max_peak_width_rt', type=int, default=10, help='Maximum peak width tolerance for the extraction from the estimated coordinate in RT.', required=False)
parser.add_argument('-mpwccs','--max_peak_width_ccs', type=int, default=20, help='Maximum peak width tolerance for the extraction from the estimated coordinate in CCS.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./tfde/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-d','--denoised', action='store_true', help='Use the denoised version of the raw database.')
parser.add_argument('-pdm','--precursor_definition_method', type=str, choices=['pasef','3did','mq'], default='pasef', help='The method used to define the precursor cuboids.', required=False)
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

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
FRAME_TYPE_MS1 = cfg.getint('common','FRAME_TYPE_MS1')
ADD_C_CYSTEINE_DA = cfg.getfloat('common','ADD_C_CYSTEINE_DA')
PROTON_MASS = cfg.getfloat('common','PROTON_MASS')
CARBON_MASS_DIFFERENCE = cfg.getfloat('common','CARBON_MASS_DIFFERENCE')
SATURATION_INTENSITY = cfg.getint('common','SATURATION_INTENSITY')
MAXIMUM_Q_VALUE = cfg.getfloat('common','MAXIMUM_Q_VALUE')

MAXIMUM_Q_VALUE_FOR_CLASSIFIER_TRAINING_SET = cfg.getfloat('extraction','MAXIMUM_Q_VALUE_FOR_CLASSIFIER_TRAINING_SET')
NUMBER_OF_ISOTOPES = cfg.getint('extraction','NUMBER_OF_ISOTOPES')
MINIMUM_NUMBER_OF_ISOTOPES_FOR_VIABLE_FEATURE = cfg.getint('extraction','MINIMUM_NUMBER_OF_ISOTOPES_FOR_VIABLE_FEATURE')
TOP_CCS_PROPORTION_TO_INCLUDE = cfg.getfloat('extraction','TOP_CCS_PROPORTION_TO_INCLUDE')
TOP_RT_PROPORTION_TO_INCLUDE = cfg.getfloat('extraction','TOP_RT_PROPORTION_TO_INCLUDE')

# check the raw database exists
if args.denoised:
    RAW_DATABASE_BASE_DIR = "{}/raw-databases/denoised".format(EXPERIMENT_DIR)
else:
    RAW_DATABASE_BASE_DIR = "{}/raw-databases".format(EXPERIMENT_DIR)
RAW_DATABASE_NAME = "{}/{}.d".format(RAW_DATABASE_BASE_DIR, args.run_name)
if not os.path.exists(RAW_DATABASE_NAME):
    print("The raw database is required but doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)

# create the TimsTOF object
RAW_HDF_FILE = '{}.hdf'.format(args.run_name)
RAW_HDF_PATH = '{}/{}'.format(RAW_DATABASE_BASE_DIR, RAW_HDF_FILE)
if not os.path.isfile(RAW_HDF_PATH):
    print('{} doesn\'t exist so loading the raw data from {}'.format(RAW_HDF_PATH, RAW_DATABASE_NAME))
    data = alphatims.bruker.TimsTOF(RAW_DATABASE_NAME)
    print('saving to {}'.format(RAW_HDF_PATH))
    _ = data.save_as_hdf(
        directory=RAW_DATABASE_BASE_DIR,
        file_name=RAW_HDF_FILE,
        overwrite=True
    )
else:
    print('loading raw data from {}'.format(RAW_HDF_PATH))
    data = alphatims.bruker.TimsTOF(RAW_HDF_PATH)

# load the MS1 frame IDs
ms1_frame_properties_df = load_ms1_frame_ids(RAW_DATABASE_NAME)

# set up the coordinate estimators directory
COORDINATE_ESTIMATORS_DIR = "{}/coordinate-estimators".format(EXPERIMENT_DIR)
if not os.path.exists(COORDINATE_ESTIMATORS_DIR):
    print("The coordinate estimators directory is required but doesn't exist: {}".format(COORDINATE_ESTIMATORS_DIR))
    sys.exit(1)

# load the sequence library
SEQUENCE_LIBRARY_DIR = "{}/sequence-library-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.feather".format(SEQUENCE_LIBRARY_DIR)
if not os.path.isfile(SEQUENCE_LIBRARY_FILE_NAME):
    print("The sequences library file doesn't exist: {}".format(SEQUENCE_LIBRARY_FILE_NAME))
    sys.exit(1)
else:
    library_sequences_for_this_run_df = pd.read_feather(SEQUENCE_LIBRARY_FILE_NAME)
    library_sequences_for_this_run_df = library_sequences_for_this_run_df[(library_sequences_for_this_run_df.q_value <= MAXIMUM_Q_VALUE)]
    print("loaded {} sequences with q-value less than {} from the library {}".format(len(library_sequences_for_this_run_df), MAXIMUM_Q_VALUE, SEQUENCE_LIBRARY_FILE_NAME))
    # for small set mode, randomly select some sequences
    if args.small_set_mode:
        if args.small_set_sequence is None:
            library_sequences_for_this_run_df.sort_values(by=['number_of_runs_identified','q_value','experiment_intensity_mean'], ascending=[False,True,False], inplace=True)
            library_sequences_for_this_run_df.reset_index(drop=True, inplace=True)
            library_sequences_for_this_run_df = library_sequences_for_this_run_df[:args.small_set_mode_size]
        else:
            library_sequences_for_this_run_df = library_sequences_for_this_run_df[(library_sequences_for_this_run_df.sequence == args.small_set_sequence) & (library_sequences_for_this_run_df.charge == args.small_set_charge)]
        print("trimmed to {} sequences for small set mode".format(len(library_sequences_for_this_run_df)))

# check the target decoy classifier directory exists
TARGET_DECOY_MODEL_DIR = "{}/target-decoy-models".format(EXPERIMENT_DIR)
if not os.path.exists(TARGET_DECOY_MODEL_DIR):
    print("The target-decoy classifier directory does not exist: {}".format(TARGET_DECOY_MODEL_DIR))

# remove the output file if it exists
LIBRARY_SEQUENCES_WITH_METRICS_FILENAME = '{}/library-sequences-in-run-{}.pkl'.format(TARGET_DECOY_MODEL_DIR, args.run_name)
if os.path.isfile(LIBRARY_SEQUENCES_WITH_METRICS_FILENAME):
    os.remove(LIBRARY_SEQUENCES_WITH_METRICS_FILENAME)

# set all the sequences with this run name to match the metrics we extract for each
library_sequences_for_this_run_df['run_name'] = args.run_name

print("calculating the feature metrics for the library sequences in run {}".format(args.run_name))

# load the coordinate estimators
MZ_ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, args.run_name, 'mz')
SCAN_ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, args.run_name, 'scan')
RT_ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, args.run_name, 'rt')

with open(MZ_ESTIMATOR_MODEL_FILE_NAME, 'rb') as file:
    mz_estimator = pickle.load(file)
with open(SCAN_ESTIMATOR_MODEL_FILE_NAME, 'rb') as file:
    scan_estimator = pickle.load(file)
with open(RT_ESTIMATOR_MODEL_FILE_NAME, 'rb') as file:
    rt_estimator = pickle.load(file)

# calculate the target coordinates
print("calculating the target coordinates for each sequence-charge")
library_sequences_for_this_run_df['target_coords'] = library_sequences_for_this_run_df.apply(lambda row: estimate_target_coordinates(row, mz_estimator, scan_estimator, rt_estimator), axis=1)

# calculate the decoy coordinates
print("calculating the decoy coordinates for each sequence-charge")
library_sequences_for_this_run_df['decoy_coords'] = library_sequences_for_this_run_df.apply(lambda row: calculate_decoy_coordinates(row), axis=1)

# extract feature metrics from the target coordinates for each sequence in the run
print("extracting feature metrics from the target coordinates")
target_metrics_l = [extract_feature_metrics_at_coords(coordinates_d=row.target_coords, data_obj=data, run_name=args.run_name, sequence=row.sequence, charge=row.charge, target_mode=True) for row in library_sequences_for_this_run_df.itertuples()]
flattened_target_metrics_l = [item for sublist in target_metrics_l for item in sublist]  # target_metrics_l is a list of lists, so we need to flatten it
target_metrics_df = pd.DataFrame(flattened_target_metrics_l, columns=['sequence','charge','peak_idx','target_metrics','attributes'])
# merge the target results with the library sequences for this run
library_sequences_with_target_metrics_df = pd.merge(library_sequences_for_this_run_df, target_metrics_df, how='left', left_on=['sequence','charge'], right_on=['sequence','charge'])

# extract feature metrics from the decoy coordinates for each sequence in the run
print("extracting feature metrics from the decoy coordinates")
decoy_metrics_l = [extract_feature_metrics_at_coords(coordinates_d=row.decoy_coords, data_obj=data, run_name=args.run_name, sequence=row.sequence, charge=row.charge, target_mode=False) for row in library_sequences_for_this_run_df.itertuples()]
flattened_decoy_metrics_l = [item for sublist in decoy_metrics_l for item in sublist]  # decoy_metrics_l is a list of lists, so we need to flatten it
decoy_metrics_df = pd.DataFrame(flattened_decoy_metrics_l, columns=['sequence','charge','peak_idx','decoy_metrics','attributes'])
# don't include the attributes because we're not interested in the decoy's attributes
decoy_metrics_df.drop(['attributes','peak_idx'], axis=1, inplace=True)

# join the two together to form the target and decoy metrics for each library sequence
library_sequences_for_this_run_df = pd.merge(library_sequences_with_target_metrics_df, decoy_metrics_df, how='left', left_on=['sequence','charge'], right_on=['sequence','charge'])

# remove the rubbish target_metrics and attributes
library_sequences_for_this_run_df = library_sequences_for_this_run_df[(library_sequences_for_this_run_df.target_metrics.notna()) & (library_sequences_for_this_run_df.attributes.notna())]

# downcast the data types to minimise the memory used
int_columns = ['charge','number_of_runs_identified','peak_idx']
library_sequences_for_this_run_df[int_columns] = library_sequences_for_this_run_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
float_columns = ['experiment_scan_mean','experiment_scan_std_dev','experiment_scan_peak_width','experiment_rt_mean','experiment_rt_std_dev','experiment_rt_peak_width','experiment_intensity_mean','experiment_intensity_std_dev','q_value']
library_sequences_for_this_run_df[float_columns] = library_sequences_for_this_run_df[float_columns].apply(pd.to_numeric, downcast="float")

# save the metrics for this run
print("writing {} metrics & attributes for the library sequences to {}".format(len(library_sequences_for_this_run_df), LIBRARY_SEQUENCES_WITH_METRICS_FILENAME))
library_sequences_for_this_run_df.to_pickle(LIBRARY_SEQUENCES_WITH_METRICS_FILENAME)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
