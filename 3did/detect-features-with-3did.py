import pandas as pd
import numpy as np
import peakutils
from scipy import signal
import math
import os
import time
import argparse
import ray
import sqlite3
import sys
import multiprocessing as mp
import pickle
import configparser
from configparser import ExtendedInterpolation
import json
from ms_deisotope import deconvolute_peaks, averagine
import warnings
from scipy.optimize import OptimizeWarning
from os.path import expanduser
import numba
import line_profiler

# set up the indexes we need for queries
def create_indexes(db_file_name):
    db_conn = sqlite3.connect(db_file_name)
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_extract_cuboids_1 on frames (frame_type,retention_time_secs,scan,mz)")
    db_conn.close()

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers

# determine the maximum filter length for the number of points
def find_filter_length(number_of_points):
    filter_lengths = [51,11,5]  # must be a positive odd number, greater than the polynomial order, and less than the number of points to be filtered
    return filter_lengths[next(x[0] for x in enumerate(filter_lengths) if x[1] < number_of_points)]

# define a straight line to exclude the charge-1 cloud
def scan_coords_for_single_charge_region(mz_lower, mz_upper):
    scan_for_mz_lower = max(int(-1 * ((1.2 * mz_lower) - 1252)), 0)
    scan_for_mz_upper = max(int(-1 * ((1.2 * mz_upper) - 1252)), 0)
    return {'scan_for_mz_lower':scan_for_mz_lower, 'scan_for_mz_upper':scan_for_mz_upper}

# calculate the intensity-weighted centroid
# takes a numpy array of intensity, and another of mz
def intensity_weighted_centroid(_int_f, _x_f):
    return ((_int_f/_int_f.sum()) * _x_f).sum()

# find 3sigma for a specified m/z
def calculate_peak_delta(mz):
    delta_m = mz / INSTRUMENT_RESOLUTION  # FWHM of the peak
    sigma = delta_m / 2.35482  # std dev is FWHM / 2.35482. See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    peak_delta = 3 * sigma  # 99.7% of values fall within +/- 3 sigma
    return peak_delta

# peaks_a is a numpy array of [mz,intensity]
# returns a numpy array of [intensity_weighted_centroid,summed_intensity]
def intensity_descent(peaks_a, peak_delta=None):
    # intensity descent
    peaks_l = []
    while len(peaks_a) > 0:
        # find the most intense point
        max_intensity_index = np.argmax(peaks_a[:,1])
        peak_mz = peaks_a[max_intensity_index,0]
        if peak_delta == None:
            peak_delta = calculate_peak_delta(mz=peak_mz)
        peak_mz_lower = peak_mz - peak_delta
        peak_mz_upper = peak_mz + peak_delta

        # get all the raw points within this m/z region
        peak_indexes = np.where((peaks_a[:,0] >= peak_mz_lower) & (peaks_a[:,0] <= peak_mz_upper))[0]
        if len(peak_indexes) > 0:
            mz_cent = intensity_weighted_centroid(peaks_a[peak_indexes,1], peaks_a[peak_indexes,0])
            summed_intensity = peaks_a[peak_indexes,1].sum()
            peaks_l.append((mz_cent, summed_intensity))
            # remove the raw points assigned to this peak
            peaks_a = np.delete(peaks_a, peak_indexes, axis=0)
    return np.array(peaks_l)

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

# calculate the characteristics of the isotopes in the feature envelope
def determine_isotope_characteristics(envelope, rt_apex, monoisotopic_mass, feature_region_3d_df, summary_df):
    voxels_processed = set()
    # calculate the isotope intensities from the constrained raw points
    isotopes_l = []
    for idx,isotope in enumerate(envelope):
        # gather the points that belong to this isotope
        iso_mz = isotope[0]
        iso_intensity = isotope[1]
        iso_mz_delta = calculate_peak_delta(iso_mz)
        iso_mz_lower = iso_mz - iso_mz_delta
        iso_mz_upper = iso_mz + iso_mz_delta
        isotope_df = feature_region_3d_df[(feature_region_3d_df.mz >= iso_mz_lower) & (feature_region_3d_df.mz <= iso_mz_upper)]
        # calculate the isotope's intensity
        if len(isotope_df) > 0:
            # record the voxels included by each isotope
            voxel_ids_for_isotope = voxels_for_points(points_df=isotope_df, voxels_df=summary_df)
            # add the voxels included in the feature's points to the list of voxels already processed
            voxels_processed.update(voxel_ids_for_isotope)
            # find the intensity by summing the maximum point in the frame closest to the RT apex, and the frame maximums either side
            frame_maximums_l = []
            for frame_id,group_df in isotope_df.groupby('frame_id'):
                frame_maximums_l.append(group_df.loc[group_df.intensity.idxmax()])
            frame_maximums_df = pd.DataFrame(frame_maximums_l)
            frame_maximums_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)
            frame_maximums_df.reset_index(drop=True, inplace=True)
            # find the index closest to the RT apex and the index either side
            frame_maximums_df['rt_delta'] = np.abs(frame_maximums_df.retention_time_secs - rt_apex)
            apex_idx = frame_maximums_df.rt_delta.idxmin()
            apex_idx_minus_one = max(0, apex_idx-1)
            apex_idx_plus_one = min(len(frame_maximums_df)-1, apex_idx+1)
            # sum the maximum intensity and the max intensity of the frame either side in RT
            summed_intensity = frame_maximums_df.loc[apex_idx_minus_one:apex_idx_plus_one].intensity.sum()
            # are any of the three points in saturation?
            isotope_in_saturation = (frame_maximums_df.loc[apex_idx_minus_one:apex_idx_plus_one].intensity.max() > SATURATION_INTENSITY)
            # add the isotope to the list
            isotopes_l.append({'mz':iso_mz, 'mz_lower':iso_mz_lower, 'mz_upper':iso_mz_upper, 'intensity':summed_intensity, 'saturated':isotope_in_saturation})
        else:
            break
    isotopes_df = pd.DataFrame(isotopes_l)

    # set the summed intensity and m/z to be the default adjusted intensity for all isotopes
    isotopes_df['inferred_intensity'] = isotopes_df.intensity
    isotopes_df['inferred'] = False

    # if the mono is saturated and there are non-saturated isotopes to use as a reference...
    outcome = ''
    if (isotopes_df.iloc[0].saturated == True):
        outcome = 'monoisotopic_saturated_adjusted'
        if (len(isotopes_df[isotopes_df.saturated == False]) > 0):
            # find the first unsaturated isotope
            unsaturated_idx = isotopes_df[(isotopes_df.saturated == False)].iloc[0].name

            # using as a reference the most intense isotope that is not in saturation, derive the isotope intensities back to the monoisotopic
            Hpn = isotopes_df.iloc[unsaturated_idx].intensity
            for peak_number in reversed(range(1,unsaturated_idx+1)):
                # calculate the phr for the next-lower peak
                phr = peak_ratio(monoisotopic_mass, peak_number, number_of_sulphur=0)
                if phr is not None:
                    Hpn_minus_1 = Hpn / phr
                    isotopes_df.at[peak_number-1, 'inferred_intensity'] = int(Hpn_minus_1)
                    isotopes_df.at[peak_number-1, 'inferred'] = True
                    Hpn = Hpn_minus_1
                else:
                    outcome = 'could_not_calculate_phr'
                    break
        else:
            outcome = 'no_nonsaturated_isotopes'
    else:
        outcome = 'monoisotopic_not_saturated'

    # package the result
    result_d = {}

    result_d['intensity_without_saturation_correction'] = isotopes_df.iloc[:3].intensity.sum()  # only take the first three isotopes for intensity, as the number of isotopes varies
    result_d['intensity_with_saturation_correction'] = isotopes_df.iloc[:3].inferred_intensity.sum()
    result_d['mono_intensity_adjustment_outcome'] = outcome

    result_d['mono_mz'] = isotopes_df.iloc[0].mz

    result_d['isotopic_peaks'] = isotopes_df.to_dict('records')
    result_d['voxels_processed'] = voxels_processed
    return result_d

# calculate the monoisotopic mass    
def calculate_monoisotopic_mass_from_mz(monoisotopic_mz, charge):
    monoisotopic_mass = (monoisotopic_mz * charge) - (PROTON_MASS * charge)
    return monoisotopic_mass

# determine the voxels included by the raw points
def voxels_for_points(points_df, voxels_df):
    # calculate the intensity contribution of the points to their voxel's intensity
    df = points_df.groupby(['voxel_id'], as_index=False, sort=False).intensity.agg(['sum','count']).reset_index()
    df.rename(columns={'sum':'intensity', 'count':'point_count'}, inplace=True)
    df = pd.merge(df, voxels_df, how='inner', left_on=['voxel_id'], right_on=['voxel_id'], suffixes=['_points','_voxel'])
    df['proportion'] = df.intensity_points / df.intensity_voxel
    # if the points comprise most of a voxel's intensity, we don't need to process that voxel later on
    df = df[(df.proportion >= INTENSITY_PROPORTION_FOR_VOXEL_TO_BE_REMOVED)]
    return set(df.voxel_id.tolist())

# generate a unique feature_id from the precursor id and the feature sequence number found for that precursor
def generate_voxel_id(segment_id, voxel_sequence_number):
    voxel_id = (segment_id * 10000000) + voxel_sequence_number
    return voxel_id

# calculate the r-squared value of series_2 against series_1, where series_1 is the original data (source: https://stackoverflow.com/a/37899817/1184799)
def calculate_r_squared(series_1, series_2):
    residuals = series_1 - series_2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((series_1 - np.mean(series_1))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# measure the R-squared value of the points. x and y are numpy arrays.
def measure_curve(x, y):
    r_squared = None
    warnings.simplefilter("error", OptimizeWarning)
    try:
        # fit a guassian to the points
        guassian_params = peakutils.peak.gaussian_fit(x, y, center_only=False)
        # use the gaussian parameters to calculate the fitted intensities at the given RT values so we can compare
        fitted_intensities = peakutils.peak.gaussian(x, guassian_params[0], guassian_params[1], guassian_params[2])
        # calculate the R-squared of the fit against the observed values
        r_squared = calculate_r_squared(fitted_intensities, y)
    except:
        pass
    return r_squared

# save visualisation data for later analysis of how feature detection works
def save_visualisation(d, segment_id):
    VIS_FILE = '{}/3did-stopping-point-segment-{}.pkl'.format(expanduser("~"), segment_id)
    print("writing stopping point info to {}".format(VIS_FILE))
    with open(VIS_FILE, 'wb') as handle:
        pickle.dump(d, handle)

# process a segment of this run's data, and return a list of features
# @ray.remote
@profile
def find_features(segment_mz_lower, segment_mz_upper, segment_id):
    features_l = []

    # find out where the charge-1 cloud ends and only include points below it (i.e. include points with a higher scan)
    scan_limit = scan_coords_for_single_charge_region(mz_lower=segment_mz_lower, mz_upper=segment_mz_upper)['scan_for_mz_upper']

    # load the raw points for this m/z segment
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    raw_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} and scan >= {} and mz >= {} and mz <= {}".format(FRAME_TYPE_MS1, args.rt_lower, args.rt_upper, scan_limit, segment_mz_lower, segment_mz_upper), db_conn)
    db_conn.close()

    if len(raw_df) > 0:
        # assign each point a unique identifier
        raw_df.reset_index(drop=True, inplace=True)  # just in case
        raw_df['point_id'] = raw_df.index

        # define bins
        rt_bins = pd.interval_range(start=raw_df.retention_time_secs.min(), end=raw_df.retention_time_secs.max()+RT_BIN_SIZE, freq=RT_BIN_SIZE, closed='left')
        scan_bins = pd.interval_range(start=raw_df.scan.min(), end=raw_df.scan.max()+SCAN_BIN_SIZE, freq=SCAN_BIN_SIZE, closed='left')
        mz_bins = pd.interval_range(start=raw_df.mz.min(), end=raw_df.mz.max()+MZ_BIN_SIZE, freq=MZ_BIN_SIZE, closed='left')

        # assign raw points to their bins
        raw_df['rt_bin'] = pd.cut(raw_df.retention_time_secs, bins=rt_bins)
        raw_df['scan_bin'] = pd.cut(raw_df.scan, bins=scan_bins)
        raw_df['mz_bin'] = pd.cut(raw_df.mz, bins=mz_bins)
        raw_df['bin_key'] = raw_df.rt_bin.astype(str) + raw_df.scan_bin.astype(str) + raw_df.mz_bin.astype(str)

        # sum the intensities in each bin
        summary_df = raw_df.groupby(['bin_key'], as_index=False, sort=False).intensity.agg(['sum','count']).reset_index()
        summary_df.rename(columns={'sum':'intensity', 'count':'point_count'}, inplace=True)
        summary_df.dropna(subset=['intensity'], inplace=True)
        summary_df = summary_df[(summary_df.intensity > 0)]
        summary_df.sort_values(by=['intensity'], ascending=False, inplace=True)
        summary_df.reset_index(drop=True, inplace=True)
        summary_df['voxel_id'] = summary_df.index
        summary_df['voxel_id'] = summary_df.apply(lambda row: generate_voxel_id(segment_id, row.voxel_id+1), axis=1)
        print('there are {} voxels for processing in segment {} ({}-{} m/z)'.format(len(summary_df), segment_id, round(segment_mz_lower,1), round(segment_mz_upper,1)))

        # assign each raw point with their voxel ID
        raw_df = pd.merge(raw_df, summary_df[['bin_key','voxel_id']], how='left', left_on=['bin_key'], right_on=['bin_key'])

        # keep track of the keys of voxels that have been processed
        voxels_processed = set()

        # process each voxel by decreasing intensity
        for voxel_idx,voxel in enumerate(summary_df.itertuples()):
            # if this voxel hasn't already been processed...
            if (voxel.voxel_id not in voxels_processed):
                # get the attributes of this voxel
                voxel_mz_lower = voxel.mz_bin.left
                voxel_mz_upper = voxel.mz_bin.right
                voxel_mz_midpoint = voxel.mz_bin.mid
                voxel_scan_lower = voxel.scan_bin.left
                voxel_scan_upper = voxel.scan_bin.right
                voxel_scan_midpoint = voxel.scan_bin.mid
                voxel_rt_lower = voxel.rt_bin.left
                voxel_rt_upper = voxel.rt_bin.right
                voxel_rt_midpoint = voxel.rt_bin.mid
                voxel_df = raw_df[(raw_df.mz >= voxel_mz_lower) & (raw_df.mz <= voxel_mz_upper) & (raw_df.scan >= voxel_scan_lower) & (raw_df.scan <= voxel_scan_upper) & (raw_df.retention_time_secs >= voxel_rt_lower) & (raw_df.retention_time_secs <= voxel_rt_upper)]

                # find the voxel's mz intensity-weighted centroid
                points_a = voxel_df[['mz','intensity']].to_numpy()
                voxel_mz_centroid = intensity_weighted_centroid(points_a[:,1], points_a[:,0])

                # isolate the isotope's points in the m/z dimension; note the isotope may be offset so some of the points may be outside the voxel
                iso_mz_delta = calculate_peak_delta(voxel_mz_centroid)
                iso_mz_lower = voxel_mz_centroid - iso_mz_delta
                iso_mz_upper = voxel_mz_centroid + iso_mz_delta

                # define the peak search area in the mobility dimension
                frame_region_scan_lower = voxel_scan_midpoint - ANCHOR_POINT_SCAN_LOWER_OFFSET
                frame_region_scan_upper = voxel_scan_midpoint + ANCHOR_POINT_SCAN_UPPER_OFFSET

                # record information about the voxel and the isotope we derived from it
                voxel_metadata_d = {'mz_lower':voxel_mz_lower, 'mz_upper':voxel_mz_upper, 'scan_lower':voxel_scan_lower, 'scan_upper':voxel_scan_upper, 'rt_lower':voxel_rt_lower, 'rt_upper':voxel_rt_upper, 'mz_centroid':voxel_mz_centroid, 
                                    'iso_mz_lower':iso_mz_lower, 'iso_mz_upper':iso_mz_upper, 'voxel_scan_midpoint':voxel_scan_midpoint, 'voxel_rt_midpoint':voxel_rt_midpoint, 
                                    'frame_region_scan_lower':frame_region_scan_lower, 'frame_region_scan_upper':frame_region_scan_upper, 'summed_intensity':voxel.intensity, 'point_count':voxel.point_count}

                # find the mobility extent of the isotope in this frame
                isotope_2d_df = raw_df[(raw_df.mz >= iso_mz_lower) & (raw_df.mz <= iso_mz_upper) & (raw_df.scan >= frame_region_scan_lower) & (raw_df.scan <= frame_region_scan_upper) & (raw_df.retention_time_secs >= voxel_rt_lower) & (raw_df.retention_time_secs <= voxel_rt_upper)]
                # collapsing the monoisotopic's summed points onto the mobility dimension
                scan_df = isotope_2d_df.groupby(['scan'], as_index=False).intensity.sum()
                scan_df.sort_values(by=['scan'], ascending=True, inplace=True)
                if len(scan_df) >= MINIMUM_NUMBER_OF_SCANS_IN_BASE_PEAK:

                    # apply a smoothing filter to the points
                    scan_df['filtered_intensity'] = scan_df.intensity  # set the default
                    try:
                        scan_df['filtered_intensity'] = signal.savgol_filter(scan_df.intensity, window_length=find_filter_length(number_of_points=len(scan_df)), polyorder=SCAN_FILTER_POLY_ORDER)
                    except:
                        pass

                    # find the peak(s)
                    peak_x_l = []
                    try:
                        peak_idxs = peakutils.indexes(scan_df.filtered_intensity.values, thres=PEAKS_THRESHOLD_SCAN, min_dist=PEAKS_MIN_DIST_SCAN, thres_abs=False)
                        peak_x_l = scan_df.iloc[peak_idxs].scan.to_list()
                    except:
                        pass
                    if len(peak_x_l) == 0:
                        # if we couldn't find any peaks, take the maximum intensity point
                        peak_x_l = [scan_df.loc[scan_df.filtered_intensity.idxmax()].scan]
                    # peaks_df should now contain the rows from flattened_points_df that represent the peaks
                    peaks_df = scan_df[scan_df.scan.isin(peak_x_l)].copy()

                    # find the closest peak to the voxel highpoint
                    peaks_df['delta'] = abs(peaks_df.scan - voxel_scan_midpoint)
                    peaks_df.sort_values(by=['delta'], ascending=True, inplace=True)
                    scan_apex = peaks_df.iloc[0].scan

                    # find the valleys nearest the scan apex
                    valley_idxs = peakutils.indexes(-scan_df.filtered_intensity.values, thres=VALLEYS_THRESHOLD_SCAN, min_dist=VALLEYS_MIN_DIST_SCAN, thres_abs=False)
                    valley_x_l = scan_df.iloc[valley_idxs].scan.to_list()
                    valleys_df = scan_df[scan_df.scan.isin(valley_x_l)]

                    upper_x = valleys_df[valleys_df.scan > scan_apex].scan.min()
                    if math.isnan(upper_x):
                        upper_x = scan_apex + (SCAN_BASE_PEAK_WIDTH / 2)
                    lower_x = valleys_df[valleys_df.scan < scan_apex].scan.max()
                    if math.isnan(lower_x):
                        lower_x = scan_apex - (SCAN_BASE_PEAK_WIDTH / 2)

                    # mobility extent of the isotope
                    iso_scan_lower = lower_x
                    iso_scan_upper = upper_x

                    # gather the isotope points constrained by m/z and CCS, and the peak search extent in RT
                    region_rt_lower = voxel_rt_midpoint - (RT_BASE_PEAK_WIDTH * 2)  # need to look quite wide so we catch the high-intensity peaks
                    region_rt_upper = voxel_rt_midpoint + (RT_BASE_PEAK_WIDTH * 2)
                    isotope_points_df = raw_df[(raw_df.mz >= iso_mz_lower) & (raw_df.mz <= iso_mz_upper) & (raw_df.scan >= iso_scan_lower) & (raw_df.scan <= iso_scan_upper) & (raw_df.retention_time_secs >= region_rt_lower) & (raw_df.retention_time_secs <= region_rt_upper)]

                    # in the RT dimension, find the apex
                    rt_df = isotope_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
                    rt_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)

                    # filter the points
                    rt_df['filtered_intensity'] = rt_df.intensity  # set the default
                    try:
                        rt_df['filtered_intensity'] = signal.savgol_filter(rt_df.intensity, window_length=find_filter_length(number_of_points=len(rt_df)), polyorder=RT_FILTER_POLY_ORDER)
                    except:
                        pass

                    # find the peak(s)
                    peak_x_l = []
                    try:
                        peak_idxs = peakutils.indexes(rt_df.filtered_intensity.values, thres=PEAKS_THRESHOLD_RT, min_dist=PEAKS_MIN_DIST_RT, thres_abs=False)
                        peak_x_l = rt_df.iloc[peak_idxs].retention_time_secs.to_list()
                    except:
                        pass
                    if len(peak_x_l) == 0:
                        # if we couldn't find any peaks, take the maximum intensity point
                        peak_x_l = [rt_df.loc[rt_df.filtered_intensity.idxmax()].retention_time_secs]
                    # peaks_df should now contain the rows from flattened_points_df that represent the peaks
                    peaks_df = rt_df[rt_df.retention_time_secs.isin(peak_x_l)].copy()

                    # find the closest peak to the voxel highpoint
                    peaks_df['delta'] = abs(peaks_df.retention_time_secs - voxel_rt_midpoint)
                    peaks_df.sort_values(by=['delta'], ascending=True, inplace=True)
                    rt_apex = peaks_df.iloc[0].retention_time_secs

                    # find the valleys nearest the RT apex
                    valley_idxs = peakutils.indexes(-rt_df.filtered_intensity.values, thres=VALLEYS_THRESHOLD_RT, min_dist=VALLEYS_MIN_DIST_RT, thres_abs=False)
                    valley_x_l = rt_df.iloc[valley_idxs].retention_time_secs.to_list()
                    valleys_df = rt_df[rt_df.retention_time_secs.isin(valley_x_l)]

                    upper_x = valleys_df[valleys_df.retention_time_secs > rt_apex].retention_time_secs.min()
                    if math.isnan(upper_x):
                        upper_x = rt_apex + (RT_BASE_PEAK_WIDTH / 2)
                    lower_x = valleys_df[valleys_df.retention_time_secs < rt_apex].retention_time_secs.max()
                    if math.isnan(lower_x):
                        lower_x = rt_apex - (RT_BASE_PEAK_WIDTH / 2)

                    # RT extent of the isotope
                    iso_rt_lower = lower_x
                    iso_rt_upper = upper_x

                    # clip the filtered intensity to zero
                    scan_df['clipped_filtered_intensity'] = scan_df['filtered_intensity'].clip(lower=1, inplace=False)
                    rt_df['clipped_filtered_intensity'] = rt_df['filtered_intensity'].clip(lower=1, inplace=False)

                    # constrain the summed CCS and RT curves to the peak of interest
                    scan_subset_df = scan_df[(scan_df.scan >= iso_scan_lower) & (scan_df.scan <= iso_scan_upper)]
                    rt_subset_df = rt_df[(rt_df.retention_time_secs >= iso_rt_lower) & (rt_df.retention_time_secs <= iso_rt_upper)]

                    # remove gaps on the edges of the RT peak
                    gaps_rt_df = rt_subset_df.copy()
                    gaps_rt_df['rt_point_delta'] = gaps_rt_df.retention_time_secs.diff()
                    gaps_rt_df['rt_apex_delta'] = abs(gaps_rt_df.retention_time_secs - rt_apex)
                    gaps_rt_df.sort_values(by=['rt_apex_delta'], ascending=True, inplace=True, ignore_index=True)

                    # adjust the lower bound if there are gaps
                    lower_gaps_df = gaps_rt_df[(gaps_rt_df.retention_time_secs < rt_apex) & (gaps_rt_df.rt_point_delta > MAXIMUM_GAP_SECS_BETWEEN_EDGE_POINTS)]
                    if len(lower_gaps_df) > 0:
                        iso_rt_lower = lower_gaps_df.retention_time_secs.max()
                        rt_subset_df = rt_df[(rt_df.retention_time_secs >= iso_rt_lower) & (rt_df.retention_time_secs <= iso_rt_upper)]  # reset the subset to the new bounds

                    # adjust the upper bound if there are gaps
                    upper_gaps_df = gaps_rt_df[(gaps_rt_df.retention_time_secs > rt_apex) & (gaps_rt_df.rt_point_delta > MAXIMUM_GAP_SECS_BETWEEN_EDGE_POINTS)]
                    if len(upper_gaps_df) > 0:
                        iso_rt_upper = upper_gaps_df.retention_time_secs.min()
                        rt_subset_df = rt_df[(rt_df.retention_time_secs >= iso_rt_lower) & (rt_df.retention_time_secs <= iso_rt_upper)]  # reset the subset to the new bounds

                    # check the base peak has at least one voxel in common with the seeding voxel
                    base_peak_df = raw_df[(raw_df.mz >= iso_mz_lower) & (raw_df.mz <= iso_mz_upper) & (raw_df.scan >= iso_scan_lower) & (raw_df.scan <= iso_scan_upper) & (raw_df.retention_time_secs >= iso_rt_lower) & (raw_df.retention_time_secs <= iso_rt_upper)].copy()
                    if voxel.voxel_id in base_peak_df.voxel_id.unique():

                        # calculate the R-squared
                        scan_r_squared = measure_curve(x=scan_subset_df.scan.to_numpy(), y=scan_subset_df.clipped_filtered_intensity.to_numpy())
                        rt_r_squared = measure_curve(x=rt_subset_df.retention_time_secs.to_numpy(), y=rt_subset_df.clipped_filtered_intensity.to_numpy())

                        # if the base isotope is sufficiently gaussian in at least one of the dimensions, it's worth processing
                        if (((scan_r_squared is not None) and (scan_r_squared >= MINIMUM_R_SQUARED)) or ((rt_r_squared is not None) and (rt_r_squared >= MINIMUM_R_SQUARED))):

                            # we now have a definition of the voxel's isotope in m/z, scan, and RT. We need to extend that in the m/z dimension to catch all the isotopes for this feature
                            region_mz_lower = voxel_mz_midpoint - ANCHOR_POINT_MZ_LOWER_OFFSET
                            region_mz_upper = voxel_mz_midpoint + ANCHOR_POINT_MZ_UPPER_OFFSET

                            # gather the raw points for the feature's 3D region (i.e. the region in which deconvolution will be performed)
                            feature_region_3d_extent_d = {'mz_lower':region_mz_lower, 'mz_upper':region_mz_upper, 'scan_lower':iso_scan_lower, 'scan_upper':iso_scan_upper, 'rt_lower':iso_rt_lower, 'rt_upper':iso_rt_upper}
                            feature_region_3d_df = raw_df[(raw_df.mz >= region_mz_lower) & (raw_df.mz <= region_mz_upper) & (raw_df.scan >= iso_scan_lower) & (raw_df.scan <= iso_scan_upper) & (raw_df.retention_time_secs >= iso_rt_lower) & (raw_df.retention_time_secs <= iso_rt_upper)].copy()

                            # intensity descent
                            raw_points_a = feature_region_3d_df[['mz','intensity']].to_numpy()
                            peaks_a = intensity_descent(peaks_a=raw_points_a, peak_delta=None)

                            # deconvolution - see https://mobiusklein.github.io/ms_deisotope/docs/_build/html/deconvolution/deconvolution.html
                            ms1_peaks_l = list(map(tuple, peaks_a))
                            deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, use_quick_charge=True, averagine=averagine.peptide, truncate_after=0.95)

                            # collect features from deconvolution
                            ms1_deconvoluted_peaks_l = []
                            for peak_idx,peak in enumerate(deconvoluted_peaks):
                                # discard a monoisotopic peak that has either of the first two peaks as placeholders (indicated by intensity of 1)
                                if ((len(peak.envelope) >= 3) and (peak.envelope[0][1] > 1) and (peak.envelope[1][1] > 1)):
                                    mono_peak_mz = peak.mz
                                    mono_intensity = peak.intensity
                                    second_peak_mz = peak.envelope[1][0]
                                    # an accepted feature must have its mono peak or base peak aligned with the voxel
                                    if (((mono_peak_mz >= iso_mz_lower) and (mono_peak_mz <= iso_mz_upper)) or ((second_peak_mz >= iso_mz_lower) and (second_peak_mz <= iso_mz_upper))):
                                        ms1_deconvoluted_peaks_l.append((mono_peak_mz, second_peak_mz, mono_intensity, peak.score, peak.signal_to_noise, peak.charge, peak.envelope, peak.neutral_mass))
                            deconvolution_features_df = pd.DataFrame(ms1_deconvoluted_peaks_l, columns=['mono_mz','second_peak_mz','intensity','score','SN','charge','envelope','neutral_mass'])
                            deconvolution_features_df.sort_values(by=['score'], ascending=False, inplace=True)

                            if len(deconvolution_features_df) > 0:
                                # determine the feature attributes
                                for idx,feature in enumerate(deconvolution_features_df.itertuples()):
                                    feature_d = {}
                                    envelope_mono_mz = feature.envelope[0][0]
                                    mz_delta = calculate_peak_delta(mz=envelope_mono_mz)
                                    mono_mz_lower = envelope_mono_mz - mz_delta
                                    mono_mz_upper = envelope_mono_mz + mz_delta
                                    feature_d['mono_mz_lower'] = mono_mz_lower
                                    feature_d['mono_mz_upper'] = mono_mz_upper

                                    feature_d['scan_apex'] = scan_apex
                                    feature_d['scan_lower'] = iso_scan_lower
                                    feature_d['scan_upper'] = iso_scan_upper

                                    feature_d['rt_apex'] = rt_apex
                                    feature_d['rt_lower'] = iso_rt_lower
                                    feature_d['rt_upper'] = iso_rt_upper

                                    isotope_characteristics_d = determine_isotope_characteristics(envelope=feature.envelope, rt_apex=rt_apex, monoisotopic_mass=feature.neutral_mass, feature_region_3d_df=feature_region_3d_df, summary_df=summary_df)
                                    if isotope_characteristics_d is not None:
                                        # add the characteristics to the feature dictionary
                                        feature_d = {**feature_d, **isotope_characteristics_d}
                                        feature_d['monoisotopic_mz'] = isotope_characteristics_d['mono_mz']
                                        feature_d['charge'] = feature.charge
                                        feature_d['monoisotopic_mass'] = calculate_monoisotopic_mass_from_mz(monoisotopic_mz=feature_d['monoisotopic_mz'], charge=feature_d['charge'])
                                        feature_d['feature_intensity'] = isotope_characteristics_d['intensity_with_saturation_correction'] if (isotope_characteristics_d['intensity_with_saturation_correction'] > isotope_characteristics_d['intensity_without_saturation_correction']) else isotope_characteristics_d['intensity_without_saturation_correction']
                                        feature_d['envelope'] = json.dumps([tuple(e) for e in feature.envelope])
                                        feature_d['isotope_count'] = len(feature.envelope)
                                        feature_d['deconvolution_score'] = feature.score
                                        # record the feature region where we found this feature
                                        feature_d['feature_region_3d_extent'] = feature_region_3d_extent_d
                                        # record the voxel from where we derived the initial isotope
                                        feature_d['voxel_id'] = voxel.voxel_id
                                        feature_d['voxel_metadata_d'] = voxel_metadata_d
                                        feature_d['scan_df'] = scan_df.to_dict('records')
                                        feature_d['scan_r_squared'] = scan_r_squared
                                        feature_d['rt_df'] = rt_df.to_dict('records')
                                        feature_d['rt_r_squared'] = rt_r_squared
                                        # add it to the list
                                        features_l.append(feature_d)

                                        # add the voxels included in the feature's isotopes to the list of voxels already processed
                                        voxels_processed.update(feature_d['voxels_processed'])

                        else:
                            # if the unviable feature is close to the RT edge, we should keep going
                            if (iso_rt_lower > (args.rt_lower + 1.0)) and (iso_rt_upper < (args.rt_upper - 1.0)):
                                print('the base isotope is insufficiently gaussian in the CCS and RT dimensions, so we\'ll stop here.')
                                if scan_r_squared is not None:
                                    print('scan: {}'.format(round(scan_r_squared,1)))
                                else:
                                    print('could not fit a curve in CCS')

                                if rt_r_squared is not None:
                                    print('rt: {}'.format(round(rt_r_squared,1)))
                                else:
                                    print('could not fit a curve in RT')

                                # save information about the stopping point for analysis
                                region_mz_lower = voxel_mz_midpoint - ANCHOR_POINT_MZ_LOWER_OFFSET
                                region_mz_upper = voxel_mz_midpoint + ANCHOR_POINT_MZ_UPPER_OFFSET
                                feature_region_3d_extent_d = {'mz_lower':region_mz_lower, 'mz_upper':region_mz_upper, 'scan_lower':iso_scan_lower, 'scan_upper':iso_scan_upper, 'rt_lower':iso_rt_lower, 'rt_upper':iso_rt_upper}

                                d = {}
                                d['voxel_id'] = voxel.voxel_id
                                d['scan_df'] = scan_df.to_dict('records')
                                d['rt_df'] = rt_df.to_dict('records')
                                d['voxel_metadata_d'] = voxel_metadata_d
                                d['feature_region_3d_extent'] = feature_region_3d_extent_d
                                d['scan_r_squared'] = scan_r_squared
                                d['rt_r_squared'] = rt_r_squared
                                d['scan_apex'] = scan_apex
                                d['scan_lower'] = iso_scan_lower
                                d['scan_upper'] = iso_scan_upper
                                d['rt_apex'] = rt_apex
                                d['rt_lower'] = iso_rt_lower
                                d['rt_upper'] = iso_rt_upper
                                save_visualisation(d, segment_id)

                                break

    features_df = pd.DataFrame(features_l)
    return features_df


# move these constants to the INI file
ANCHOR_POINT_MZ_LOWER_OFFSET = 0.6   # one isotope for charge-2 plus a little bit more
ANCHOR_POINT_MZ_UPPER_OFFSET = 3.0   # six isotopes for charge-2 plus a little bit more

ANCHOR_POINT_SCAN_LOWER_OFFSET = 40  # twice the base peak width
ANCHOR_POINT_SCAN_UPPER_OFFSET = 40

# peak and valley detection parameters
PEAKS_THRESHOLD_RT = 0.5    # only consider peaks that are higher than this proportion of the normalised maximum
PEAKS_THRESHOLD_SCAN = 0.5
PEAKS_MIN_DIST_RT = 2.0     # seconds
PEAKS_MIN_DIST_SCAN = 10.0  # scans

VALLEYS_THRESHOLD_RT = 0.25    # only consider valleys that drop more than this proportion of the normalised maximum
VALLEYS_THRESHOLD_SCAN = 0.25
VALLEYS_MIN_DIST_RT = 2.0     # seconds
VALLEYS_MIN_DIST_SCAN = 10.0  # scans

# filter parameters
SCAN_FILTER_POLY_ORDER = 5
RT_FILTER_POLY_ORDER = 3

# bin sizes; RT and CCS are determined from half the mean base peak width
RT_BIN_SIZE = 5
SCAN_BIN_SIZE = 10
MZ_BIN_SIZE = 0.1

MINIMUM_NUMBER_OF_SCANS_IN_BASE_PEAK = 5
MINIMUM_R_SQUARED = 0.0  # for the curves fitted in the RT and CCS dimensions
MAXIMUM_GAP_SECS_BETWEEN_EDGE_POINTS = 1.0
INTENSITY_PROPORTION_FOR_VOXEL_TO_BE_REMOVED = 0.8


#######################
parser = argparse.ArgumentParser(description='Find all the features in a run with 3D intensity descent.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-ml','--mz_lower', type=int, default='100', help='Lower limit for m/z.', required=False)
parser.add_argument('-mu','--mz_upper', type=int, default='1700', help='Upper limit for m/z.', required=False)
parser.add_argument('-mw','--mz_width_per_segment', type=int, default=20, help='Width in Da of the m/z processing window per segment.', required=False)
parser.add_argument('-rl','--rt_lower', type=int, default='1650', help='Lower limit for retention time.', required=False)
parser.add_argument('-ru','--rt_upper', type=int, default='2200', help='Upper limit for retention time.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
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

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
FRAME_TYPE_MS1 = cfg.getint('common','FRAME_TYPE_MS1')
MS1_PEAK_DELTA = cfg.getfloat('ms1','MS1_PEAK_DELTA')
RT_BASE_PEAK_WIDTH = cfg.getfloat('common','RT_BASE_PEAK_WIDTH_SECS')
SCAN_BASE_PEAK_WIDTH = cfg.getfloat('common','SCAN_BASE_PEAK_WIDTH')
CARBON_MASS_DIFFERENCE = cfg.getfloat('common','CARBON_MASS_DIFFERENCE')
PROTON_MASS = cfg.getfloat('common', 'PROTON_MASS')
INSTRUMENT_RESOLUTION = cfg.getfloat('common', 'INSTRUMENT_RESOLUTION')
SATURATION_INTENSITY = cfg.getint('common', 'SATURATION_INTENSITY')
TARGET_NUMBER_OF_FEATURES_FOR_CUBOID = cfg.getint('ms1', 'TARGET_NUMBER_OF_FEATURES_FOR_CUBOID')


# set up the indexes
print('setting up indexes on {}'.format(CONVERTED_DATABASE_NAME))
create_indexes(db_file_name=CONVERTED_DATABASE_NAME)

# output features
FEATURES_DIR = "{}/features-3did".format(EXPERIMENT_DIR)
FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name)
# set up the output directory
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)

# set up Ray
print("setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

# calculate the segments
mz_range = args.mz_upper - args.mz_lower
NUMBER_OF_MZ_SEGMENTS = (mz_range // args.mz_width_per_segment) + (mz_range % args.mz_width_per_segment > 0)  # thanks to https://stackoverflow.com/a/23590097/1184799

# find all the features
print('finding features')
# features_l = ray.get([find_features.remote(segment_mz_lower=args.mz_lower+(i*args.mz_width_per_segment), segment_mz_upper=args.mz_lower+(i*args.mz_width_per_segment)+args.mz_width_per_segment, segment_id=i+1) for i in range(NUMBER_OF_MZ_SEGMENTS)])
features_l = [find_features(segment_mz_lower=args.mz_lower+(i*args.mz_width_per_segment), segment_mz_upper=args.mz_lower+(i*args.mz_width_per_segment)+args.mz_width_per_segment, segment_id=i+1) for i in range(NUMBER_OF_MZ_SEGMENTS)]

# join the list of dataframes into a single dataframe
features_df = pd.concat(features_l, axis=0, sort=False, ignore_index=True)

# assign each feature a unique identifier
features_df['feature_id'] = features_df.index

# ... and save them in a file
print()
print('saving {} features to {}'.format(len(features_df), FEATURES_FILE))
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
content_d = {'features_df':features_df, 'metadata':info}
with open(FEATURES_FILE, 'wb') as handle:
    pickle.dump(content_d, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
