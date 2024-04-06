import pandas as pd
import numpy as np
import peakutils
from scipy import signal
import math
import os
import time
import argparse
import ray
import multiprocessing as mp
import sys
import configparser
from configparser import ExtendedInterpolation
import json
from ms_deisotope import deconvolute_peaks, averagine
import warnings
from scipy.optimize import OptimizeWarning
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import alphatims.bruker


# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers

# determine the maximum filter length for the number of points
def find_filter_length(number_of_points):
    filter_lengths = [51,11,5]  # must be a positive odd number, greater than the polynomial order, and less than the number of points to be filtered
    return filter_lengths[next(x[0] for x in enumerate(filter_lengths) if x[1] < number_of_points)]

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

# calculate the cosine similarity of two peaks; each DF is assumed to have an 'x' column that reflects the x-axis values, and an 'intensity' column
def measure_peak_similarity(isotopeA_df, isotopeB_df, x_label, scale):
    # scale the x axis so we can join them
    isotopeA_df['x_scaled'] = (isotopeA_df[x_label] * scale).astype(int)
    isotopeB_df['x_scaled'] = (isotopeB_df[x_label] * scale).astype(int)
    # combine the isotopes by aligning the x-dimension points they have in common
    combined_df = pd.merge(isotopeA_df, isotopeB_df, on='x_scaled', how='inner', suffixes=('_A', '_B')).sort_values(by='x_scaled')
    combined_df = combined_df[['x_scaled','intensity_A','intensity_B']]
    # calculate the similarity
    return float(cosine_similarity([combined_df.intensity_A.values], [combined_df.intensity_B.values])) if len(combined_df) > 0 else 0.0

# calculate the characteristics of the isotopes in the feature envelope
def determine_isotope_characteristics(envelope, rt_apex, monoisotopic_mass, feature_region_3d_df):
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
            points_voxels = list(isotope_df.voxel_id.unique())
            # record the voxels included by this isotope
            voxel_ids_for_isotope = voxels_for_points(points_df=isotope_df)
            # add the voxels included in the feature's points to the list of voxels already processed
            voxels_processed.update(voxel_ids_for_isotope)
            # find the intensity by summing the maximum point in the frame closest to the RT apex, and the frame maximums either side
            frame_maximums_df = isotope_df.groupby(['retention_time_secs'], as_index=False, sort=False).intensity.agg(['max']).reset_index()
            frame_maximums_df['rt_delta'] = np.abs(frame_maximums_df.retention_time_secs - rt_apex)
            frame_maximums_df.sort_values(by=['rt_delta'], ascending=True, inplace=True)
            # sum the maximum intensity and the max intensity of the frame either side in RT
            summed_intensity = frame_maximums_df[:3]['max'].sum()
            # are any of the three points in saturation?
            isotope_in_saturation = (frame_maximums_df[:3]['max'].max() > SATURATION_INTENSITY)
            # determine the isotope's profile in retention time
            rt_df = isotope_df.groupby(['retention_time_secs'], as_index=False).intensity.sum()
            rt_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)
            # measure it's elution similarity with the previous isotope
            similarity_rt = measure_peak_similarity(pd.DataFrame(isotopes_l[idx-1]['rt_df']), rt_df, x_label='retention_time_secs', scale=100) if idx > 0 else None
            # determine the isotope's profile in mobility
            scan_df = isotope_df.groupby(['scan'], as_index=False).intensity.sum()
            scan_df.sort_values(by=['scan'], ascending=True, inplace=True)
            # measure it's elution similarity with the previous isotope
            similarity_scan = measure_peak_similarity(pd.DataFrame(isotopes_l[idx-1]['scan_df']), scan_df, x_label='scan', scale=1) if idx > 0 else None
            if (idx == 0) or ((idx > 0) and (similarity_rt >= ISOTOPE_SIMILARITY_RT_THRESHOLD) and (similarity_scan >= ISOTOPE_SIMILARITY_CCS_THRESHOLD)):
                # add the isotope to the list
                isotopes_l.append({'mz':iso_mz, 'mz_lower':iso_mz_lower, 'mz_upper':iso_mz_upper, 'intensity':summed_intensity, 'saturated':isotope_in_saturation, 'rt_df':rt_df.to_dict('records'), 'scan_df':scan_df.to_dict('records'), 'similarity_rt':similarity_rt, 'similarity_scan':similarity_scan, 'points_voxels':points_voxels, 'voxel_ids_for_isotope':voxel_ids_for_isotope})
            else:
                break
        else:
            break
    isotopes_df = pd.DataFrame(isotopes_l)

    # calculate the coelution coefficient for the isotopic peak series
    coelution_coefficient = isotopes_df.similarity_rt.mean()
    mobility_coefficient = isotopes_df.similarity_scan.mean()

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
    result_d['isotopic_peaks'] = isotopes_df.to_json(orient='records')
    result_d['isotope_count'] = len(isotopes_df)
    result_d['envelope'] = json.dumps([tuple(e) for e in envelope[:result_d['isotope_count']]])  # modify the envelope according to how many similar isotopes we found
    result_d['coelution_coefficient'] = coelution_coefficient
    result_d['mobility_coefficient'] = mobility_coefficient
    result_d['voxels_processed'] = voxels_processed
    return result_d

# calculate the monoisotopic mass    
def calculate_monoisotopic_mass_from_mz(monoisotopic_mz, charge):
    monoisotopic_mass = (monoisotopic_mz * charge) - (PROTON_MASS * charge)
    return monoisotopic_mass

# determine the voxels included by the raw points
def voxels_for_points(points_df):
    # calculate the intensity contribution of the points to their voxel's intensity
    df = points_df.groupby(['voxel_id'], as_index=False, sort=False).voxel_proportion.agg(['sum']).reset_index()
    # if the points comprise most of a voxel's intensity, we don't need to process that voxel later on
    df = df[(df['sum'] >= INTENSITY_PROPORTION_FOR_VOXEL_TO_BE_REMOVED)]
    return set(df.voxel_id.astype(int).tolist())

# generate a unique feature_id from the precursor id and the feature sequence number found for that precursor
def generate_voxel_id(segment_id, voxel_sequence_number):
    voxel_id = (segment_id * 10000000) + voxel_sequence_number
    return voxel_id

# calculate the r-squared value of series_2 against series_1, where series_1 is the original data (source: https://stackoverflow.com/a/37899817/1184799)
def calculate_r_squared(series_1, series_2):
    residuals = series_1 - series_2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((series_1 - np.mean(series_1))**2)
    if ss_tot == 0:
        r_squared = 0
    else:
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

# process a segment of this run's data, and return a list of features
@ray.remote
def find_features(segment_d):
    # segment_df = pd.read_pickle(segment_d['segment_name'])
    segment_df = segment_d['segment_df'].copy()
    segment_id = segment_d['segment_id']
    features_l = []
    if len(segment_df) > 0:
        # assign each point a unique identifier
        segment_df.reset_index(drop=True, inplace=True)  # just in case
        segment_df['point_id'] = segment_df.index

        # define bins
        rt_bins = pd.interval_range(start=segment_df.retention_time_secs.min(), end=segment_df.retention_time_secs.max()+(VOXEL_SIZE_RT*2), freq=VOXEL_SIZE_RT, closed='left')
        scan_bins = pd.interval_range(start=segment_df.scan.min(), end=segment_df.scan.max()+(VOXEL_SIZE_SCAN*2), freq=VOXEL_SIZE_SCAN, closed='left')
        mz_bins = pd.interval_range(start=segment_d['mz_lower'], end=segment_d['mz_upper']+SEGMENT_EXTENSION+(VOXEL_SIZE_MZ*2), freq=VOXEL_SIZE_MZ, closed='left')

        # assign raw points to their bins
        segment_df['rt_bin'] = pd.cut(segment_df.retention_time_secs, bins=rt_bins)
        segment_df['scan_bin'] = pd.cut(segment_df.scan, bins=scan_bins)
        segment_df['mz_bin'] = pd.cut(segment_df.mz, bins=mz_bins)
        segment_df['bin_key'] = list(zip(segment_df.mz_bin, segment_df.scan_bin, segment_df.rt_bin))

        # sum the intensities in each bin
        summary_df = segment_df.groupby(['bin_key'], as_index=False, sort=False).intensity.agg(['sum','count','mean']).reset_index()
        summary_df['extension_zone'] = summary_df.apply(lambda row: row.bin_key[0].mid > segment_d['mz_upper'], axis=1)  # identify which voxels are in the extension zone
        summary_df = summary_df[(summary_df.extension_zone == False)]                                               # and remove them from the summary
        summary_df.rename(columns={'sum':'voxel_intensity', 'count':'point_count', 'mean':'voxel_mean'}, inplace=True)
        summary_df.dropna(subset=['voxel_intensity'], inplace=True)
        summary_df.dropna(subset=['voxel_mean'], inplace=True)

        if len(summary_df) > 0:
            summary_df.sort_values(by=['voxel_intensity'], ascending=False, inplace=True)
            summary_df.reset_index(drop=True, inplace=True)
            summary_df['voxel_id'] = summary_df.index
            summary_df['voxel_id'] = summary_df.apply(lambda row: generate_voxel_id(segment_id, row.voxel_id+1), axis=1)
            summary_df_name = '{}/summary-{}-{}.pkl'.format(SUMMARY_DIR, round(segment_d['mz_lower']), round(segment_d['mz_upper']))
            summary_df.to_pickle(summary_df_name)

            # assign each raw point with their voxel ID
            segment_df = pd.merge(segment_df, summary_df[['bin_key','voxel_id','voxel_intensity']], how='left', left_on=['bin_key'], right_on=['bin_key'])

            # determine each point\'s contribution to its voxel intensity
            segment_df['voxel_proportion'] = segment_df.intensity / segment_df.voxel_intensity

            # keep track of the keys of voxels that have been processed
            voxels_processed = set()

            # process each voxel by decreasing intensity
            base_peak_voxels_df = summary_df[(summary_df.voxel_intensity >= args.minimum_voxel_intensity)]
            print('there are {} voxels for processing in segment {} ({}-{} m/z)'.format(len(base_peak_voxels_df), segment_d['segment_id'], round(segment_d['mz_lower']), round(segment_d['mz_upper'])))
            for voxel_idx,voxel in enumerate(base_peak_voxels_df.itertuples()):
                # if this voxel hasn't already been processed...
                if (voxel.voxel_id not in voxels_processed):
                    # retrieve the bins from the voxel key
                    (mz_bin, scan_bin, rt_bin) = voxel.bin_key

                    # get the attributes of this voxel
                    voxel_mz_lower = mz_bin.left
                    voxel_mz_upper = mz_bin.right
                    voxel_mz_midpoint = mz_bin.mid
                    voxel_scan_lower = scan_bin.left
                    voxel_scan_upper = scan_bin.right
                    voxel_scan_midpoint = scan_bin.mid
                    voxel_rt_lower = rt_bin.left
                    voxel_rt_upper = rt_bin.right
                    voxel_rt_midpoint = rt_bin.mid
                    voxel_rt_condition = (segment_df.retention_time_secs >= voxel_rt_lower) & (segment_df.retention_time_secs <= voxel_rt_upper)
                    voxel_points_df = segment_df[(segment_df.mz >= voxel_mz_lower) & (segment_df.mz <= voxel_mz_upper) & (segment_df.scan >= voxel_scan_lower) & (segment_df.scan <= voxel_scan_upper) & voxel_rt_condition]

                    # find the voxel's mz intensity-weighted centroid
                    points_a = voxel_points_df[['mz','intensity']].to_numpy()
                    voxel_mz_centroid = intensity_weighted_centroid(points_a[:,1], points_a[:,0])

                    # isolate the isotope's points in the m/z dimension; note the isotope may be offset so some of the points may be outside the voxel
                    iso_mz_delta = calculate_peak_delta(voxel_mz_centroid)
                    iso_mz_lower = voxel_mz_centroid - iso_mz_delta
                    iso_mz_upper = voxel_mz_centroid + iso_mz_delta

                    # define the peak search area in the mobility dimension
                    frame_region_scan_lower = voxel_scan_midpoint - ANCHOR_POINT_SCAN_LOWER_OFFSET
                    frame_region_scan_upper = voxel_scan_midpoint + ANCHOR_POINT_SCAN_UPPER_OFFSET

                    # record information about the voxel and the isotope we derived from it
                    voxel_metadata_d = {'mz_lower':voxel_mz_lower, 'mz_upper':voxel_mz_upper, 'scan_lower':int(voxel_scan_lower), 'scan_upper':int(voxel_scan_upper), 'rt_lower':voxel_rt_lower, 'rt_upper':voxel_rt_upper, 'mz_centroid':voxel_mz_centroid, 
                                        'iso_mz_lower':iso_mz_lower, 'iso_mz_upper':iso_mz_upper, 'voxel_scan_midpoint':int(voxel_scan_midpoint), 'voxel_rt_midpoint':voxel_rt_midpoint, 
                                        'frame_region_scan_lower':int(frame_region_scan_lower), 'frame_region_scan_upper':int(frame_region_scan_upper), 'summed_intensity':int(voxel.voxel_intensity), 'point_count':int(voxel.point_count)}

                    # find the mobility extent of the isotope in this frame
                    iso_mz_condition = (segment_df.mz >= iso_mz_lower) & (segment_df.mz <= iso_mz_upper)
                    isotope_2d_df = segment_df[iso_mz_condition & (segment_df.scan >= frame_region_scan_lower) & (segment_df.scan <= frame_region_scan_upper) & voxel_rt_condition]
                    # collapsing the monoisotopic's summed points onto the mobility dimension
                    scan_df = isotope_2d_df.groupby(['scan','inverse_k0'], as_index=False).intensity.sum()
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
                            peak_idxs = peakutils.indexes(scan_df.filtered_intensity.values.astype(int), thres=PEAKS_THRESHOLD_SCAN, min_dist=PEAKS_MIN_DIST_SCAN, thres_abs=False)
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
                        inverse_k0_apex = peaks_df.iloc[0].inverse_k0

                        # find the valleys nearest the scan apex
                        valley_idxs = peakutils.indexes(-scan_df.filtered_intensity.values.astype(int), thres=VALLEYS_THRESHOLD_SCAN, min_dist=VALLEYS_MIN_DIST_SCAN, thres_abs=False)
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
                        region_rt_lower = voxel_rt_lower - RT_BASE_PEAK_WIDTH
                        region_rt_upper = voxel_rt_upper + RT_BASE_PEAK_WIDTH
                        iso_scan_condition = (segment_df.scan >= iso_scan_lower) & (segment_df.scan <= iso_scan_upper)
                        isotope_points_df = segment_df[iso_mz_condition & iso_scan_condition & (segment_df.retention_time_secs >= region_rt_lower) & (segment_df.retention_time_secs <= region_rt_upper)]

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
                            peak_idxs = peakutils.indexes(rt_df.filtered_intensity.values.astype(int), thres=PEAKS_THRESHOLD_RT, min_dist=PEAKS_MIN_DIST_RT, thres_abs=False)
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
                        valley_idxs = peakutils.indexes(-rt_df.filtered_intensity.values.astype(int), thres=VALLEYS_THRESHOLD_RT, min_dist=VALLEYS_MIN_DIST_RT, thres_abs=False)
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
                        iso_rt_condition = (segment_df.retention_time_secs >= iso_rt_lower) & (segment_df.retention_time_secs <= iso_rt_upper)
                        base_peak_df = segment_df[iso_mz_condition & iso_scan_condition & iso_rt_condition]
                        if voxel.voxel_id in base_peak_df.voxel_id.unique():

                            # calculate the R-squared
                            scan_r_squared = measure_curve(x=scan_subset_df.scan.to_numpy(), y=scan_subset_df.clipped_filtered_intensity.to_numpy())
                            rt_r_squared = measure_curve(x=rt_subset_df.retention_time_secs.to_numpy(), y=rt_subset_df.clipped_filtered_intensity.to_numpy())

                            # we now have a definition of the voxel's isotope in m/z, scan, and RT. We need to extend that in the m/z dimension to catch all the isotopes for this feature
                            region_mz_lower = voxel_mz_midpoint - ANCHOR_POINT_MZ_LOWER_OFFSET
                            region_mz_upper = voxel_mz_midpoint + ANCHOR_POINT_MZ_UPPER_OFFSET

                            # gather the raw points for the feature's 3D region (i.e. the region in which deconvolution will be performed)
                            feature_region_3d_extent_d = {'mz_lower':region_mz_lower, 'mz_upper':region_mz_upper, 'scan_lower':int(iso_scan_lower), 'scan_upper':int(iso_scan_upper), 'rt_lower':iso_rt_lower, 'rt_upper':iso_rt_upper}
                            feature_region_3d_df = segment_df[(segment_df.mz >= region_mz_lower) & (segment_df.mz <= region_mz_upper) & iso_scan_condition & iso_rt_condition]

                            # intensity descent
                            raw_points_a = feature_region_3d_df[['mz','intensity']].to_numpy()
                            peaks_a = intensity_descent(peaks_a=raw_points_a, peak_delta=None)

                            # deconvolution - see https://mobiusklein.github.io/ms_deisotope/docs/_build/html/deconvolution/deconvolution.html
                            # returns a collection of DeconvolutedPeak (https://github.com/mobiusklein/ms_deisotope/blob/bce522a949579a5f54465eab24194eb5693f40ef/ms_deisotope/peak_set.py#L78) representing a single deconvoluted peak 
                            # which represents an aggregated isotopic pattern collapsed to its monoisotopic peak, with a known charge state
                            ms1_peaks_l = list(map(tuple, peaks_a))
                            deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, use_quick_charge=True, averagine=averagine.peptide, truncate_after=0.95)

                            # collect features from deconvolution
                            ms1_deconvoluted_peaks_l = []
                            for peak_idx,peak in enumerate(deconvoluted_peaks):
                                # discard a monoisotopic peak that has either of the first two peaks as placeholders (indicated by intensity of 1)
                                if ((len(peak.envelope) >= MINIMUM_NUMBER_OF_ISOTOPES) and (peak.envelope[0][1] > 1) and (peak.envelope[1][1] > 1)):
                                    mono_peak_mz = peak.mz
                                    mono_intensity = peak.intensity
                                    second_peak_mz = peak.envelope[1][0]
                                    # an accepted feature must have its mono peak or base peak aligned with the voxel
                                    if (((mono_peak_mz >= iso_mz_lower) and (mono_peak_mz <= iso_mz_upper)) or ((second_peak_mz >= iso_mz_lower) and (second_peak_mz <= iso_mz_upper))):
                                        ms1_deconvoluted_peaks_l.append((mono_peak_mz, second_peak_mz, mono_intensity, peak.score, peak.signal_to_noise, peak.charge, peak.envelope, peak.neutral_mass))
                            deconvolution_features_df = pd.DataFrame(ms1_deconvoluted_peaks_l, columns=['mono_mz','second_peak_mz','intensity','score','SN','charge','envelope','neutral_mass'])
                            deconvolution_features_df.sort_values(by=['score'], ascending=False, inplace=True)

                            if len(deconvolution_features_df) > 0:
                                # take the top N scoring features
                                deconvolution_features_df = deconvolution_features_df.head(n=TARGET_NUMBER_OF_FEATURES_FOR_CUBOID)
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

                                    feature_d['inverse_k0_apex'] = inverse_k0_apex

                                    feature_d['rt_apex'] = rt_apex
                                    feature_d['rt_lower'] = iso_rt_lower
                                    feature_d['rt_upper'] = iso_rt_upper

                                    isotope_characteristics_d = determine_isotope_characteristics(envelope=feature.envelope, rt_apex=rt_apex, monoisotopic_mass=feature.neutral_mass, feature_region_3d_df=feature_region_3d_df)
                                    if isotope_characteristics_d is not None:
                                        # add the characteristics to the feature dictionary
                                        feature_d = {**feature_d, **isotope_characteristics_d}
                                        # add the voxels included in the feature's isotopes to the set of voxels already processed
                                        voxels_processed.update(feature_d['voxels_processed'])
                                        # only add the feature to the list if it has a minimum number of isotopes
                                        if feature_d['isotope_count'] >= MINIMUM_NUMBER_OF_ISOTOPES:
                                            feature_d['monoisotopic_mz'] = feature.mono_mz
                                            feature_d['charge'] = feature.charge
                                            feature_d['neutral_mass'] = feature.neutral_mass
                                            feature_d['monoisotopic_mass'] = calculate_monoisotopic_mass_from_mz(feature.mono_mz, feature.charge)
                                            feature_d['feature_intensity'] = isotope_characteristics_d['intensity_with_saturation_correction'] if (isotope_characteristics_d['intensity_with_saturation_correction'] > isotope_characteristics_d['intensity_without_saturation_correction']) else isotope_characteristics_d['intensity_without_saturation_correction']
                                            feature_d['deconvolution_envelope'] = json.dumps([tuple(e) for e in feature.envelope])
                                            feature_d['deconvolution_score'] = feature.score
                                            # record the feature region where we found this feature
                                            feature_d['feature_region_3d_extent'] = json.dumps(feature_region_3d_extent_d)
                                            # record the voxel from where we derived the initial isotope
                                            feature_d['voxel_id'] = voxel.voxel_id
                                            feature_d['voxel_metadata_d'] = json.dumps(voxel_metadata_d)
                                            feature_d['scan_df'] = scan_df.to_json(orient='records')
                                            feature_d['scan_r_squared'] = scan_r_squared
                                            feature_d['rt_df'] = rt_df.to_json(orient='records')
                                            feature_d['rt_r_squared'] = rt_r_squared
                                            feature_d['number_of_frames'] = len(base_peak_df.frame_id.unique())
                                            feature_d['voxels_processed'] = json.dumps(list(feature_d['voxels_processed']))
                                            # add it to the list
                                            features_l.append(feature_d)
                                        else:
                                            print('not enough isotopes ({}) were found for the feature derived from voxel {}'.format(feature_d['isotope_count'], voxel.voxel_id)) if args.verbose else None
                                    else:
                                        print('no feature characteristics were determined for voxel {}'.format(voxel.voxel_id)) if args.verbose else None
                            else:
                                print('deconvolution yielded no features for voxel {}'.format(voxel.voxel_id)) if args.verbose else None
                        else:
                            print('the base peak formed from voxel {} does not include it'.format(voxel.voxel_id)) if args.verbose else None
                    else:
                        print('the base peak for voxel {} does not have enough points in the mobility dimension ({})'.format(voxel.voxel_id, len(scan_df))) if args.verbose else None
                else:
                    print('voxel {} has already been used by another feature'.format(voxel.voxel_id)) if args.verbose else None
        else:
            print('no summaries to process in segment {} ({}-{} m/z)'.format(segment_d['segment_id'], round(segment_d['mz_lower']), round(segment_d['mz_upper'])))
    else:
        print('no raw points were found in segment {} ({}-{} m/z)'.format(segment_d['segment_id'], round(segment_d['mz_lower']), round(segment_d['mz_upper'])))
        
    features_df = pd.DataFrame(features_l)
    if len(features_df) > 0:
        # downcast the data types to minimise the memory used
        int_columns = ['intensity_without_saturation_correction','intensity_with_saturation_correction','isotope_count','charge','feature_intensity','voxel_id']
        features_df[int_columns] = features_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
        float_columns = ['mono_mz_lower','mono_mz_upper','scan_apex','scan_lower','scan_upper','rt_apex','rt_lower','rt_upper','coelution_coefficient','mobility_coefficient','monoisotopic_mz','monoisotopic_mass','deconvolution_score','scan_r_squared','rt_r_squared']
        features_df[float_columns] = features_df[float_columns].apply(pd.to_numeric, downcast="float")    
    # save these features until we have all the segments processed
    interim_df_name = '{}/features-segment-{}.feather'.format(INTERIM_FEATURES_DIR, segment_d['segment_id'])
    features_df.reset_index().to_feather(interim_df_name)
    return interim_df_name

# remove the isotopic peak profiles to save space
def strip_peaks(peaks_l):
    df = pd.read_json(peaks_l)
    df.drop(['scan_df','rt_df'], axis=1, inplace=True)
    return df.to_json(orient='records')


#######################
parser = argparse.ArgumentParser(description='Find all the features in a run with 3D intensity descent.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-ml','--mz_lower', type=int, help='Lower limit for m/z. If not specified, use the lowest m/z in the data.', required=False)
parser.add_argument('-mu','--mz_upper', type=int, help='Upper limit for m/z. If not specified, use the highest m/z in the data.', required=False)
parser.add_argument('-mw','--mz_width_per_segment', type=int, default=20, help='Width in Da of the m/z processing window per segment.', required=False)
parser.add_argument('-rl','--rt_lower', type=int, help='Lower limit for retention time. If not specified, use the lowest RT in the data.', required=False)
parser.add_argument('-ru','--rt_upper', type=int, help='Upper limit for retention time. If not specified, use the highest RT in the data.', required=False)
parser.add_argument('-minvi','--minimum_voxel_intensity', type=int, default='2500', help='The minimum voxel intensity to analyse.', required=False)
parser.add_argument('-mi','--minimum_point_intensity', type=int, default='200', help='The minimum point intensity to include.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./tfde/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], help='The Ray mode to use.', required=True)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.9, help='Proportion of the machine\'s cores to use for this program.', required=False)
parser.add_argument('-v','--verbose', action='store_true', help='Print more information during processing.')
parser.add_argument('-d','--denoised', action='store_true', help='Use the denoised version of the raw database.')
args = parser.parse_args()

# print the arguments for the log
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

# check the raw database exists
if args.denoised:
    RAW_DATABASE_BASE_DIR = "{}/raw-databases/denoised".format(EXPERIMENT_DIR)
else:
    RAW_DATABASE_BASE_DIR = "{}/raw-databases".format(EXPERIMENT_DIR)
RAW_DATABASE_NAME = "{}/{}.d".format(RAW_DATABASE_BASE_DIR, args.run_name)
if not os.path.exists(RAW_DATABASE_NAME):
    print("The raw database is required but doesn't exist: {}".format(RAW_DATABASE_NAME))
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
TARGET_NUMBER_OF_FEATURES_FOR_CUBOID = cfg.getint('3did', 'TARGET_NUMBER_OF_FEATURES_FOR_CUBOID')

VOXEL_SIZE_RT = cfg.getint('3did', 'VOXEL_SIZE_RT')
VOXEL_SIZE_SCAN = cfg.getint('3did', 'VOXEL_SIZE_SCAN')
VOXEL_SIZE_MZ = cfg.getfloat('3did', 'VOXEL_SIZE_MZ')

MINIMUM_NUMBER_OF_ISOTOPES = cfg.getfloat('ms1', 'MINIMUM_NUMBER_OF_ISOTOPES')

MINIMUM_NUMBER_OF_SCANS_IN_BASE_PEAK = cfg.getint('3did', 'MINIMUM_NUMBER_OF_SCANS_IN_BASE_PEAK')
MAXIMUM_GAP_SECS_BETWEEN_EDGE_POINTS = cfg.getfloat('3did', 'MAXIMUM_GAP_SECS_BETWEEN_EDGE_POINTS')
INTENSITY_PROPORTION_FOR_VOXEL_TO_BE_REMOVED = cfg.getfloat('3did', 'INTENSITY_PROPORTION_FOR_VOXEL_TO_BE_REMOVED')
ISOTOPE_SIMILARITY_RT_THRESHOLD = cfg.getfloat('3did', 'ISOTOPE_SIMILARITY_RT_THRESHOLD')
ISOTOPE_SIMILARITY_CCS_THRESHOLD = cfg.getfloat('3did', 'ISOTOPE_SIMILARITY_CCS_THRESHOLD')

SEGMENT_EXTENSION = cfg.getfloat('3did', 'SEGMENT_EXTENSION')

# move these constants to the INI file
ANCHOR_POINT_MZ_LOWER_OFFSET = CARBON_MASS_DIFFERENCE / 1
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

# filter parameters; the smaller the order compared to the window length, the more smoothing
SCAN_FILTER_POLY_ORDER = 5
RT_FILTER_POLY_ORDER = 5

# set up the output features
FEATURES_DIR = "{}/features-3did".format(EXPERIMENT_DIR)
FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.feather'.format(FEATURES_DIR, args.experiment_name, args.run_name)
# set up the output directory
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)

# set up the interim features directory
INTERIM_FEATURES_DIR = "{}/interim".format(FEATURES_DIR)
if os.path.exists(INTERIM_FEATURES_DIR):
    shutil.rmtree(INTERIM_FEATURES_DIR)
os.makedirs(INTERIM_FEATURES_DIR)

# set up the segments directory
SEGMENTS_DIR = "{}/segments".format(FEATURES_DIR)
if os.path.exists(SEGMENTS_DIR):
    shutil.rmtree(SEGMENTS_DIR)
os.makedirs(SEGMENTS_DIR)

# set up the summary directory
SUMMARY_DIR = "{}/summary".format(FEATURES_DIR)
if os.path.exists(SUMMARY_DIR):
    shutil.rmtree(SUMMARY_DIR)
os.makedirs(SUMMARY_DIR)

# set up Ray
print("setting up Ray")
if not ray.is_initialized():
    if args.ray_mode == "cluster":
        ray.init(num_cpus=number_of_workers())
    else:
        ray.init(local_mode=True)

# load the raw database
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

# set the retention time to the whole scope of the data if no arguments have been provided
if (args.rt_lower is None) or (args.rt_lower == -1):
    RT_LOWER = 1.0
    info.append(('rt_lower', RT_LOWER))
else:
    RT_LOWER = args.rt_lower
if (args.rt_upper is None) or (args.rt_upper == -1):
    RT_UPPER = data.rt_max_value
    info.append(('rt_upper', RT_UPPER))
else:
    RT_UPPER = args.rt_upper

# set the m/z range to the whole scope of the data if no arguments have been provided
if (args.mz_lower is None) or (args.mz_lower == -1):
    MZ_LOWER = data.mz_min_value
    info.append(('mz_lower', MZ_LOWER))
else:
    MZ_LOWER = args.mz_lower
if (args.mz_upper is None) or (args.mz_upper == -1):
    MZ_UPPER = data.mz_max_value
    info.append(('mz_upper', MZ_UPPER))
else:
    MZ_UPPER = args.mz_upper

print('region of analysis: {}-{} m/z, {}-{} secs'.format(MZ_LOWER, MZ_UPPER, RT_LOWER, RT_UPPER))

# calculate the segments
mz_range = MZ_UPPER - MZ_LOWER
NUMBER_OF_MZ_SEGMENTS = (mz_range // args.mz_width_per_segment) + (mz_range % args.mz_width_per_segment > 0)  # thanks to https://stackoverflow.com/a/23590097/1184799

# split the raw data into segments
print('segmenting the raw data')
segment_packages_l = []
for i in range(NUMBER_OF_MZ_SEGMENTS):
    mz_lower=float(MZ_LOWER+(i*args.mz_width_per_segment))
    mz_upper=float(MZ_LOWER+(i*args.mz_width_per_segment)+args.mz_width_per_segment)
    rt_lower=float(RT_LOWER)
    rt_upper=float(RT_UPPER)
    segment_id=i+1
    # extract the raw points for this segment
    segment_df = data[
        {
            "rt_values": slice(rt_lower, rt_upper),
            "mz_values": slice(mz_lower, mz_upper+SEGMENT_EXTENSION),
            "intensity_values": slice(args.minimum_point_intensity, None),
            "precursor_indices": 0,
        }
    ][['mz_values','scan_indices','mobility_values','frame_indices','rt_values','intensity_values']]
    segment_df.rename(columns={'mz_values':'mz', 'scan_indices':'scan', 'mobility_values':'inverse_k0', 'frame_indices':'frame_id', 'rt_values':'retention_time_secs', 'intensity_values':'intensity'}, inplace=True)
    # downcast the data types to minimise the memory used
    int_columns = ['frame_id','scan','intensity']
    segment_df[int_columns] = segment_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
    float_columns = ['retention_time_secs','inverse_k0']
    segment_df[float_columns] = segment_df[float_columns].apply(pd.to_numeric, downcast="float")    
    # save the segment
    # segment_name = '{}/segment-{}.pkl'.format(SEGMENTS_DIR, segment_id)
    # segment_df.to_pickle(segment_name)
    # segment_packages_l.append({'mz_lower':mz_lower, 'mz_upper':mz_upper, 'rt_lower':rt_lower, 'rt_upper':rt_upper, 'scan_limit':scan_limit, 'segment_id':segment_id, 'segment_name':segment_name})
    segment_packages_l.append({'mz_lower':mz_lower, 'mz_upper':mz_upper, 'rt_lower':rt_lower, 'rt_upper':rt_upper, 'segment_id':segment_id, 'segment_df':segment_df})
del data

# find all the features
print('finding features')
interim_names_l = ray.get([find_features.remote(segment_d=sp) for sp in segment_packages_l])
# interim_names_l = [find_features(segment_d=sp) for sp in segment_packages_l]
segment_packages_l = None

# join the list of dataframes into a single dataframe
print('collating the detected features')
features_l = []
for segment_file_name in interim_names_l:
    df = pd.read_feather(segment_file_name)
    if len(df) > 0:
        df['isotopic_peaks'] = df.apply(lambda row: strip_peaks(row.isotopic_peaks), axis=1)
        features_l.append(df)
features_df = pd.concat(features_l, axis=0, sort=False, ignore_index=True)
del features_l

# assign each feature a unique identifier
features_df['feature_id'] = features_df.index

# ... and save them in a file
print()
FEATURES_FILE = '{}/exp-{}-run-{}-features-3did.feather'.format(FEATURES_DIR, args.experiment_name, args.run_name)
features_df.reset_index(drop=True).to_feather(FEATURES_FILE)

# write the metadata
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
FEATURES_METADATA_FILE = '{}/exp-{}-run-{}-features-3did.json'.format(FEATURES_DIR, args.experiment_name, args.run_name)
with open(FEATURES_METADATA_FILE, 'w') as handle:
    json.dump(info, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
