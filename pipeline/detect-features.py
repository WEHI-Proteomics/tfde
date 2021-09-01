import pandas as pd
import numpy as np
import sys
import os.path
import argparse
import time
import json
import multiprocessing as mp
import ray
from ms_deisotope import deconvolute_peaks, averagine, scoring
from ms_deisotope.deconvolution import peak_retention_strategy
import pickle
import configparser
from configparser import ExtendedInterpolation
from os.path import expanduser
import peakutils
from scipy import signal
import math
from sklearn.metrics.pairwise import cosine_similarity
import alphatims.bruker


# peak and valley detection parameters
PEAKS_THRESHOLD_RT = 0.5    # only consider peaks that are higher than this proportion of the normalised maximum
PEAKS_THRESHOLD_SCAN = 0.5
PEAKS_MIN_DIST_RT = 2.0     # seconds
PEAKS_MIN_DIST_SCAN = 10.0  # scans

VALLEYS_THRESHOLD_RT = 0.5    # only consider valleys that drop more than this proportion of the normalised maximum
VALLEYS_THRESHOLD_SCAN = 0.5
VALLEYS_MIN_DIST_RT = 2.0     # seconds
VALLEYS_MIN_DIST_SCAN = 10.0  # scans

# filter parameters
SCAN_FILTER_POLY_ORDER = 5
RT_FILTER_POLY_ORDER = 3

# determine the maximum filter length for the number of points
def find_filter_length(number_of_points):
    filter_lengths = [51,11,5]  # must be a positive odd number, greater than the polynomial order, and less than the number of points to be filtered
    return filter_lengths[next(x[0] for x in enumerate(filter_lengths) if x[1] < number_of_points)]

# calculate the intensity-weighted centroid
# takes a numpy array of intensity, and another of mz
def intensity_weighted_centroid(_int_f, _x_f):
    return ((_int_f/_int_f.sum()) * _x_f).sum()

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

# find 3sigma for a specified m/z
def calculate_peak_delta(mz):
    delta_m = mz / INSTRUMENT_RESOLUTION  # FWHM of the peak
    sigma = delta_m / 2.35482  # std dev is FWHM / 2.35482. See https://mathworld.wolfram.com/GaussianFunction.html
    peak_delta = 3 * sigma  # 99.7% of values fall within +/- 3 sigma
    return peak_delta

# calculate the monoisotopic mass    
def calculate_monoisotopic_mass_from_mz(monoisotopic_mz, charge):
    monoisotopic_mass = (monoisotopic_mz * charge) - (PROTON_MASS * charge)
    return monoisotopic_mass

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
    return float(cosine_similarity([combined_df.intensity_A.values], [combined_df.intensity_B.values])) if len(combined_df) > 0 else None

# determine the mono peak apex and extent in CCS and RT and calculate isotopic peak intensities
def determine_mono_characteristics(envelope, mono_mz_lower, mono_mz_upper, monoisotopic_mass, cuboid_points_df):

    # determine the raw points that belong to the mono peak
    # we use the wider cuboid points because we want to discover the apex and extent in CCS and RT
    mono_points_df = cuboid_points_df[(cuboid_points_df.mz >= mono_mz_lower) & (cuboid_points_df.mz <= mono_mz_upper)]

    # determine the peak's extent in CCS and RT
    if len(mono_points_df) > 0:
        # collapsing the monoisotopic's summed points onto the mobility dimension
        scan_df = mono_points_df.groupby(['scan'], as_index=False).intensity.sum()
        scan_df.sort_values(by=['scan'], ascending=True, inplace=True)

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

        # find the closest peak to the cuboid midpoint
        cuboid_midpoint_scan = scan_df.scan.min() + ((scan_df.scan.max() - scan_df.scan.min()) / 2)
        peaks_df['delta'] = abs(peaks_df.scan - cuboid_midpoint_scan)
        peaks_df.sort_values(by=['delta'], ascending=True, inplace=True)
        scan_apex = peaks_df.iloc[0].scan

        # find the valleys nearest the scan apex
        valley_idxs = peakutils.indexes(-scan_df.filtered_intensity.values.astype(int), thres=VALLEYS_THRESHOLD_SCAN, min_dist=VALLEYS_MIN_DIST_SCAN, thres_abs=False)
        valley_x_l = scan_df.iloc[valley_idxs].scan.to_list()
        valleys_df = scan_df[scan_df.scan.isin(valley_x_l)]

        upper_x = valleys_df[valleys_df.scan > scan_apex].scan.min()
        if math.isnan(upper_x):
            upper_x = scan_df.scan.max()
        lower_x = valleys_df[valleys_df.scan < scan_apex].scan.max()
        if math.isnan(lower_x):
            lower_x = scan_df.scan.min()

        scan_lower = lower_x
        scan_upper = upper_x

        # constrain the mono points to the CCS extent
        mono_points_df = mono_points_df[(mono_points_df.scan >= scan_lower) & (mono_points_df.scan <= scan_upper)]

        # in the RT dimension, look wider to find the apex
        rt_df = mono_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
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

        # find the closest peak to the cuboid midpoint
        cuboid_midpoint_rt = rt_df.retention_time_secs.min() + ((rt_df.retention_time_secs.max() - rt_df.retention_time_secs.min()) / 2)
        peaks_df['delta'] = abs(peaks_df.retention_time_secs - cuboid_midpoint_rt)
        peaks_df.sort_values(by=['delta'], ascending=True, inplace=True)
        rt_apex = peaks_df.iloc[0].retention_time_secs

        # find the valleys nearest the RT apex
        valley_idxs = peakutils.indexes(-rt_df.filtered_intensity.values.astype(int), thres=VALLEYS_THRESHOLD_RT, min_dist=VALLEYS_MIN_DIST_RT, thres_abs=False)
        valley_x_l = rt_df.iloc[valley_idxs].retention_time_secs.to_list()
        valleys_df = rt_df[rt_df.retention_time_secs.isin(valley_x_l)]

        upper_x = valleys_df[valleys_df.retention_time_secs > rt_apex].retention_time_secs.min()
        if math.isnan(upper_x):
            upper_x = rt_df.retention_time_secs.max()
        lower_x = valleys_df[valleys_df.retention_time_secs < rt_apex].retention_time_secs.max()
        if math.isnan(lower_x):
            lower_x = rt_df.retention_time_secs.min()

        rt_lower = lower_x
        rt_upper = upper_x

        # constrain the mono points to the RT extent
        mono_points_df = mono_points_df[(mono_points_df.retention_time_secs >= rt_lower) & (mono_points_df.retention_time_secs <= rt_upper)]

        # for the whole feature, constrain the raw points to the CCS and RT extent of the monoisotopic peak
        mono_ccs_rt_extent_df = cuboid_points_df[(cuboid_points_df.scan >= scan_lower) & (cuboid_points_df.scan <= scan_upper) & (cuboid_points_df.retention_time_secs >= rt_lower) & (cuboid_points_df.retention_time_secs <= rt_upper)]

        # calculate the isotope intensities from the constrained raw points
        isotopes_l = []
        for idx,isotope in enumerate(envelope):
            # gather the points that belong to this isotope
            iso_mz = isotope[0]
            iso_intensity = isotope[1]
            iso_mz_delta = calculate_peak_delta(iso_mz)
            iso_mz_lower = iso_mz - iso_mz_delta
            iso_mz_upper = iso_mz + iso_mz_delta
            isotope_df = mono_ccs_rt_extent_df[(mono_ccs_rt_extent_df.mz >= iso_mz_lower) & (mono_ccs_rt_extent_df.mz <= iso_mz_upper)]
            if len(isotope_df) > 0:
                # find the intensity by summing the maximum point in the frame closest to the RT apex, and the frame maximums either side
                frame_maximums_df = isotope_df.groupby(['retention_time_secs'], as_index=False, sort=False).intensity.agg(['max']).reset_index()
                frame_maximums_df['rt_delta'] = np.abs(frame_maximums_df.retention_time_secs - rt_apex)
                frame_maximums_df.sort_values(by=['rt_delta'], ascending=True, inplace=True)
                # sum the maximum intensity and the max intensity of the frame either side in RT
                summed_intensity = frame_maximums_df[:3]['max'].sum()
                # are any of the three points in saturation?
                isotope_in_saturation = (frame_maximums_df[:3]['max'].max() > SATURATION_INTENSITY)
                # determine the isotope's profile in retention time
                iso_rt_df = isotope_df.groupby(['retention_time_secs'], as_index=False).intensity.sum()
                iso_rt_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)
                # measure it's elution similarity with the previous isotope
                similarity_rt = measure_peak_similarity(pd.read_json(isotopes_l[idx-1]['rt_df']), iso_rt_df, x_label='retention_time_secs', scale=100) if idx > 0 else None
                # determine the isotope's profile in mobility
                iso_scan_df = isotope_df.groupby(['scan'], as_index=False).intensity.sum()
                iso_scan_df.sort_values(by=['scan'], ascending=True, inplace=True)
                # measure it's elution similarity with the previous isotope
                similarity_scan = measure_peak_similarity(pd.read_json(isotopes_l[idx-1]['scan_df']), iso_scan_df, x_label='scan', scale=1) if idx > 0 else None
                # add the isotope to the list
                isotopes_l.append({'mz':iso_mz, 'mz_lower':iso_mz_lower, 'mz_upper':iso_mz_upper, 'intensity':summed_intensity, 'saturated':isotope_in_saturation, 'rt_df':iso_rt_df.to_json(orient='records'), 'scan_df':iso_scan_df.to_json(orient='records'), 'similarity_rt':similarity_rt, 'similarity_scan':similarity_scan})
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
        result_d['scan_apex'] = scan_apex
        result_d['scan_lower'] = scan_lower
        result_d['scan_upper'] = scan_upper
        result_d['rt_apex'] = rt_apex
        result_d['rt_lower'] = rt_lower
        result_d['rt_upper'] = rt_upper
        result_d['intensity_without_saturation_correction'] = isotopes_df.iloc[:3].intensity.sum()  # only take the first three isotopes for intensity, as the number of isotopes varies
        result_d['intensity_with_saturation_correction'] = isotopes_df.iloc[:3].inferred_intensity.sum()
        result_d['mono_intensity_adjustment_outcome'] = outcome
        result_d['isotopic_peaks'] = isotopes_df.to_json(orient='records')
        result_d['coelution_coefficient'] = coelution_coefficient
        result_d['mobility_coefficient'] = mobility_coefficient
        result_d['scan_df'] = scan_df.to_json(orient='records')
        result_d['rt_df'] = rt_df.to_json(orient='records')
    else:
        print('found no raw points where the mono peak should be')
        result_d = None
    return result_d

# create the bins for mass defect windows in Da space
def generate_mass_defect_windows(mass_defect_window_da_min, mass_defect_window_da_max):
    bin_edges_l = []
    for nominal_mass in range(mass_defect_window_da_min, mass_defect_window_da_max):
        mass_centre = nominal_mass * 1.00048  # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3184890/
        width = 0.19 + (0.0001 * nominal_mass)
        lower_mass = mass_centre - (width / 2)
        upper_mass = mass_centre + (width / 2)
        bin_edges_l.append((lower_mass, upper_mass))
    return bin_edges_l

# resolve the fragment ions for this feature
# returns a decharged peak list (neutral mass+proton mass, intensity)
def resolve_fragment_ions(feature_d, ms2_points_df, mass_defect_bins):
    vis_d = {}
    vis_d['ms2_points_l'] = ms2_points_df[['mz','intensity']].to_json(orient='records')
    # perform intensity descent to resolve peaks
    raw_points_a = ms2_points_df[['mz','intensity']].to_numpy()
    peaks_a = intensity_descent(peaks_a=raw_points_a, peak_delta=None)
    # deconvolution - see https://mobiusklein.github.io/ms_deisotope/docs/_build/html/deconvolution/deconvolution.html
    peaks_l = list(map(tuple, peaks_a))
    maximum_neutral_mass = 1700*feature_d['charge']  # give the deconvolution a reasonable upper limit to search within
    deconvoluted_peaks, _ = deconvolute_peaks(peaks_l, use_quick_charge=True, averagine=averagine.peptide, scorer=scoring.PenalizedMSDeconVFitter(minimum_score=20., penalty_factor=3.0), truncate_after=0.95, ignore_below=0.0, charge_range=(1,feature_d['charge']), retention_strategy=peak_retention_strategy.TopNRetentionStrategy(n_peaks=100, base_peak_coefficient=1e-6, max_mass=maximum_neutral_mass))
    # package the spectra as a list
    deconvoluted_peaks_l = []
    for peak in deconvoluted_peaks:
        d = {}
        d['singly_protonated_mass'] = round(peak.neutral_mass+PROTON_MASS, 4)
        d['neutral_mass'] = round(peak.neutral_mass, 4)
        d['intensity'] = peak.intensity
        deconvoluted_peaks_l.append(d)
    vis_d['before_fmdw'] = deconvoluted_peaks_l

    if args.filter_by_mass_defect:
        fragment_ions_df = pd.DataFrame(deconvoluted_peaks_l)
        fragment_ions_df['bin'] = pd.cut(fragment_ions_df.neutral_mass, mass_defect_bins)
        filtered_fragment_ions_df = fragment_ions_df.dropna(subset = ['bin']).copy()
        filtered_fragment_ions_df.drop('bin', axis=1, inplace=True)
        deconvoluted_peaks_l = filtered_fragment_ions_df.to_dict(orient='records')

        vis_d['after_fmdw'] = deconvoluted_peaks_l

        # removed = len(fragment_ions_df) - len(filtered_fragment_ions_df)
        # print('removed {} fragment ions ({}%)'.format(removed, round(removed/len(fragment_ions_df)*100,1)))
    else:
        vis_d['after_fmdw'] = []

    return {'deconvoluted_peaks_l':deconvoluted_peaks_l, 'vis_d':vis_d}

# save visualisation data for later analysis of how feature detection works
def save_visualisation(visualise_d):
    precursor_cuboid_id = visualise_d['precursor_cuboid_d']['precursor_cuboid_id']
    VIS_FILE = '{}/feature-detection-pasef-visualisation-{}.pkl'.format(expanduser("~"), precursor_cuboid_id)
    print("writing feature detection visualisation data to {}".format(VIS_FILE))
    with open(VIS_FILE, 'wb') as handle:
        pickle.dump(visualise_d, handle)

# prepare the metadata and raw points for the feature detection
@ray.remote
def detect_features(cuboid, mass_defect_bins, visualise):
    # load the raw points for this cuboid
    wide_ms1_points_df = cuboid['ms1_df']
    precursor_cuboid = cuboid['precursor_cuboid']
    # for deconvolution, constrain the CCS and RT dimensions to the fragmentation event
    fe_ms1_points_df = wide_ms1_points_df[(wide_ms1_points_df.retention_time_secs >= precursor_cuboid.fe_ms1_rt_lower) & (wide_ms1_points_df.retention_time_secs <= precursor_cuboid.fe_ms1_rt_upper) & (wide_ms1_points_df.scan >= precursor_cuboid.fe_scan_lower) & (wide_ms1_points_df.scan <= precursor_cuboid.fe_scan_upper)]

    # intensity descent
    raw_points_a = fe_ms1_points_df[['mz','intensity']].to_numpy()
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
            ms1_deconvoluted_peaks_l.append((mono_peak_mz, second_peak_mz, mono_intensity, peak.score, peak.signal_to_noise, peak.charge, peak.envelope, peak.neutral_mass))
    df = pd.DataFrame(ms1_deconvoluted_peaks_l, columns=['mono_mz','second_peak_mz','intensity','score','SN','charge','envelope','neutral_mass'])
    df.sort_values(by=['score'], ascending=False, inplace=True)

    if len(df) > 0:
        # take the top N scoring features
        deconvolution_features_df = df.head(n=TARGET_NUMBER_OF_FEATURES_FOR_CUBOID)

        # load the ms2 data for the precursor
        ms2_points_df = cuboid['ms2_df']

        # determine the feature attributes
        feature_l = []
        for idx,row in enumerate(deconvolution_features_df.itertuples()):
            feature_d = {}
            envelope_mono_mz = row.envelope[0][0]
            mz_delta = calculate_peak_delta(mz=envelope_mono_mz)
            mono_mz_lower = envelope_mono_mz - mz_delta
            mono_mz_upper = envelope_mono_mz + mz_delta
            feature_d['mono_mz_lower'] = mono_mz_lower
            feature_d['mono_mz_upper'] = mono_mz_upper
            mono_characteristics_d = determine_mono_characteristics(envelope=row.envelope, mono_mz_lower=mono_mz_lower, mono_mz_upper=mono_mz_upper, monoisotopic_mass=row.neutral_mass, cuboid_points_df=wide_ms1_points_df)
            if mono_characteristics_d is not None:
                # add the characteristics to the feature dictionary
                feature_d = {**feature_d, **mono_characteristics_d}
                feature_d['monoisotopic_mz'] = row.mono_mz
                feature_d['charge'] = row.charge
                feature_d['monoisotopic_mass'] = calculate_monoisotopic_mass_from_mz(monoisotopic_mz=feature_d['monoisotopic_mz'], charge=feature_d['charge'])
                feature_d['feature_intensity'] = mono_characteristics_d['intensity_with_saturation_correction'] if (args.correct_for_saturation and (mono_characteristics_d['intensity_with_saturation_correction'] > mono_characteristics_d['intensity_without_saturation_correction'])) else mono_characteristics_d['intensity_without_saturation_correction']
                feature_d['envelope'] = json.dumps([tuple(e) for e in row.envelope])
                feature_d['isotope_count'] = len(row.envelope)
                feature_d['deconvolution_score'] = row.score
                # from the precursor cuboid
                feature_d['precursor_cuboid_id'] = precursor_cuboid.precursor_cuboid_id
                # resolve the feature's fragment ions
                ms2_resolution_d = resolve_fragment_ions(feature_d, ms2_points_df, mass_defect_bins)
                feature_d['fragment_ions_l'] = json.dumps(ms2_resolution_d['deconvoluted_peaks_l'])
                feature_d['fmdw_before_after_d'] = ms2_resolution_d['vis_d']
                # assign a unique identifier to this feature
                feature_d['feature_id'] = generate_feature_id(precursor_cuboid.precursor_cuboid_id, idx+1)
                # add it to the list
                feature_l.append(feature_d)
        features_df = pd.DataFrame(feature_l)

        # downcast the data types to minimise the memory used
        int_columns = ['scan_lower','scan_upper','intensity_without_saturation_correction','intensity_with_saturation_correction','charge','feature_intensity','isotope_count','precursor_cuboid_id','feature_id']
        features_df[int_columns] = features_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
        float_columns = ['mono_mz_lower','mono_mz_upper','scan_apex','rt_apex','rt_lower','rt_upper','coelution_coefficient','mobility_coefficient','monoisotopic_mz','monoisotopic_mass','deconvolution_score']
        features_df[float_columns] = features_df[float_columns].apply(pd.to_numeric, downcast="float")    
    else:
        deconvolution_features_df = pd.DataFrame()
        features_df = pd.DataFrame()

    # gather the information for visualisation if required
    if visualise:
        visualisation_d = {
            'precursor_cuboid_d':precursor_cuboid._asdict(),
            'wide_ms1_points_df':wide_ms1_points_df,
            'fe_ms1_points_df':fe_ms1_points_df,
            'peaks_after_intensity_descent':peaks_a,
            'deconvolution_features_df':deconvolution_features_df,
            'features_df':features_df
        }
        save_visualisation(visualisation_d)

    # print("found {} features for precursor {}".format(len(features_df), precursor_cuboid.precursor_cuboid_id))
    return features_df

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = round(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers

# generate a unique feature_id from the precursor id and the feature sequence number found for that precursor
def generate_feature_id(precursor_id, feature_sequence_number):
    feature_id = (precursor_id * 100) + feature_sequence_number  # assumes there will not be more than 99 features found for a precursor
    return feature_id

###################################
parser = argparse.ArgumentParser(description='Detect the features in a run\'s precursor cuboids.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-rl','--rt_lower', type=int, default='1650', help='Lower limit for retention time.', required=False)
parser.add_argument('-ru','--rt_upper', type=int, default='2200', help='Upper limit for retention time.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-pid', '--precursor_id', type=int, help='Only process this precursor ID.', required=False)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], help='The Ray mode to use.', required=True)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.9, help='Proportion of the machine\'s cores to use for this program.', required=False)
parser.add_argument('-cs','--correct_for_saturation', action='store_true', help='Correct for saturation when calculating monoisotopic m/z and intensity.')
parser.add_argument('-fmdw','--filter_by_mass_defect', action='store_true', help='Filter fragment ions by mass defect windows.')
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

# check the raw database exists
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
FRAME_TYPE_MS2 = cfg.getint('common','FRAME_TYPE_MS2')
MS1_PEAK_DELTA = cfg.getfloat('ms1', 'MS1_PEAK_DELTA')
MS2_PEAK_DELTA = cfg.getfloat('ms2', 'MS2_PEAK_DELTA')
PROTON_MASS = cfg.getfloat('common', 'PROTON_MASS')
RT_BASE_PEAK_WIDTH_SECS = cfg.getfloat('common', 'RT_BASE_PEAK_WIDTH_SECS')
SCAN_BASE_PEAK_WIDTH = cfg.getint('common', 'SCAN_BASE_PEAK_WIDTH')
INSTRUMENT_RESOLUTION = cfg.getfloat('common', 'INSTRUMENT_RESOLUTION')
SATURATION_INTENSITY = cfg.getint('common', 'SATURATION_INTENSITY')
TARGET_NUMBER_OF_FEATURES_FOR_CUBOID = cfg.getint('ms1', 'TARGET_NUMBER_OF_FEATURES_FOR_CUBOID')
CARBON_MASS_DIFFERENCE = cfg.getfloat('common', 'CARBON_MASS_DIFFERENCE')

# input cuboids
CUBOIDS_DIR = "{}/precursor-cuboids-pasef".format(EXPERIMENT_DIR)
CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-pasef.feather'.format(CUBOIDS_DIR, args.experiment_name, args.run_name)

# output features
FEATURES_DIR = "{}/features-pasef".format(EXPERIMENT_DIR)

# check the cuboids file
if not os.path.isfile(CUBOIDS_FILE):
    print("The cuboids file is required but doesn't exist: {}".format(CUBOIDS_FILE))
    sys.exit(1)

# load the precursor cuboids
precursor_cuboids_df = pd.read_feather(CUBOIDS_FILE)

# constrain the detection to the define RT limits
precursor_cuboids_df = precursor_cuboids_df[(precursor_cuboids_df['wide_ms1_rt_lower'] > args.rt_lower) & (precursor_cuboids_df['wide_ms1_rt_upper'] < args.rt_upper)]
print('loaded {} precursor cuboids within RT {}-{} from {}'.format(len(precursor_cuboids_df), args.rt_lower, args.rt_upper, CUBOIDS_FILE))

# limit the cuboids to just the selected one
if args.precursor_id is not None:
    precursor_cuboids_df = precursor_cuboids_df[(precursor_cuboids_df.precursor_cuboid_id == args.precursor_id)]
    if len(precursor_cuboids_df) == 0:
        print("The cuboids file doesn't contain precursor ID: {}".format(args.precursor_id))
        sys.exit(1)

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

print('loading the cuboids')
cuboids_l = []
for row in precursor_cuboids_df.itertuples():
    # load the ms1 points for this cuboid
    ms1_df = data[
        {
            "rt_values": slice(float(row.wide_ms1_rt_lower), float(row.wide_ms1_rt_upper)),
            "mz_values": slice(float(row.wide_mz_lower), float(row.wide_mz_upper)),
            "scan_indices": slice(int(row.wide_scan_lower), int(row.wide_scan_upper+1)),
            "precursor_indices": 0,  # ms1 frames only
        }
    ][['mz_values','scan_indices','frame_indices','rt_values','intensity_values']]
    ms1_df.rename(columns={'mz_values':'mz', 'scan_indices':'scan', 'frame_indices':'frame_id', 'rt_values':'retention_time_secs', 'intensity_values':'intensity'}, inplace=True)
    # downcast the data types to minimise the memory used
    int_columns = ['frame_id','scan','intensity']
    ms1_df[int_columns] = ms1_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
    float_columns = ['retention_time_secs']
    ms1_df[float_columns] = ms1_df[float_columns].apply(pd.to_numeric, downcast="float")
    # load the ms2 points for this cuboid
    ms2_df = data[
        {
            "frame_indices": slice(int(row.fe_ms2_frame_lower), int(row.fe_ms2_frame_upper+1)),
            "scan_indices": slice(int(row.fe_scan_lower), int(row.fe_scan_upper+1)),
            "precursor_indices": slice(1, None)  # ms2 frames only
        }
    ][['mz_values','scan_indices','frame_indices','rt_values','intensity_values']]
    ms2_df.rename(columns={'mz_values':'mz', 'scan_indices':'scan', 'frame_indices':'frame_id', 'rt_values':'retention_time_secs', 'intensity_values':'intensity'}, inplace=True)
    # downcast the data types to minimise the memory used
    int_columns = ['frame_id','scan','intensity']
    ms2_df[int_columns] = ms2_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
    float_columns = ['retention_time_secs']
    ms2_df[float_columns] = ms2_df[float_columns].apply(pd.to_numeric, downcast="float")
    # add them to the list
    cuboids_l.append({'ms1_df':ms1_df, 'ms2_df':ms2_df, 'precursor_cuboid':row})
del data

# generate the mass defect windows
mass_defect_bins = pd.IntervalIndex.from_tuples(generate_mass_defect_windows(100, 8000))

# find the features in each precursor cuboid
print('detecting features')
features_l = ray.get([detect_features.remote(cuboid=cuboid, mass_defect_bins=mass_defect_bins, visualise=(args.precursor_id is not None)) for cuboid in cuboids_l])
del cuboids_l[:]

# join the list of dataframes into a single dataframe
features_df = pd.concat(features_l, axis=0, sort=False, ignore_index=True)
del features_l[:]

# check we got something
if len(features_df) == 0:
    print('no features were found')
    sys.exit(1)

# add the run name
features_df['run_name'] = args.run_name

# write out all the features
print("writing {} features to {}".format(len(features_df), FEATURES_DIR))
chunk_size = 1000
num_chunks = len(features_df) // chunk_size
if len(features_df) % chunk_size != 0:
    num_chunks += 1
for i in range(num_chunks):
    FEATURES_FILE = '{}/exp-{}-run-{}-features-pasef-{:03d}.feather'.format(FEATURES_DIR, args.experiment_name, args.run_name, i)
    features_df[i*chunk_size:(i + 1) * chunk_size].reset_index().to_feather(FEATURES_FILE)

# write the metadata
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
with open(FEATURES_FILE.replace('.feather','-metadata.json'), 'w') as handle:
    json.dump(info, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
