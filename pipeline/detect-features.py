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
import pickle
import configparser
from configparser import ExtendedInterpolation


# set up the indexes we need for queries
def create_indexes(db_file_name):
    db_conn = sqlite3.connect(db_file_name)
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_extract_cuboids_1 on frames (frame_type,retention_time_secs,scan,mz)")
    db_conn.close()

# takes a numpy array of intensity, and another of mz
def mz_centroid(_int_f, _mz_f):
    return ((_int_f/_int_f.sum()) * _mz_f).sum()

# peaks_a is a numpy array of [mz,intensity]
# returns a numpy array of [mz_centroid,summed_intensity]
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
            mz_cent = mz_centroid(peaks_a[peak_indexes,1], peaks_a[peak_indexes,0])
            summed_intensity = peaks_a[peak_indexes,1].sum()
            peaks_l.append((mz_cent, summed_intensity))
            # remove the raw points assigned to this peak
            peaks_a = np.delete(peaks_a, peak_indexes, axis=0)
    return np.array(peaks_l)

# find 3sigma for a specified m/z
def calculate_peak_delta(mz):
    delta_m = mz / INSTRUMENT_RESOLUTION  # FWHM of the peak
    sigma = delta_m / 2.35482  # std dev is FWHM / 2.35482. See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    peak_delta = 3 * sigma  # 99.7% of values fall within +/- 3 sigma
    return peak_delta
    
# calculate the sum of the raw points in the mono m/z
def calculate_peak_intensity(peak_characteristics, raw_points):
    # extract the raw points for this peak
    mono_points_df = raw_points[(raw_points.mz >= peak_characteristics['mz_lower']) & (raw_points.mz <= peak_characteristics['mz_upper']) & (raw_points.scan >= peak_characteristics['scan_lower']) & (raw_points.scan <= peak_characteristics['scan_upper']) & (raw_points.retention_time_secs >= peak_characteristics['rt_lower']) & (raw_points.retention_time_secs <= peak_characteristics['rt_upper'])]
    mono_intensity = mono_points_df.intensity.sum()
    return mono_intensity

# calculate the mono intensity when it's model-adjusted for point saturation
def calculate_phr_adjusted_intensity(peak_characteristics, envelope, raw_points):
    # get the raw points for each isotope
    rt_lower = peak_characteristics['rt_lower']
    rt_upper = peak_characteristics['rt_upper']
    scan_lower = peak_characteristics['scan_lower']
    scan_upper = peak_characteristics['scan_upper']
    isotopes_l = []
    for idx,isotope in enumerate(envelope):
        mz = isotope[0]
        intensity = isotope[1]
        mz_delta = calculate_peak_delta(mz)
        mz_lower = mz - mz_delta
        mz_upper = mz + mz_delta
        df = raw_points[(raw_points.mz >= mz_lower) & (raw_points.mz <= mz_upper) & (raw_points.scan >= scan_lower) & (raw_points.scan <= scan_upper) & (raw_points.retention_time_secs >= rt_lower) & (raw_points.retention_time_secs <= rt_upper)]
        saturated = df.intensity.max() > SATURATION_INTENSITY
        isotopes_l.append({'intensity':df.intensity.sum(), 'saturated':saturated})
    isotopes_df = pd.DataFrame(isotopes_l)

    # set the summed intensity to be the default adjusted intensity for all isotopes
    isotopes_df['inferred_intensity'] = isotopes_df.intensity
    isotopes_df['inferred'] = False

    if len(isotopes_df.saturated.unique()) == 2:  # there are saturated and unsaturated isotopes
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
                break
        
    # return the mono intensity and whether it was adjusted for saturation
    mono_intensity = isotopes_df.iloc[0].inferred_intensity
    mono_inferred = isotopes_df.iloc[0].inferred
    return {'mono_intensity':mono_intensity, 'mono_inferred':mono_inferred, 'adjusted_isotopes':isotopes_df.to_dict('records')}

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

# determine the mono peak apex and extent in CCS and RT
def determine_mono_peak_characteristics(centre_mz, ms1_raw_points_df):
    # determine the raw points that belong to the mono peak
    mz_delta = calculate_peak_delta(centre_mz)
    mz_lower = centre_mz - mz_delta
    mz_upper = centre_mz + mz_delta
    mono_points_df = ms1_raw_points_df[(ms1_raw_points_df.mz >= mz_lower) & (ms1_raw_points_df.mz <= mz_upper)]

    # determine the peak's extent in CCS and RT
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

        # constrain the mono points to the CCS extent
        mono_points_df = mono_points_df[(mono_points_df.scan >= scan_lower) & (mono_points_df.scan <= scan_upper)]

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
            rt_lower = rt_apex - (RT_BASE_PEAK_WIDTH_SECS / 2)
            rt_upper = rt_apex + (RT_BASE_PEAK_WIDTH_SECS / 2)

        # constrain the mono points to the RT extent
        mono_points_df = mono_points_df[(mono_points_df.retention_time_secs >= rt_lower) & (mono_points_df.retention_time_secs <= rt_upper)]

        # now that we have the full extent of the feature in RT, recalculate the feature m/z to gain the most mass accuracy (without points in saturation)
        mono_points_without_saturation_df = mono_points_df[(mono_points_df.intensity < SATURATION_INTENSITY)]
        monoisotopic_mz_without_saturated_points = mz_centroid(mono_points_without_saturation_df.intensity.to_numpy(), mono_points_without_saturation_df.mz.to_numpy())

        # package the result
        result_d = {}
        result_d['mz_apex'] = centre_mz
        result_d['mz_lower'] = mz_lower
        result_d['mz_upper'] = mz_upper
        result_d['mono_mz_without_saturated_points'] = monoisotopic_mz_without_saturated_points
        result_d['scan_apex'] = scan_apex
        result_d['scan_lower'] = scan_lower
        result_d['scan_upper'] = scan_upper
        result_d['rt_apex'] = rt_apex
        result_d['rt_lower'] = rt_lower
        result_d['rt_upper'] = rt_upper
    else:
        print('found no raw points where the mono peak should be: {}'.format(round(centre_mz,4)))
        result_d = None
    return result_d

# resolve the fragment ions for this feature
# returns a decharged peak list (neutral mass+proton mass, intensity)
def resolve_fragment_ions(feature_d, ms2_points_df):
    # perform intensity descent to resolve peaks
    raw_points_a = ms2_points_df[['mz','intensity']].to_numpy()
    peaks_a = intensity_descent(peaks_a=raw_points_a, peak_delta=MS2_PEAK_DELTA)
    # deconvolute the spectra
    peaks_l = list(map(tuple, peaks_a))
    deconvoluted_peaks, _ = deconvolute_peaks(peaks_l, use_quick_charge=True, averagine=averagine.peptide, charge_range=(1,feature_d['charge']), scorer=scoring.MSDeconVFitter(minimum_score=MIN_SCORE_MS2_DECONVOLUTION_FEATURE, mass_error_tolerance=0.1), error_tolerance=4e-5, truncate_after=0.8, retention_strategy=peak_retention_strategy.TopNRetentionStrategy(n_peaks=100, base_peak_coefficient=1e-6, max_mass=1800.0))
    # package the spectra as a list
    deconvoluted_peaks_l = []
    for peak in deconvoluted_peaks:
        d = {}
        d['decharged_mass'] = round(peak.neutral_mass+PROTON_MASS, 4)
        d['intensity'] = peak.intensity
        deconvoluted_peaks_l.append(d)
    return deconvoluted_peaks_l

# prepare the metadata and raw points for the feature detection
@ray.remote
def detect_features(precursor_cuboid_d, converted_db_name):
    # load the raw points for this cuboid
    db_conn = sqlite3.connect(converted_db_name)
    wide_ms1_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} and scan >= {} and scan <= {} and mz >= {} and mz <= {}".format(FRAME_TYPE_MS1, precursor_cuboid_d['wide_ms1_rt_lower'], precursor_cuboid_d['wide_ms1_rt_upper'], precursor_cuboid_d['wide_scan_lower'], precursor_cuboid_d['wide_scan_upper'], precursor_cuboid_d['wide_mz_lower'], precursor_cuboid_d['wide_mz_upper']), db_conn)
    db_conn.close()

    # constrain the raw points to the isolation windows so we can find the features
    ms1_points_df = wide_ms1_points_df[(wide_ms1_points_df.mz >= precursor_cuboid_d['mz_lower']) & (wide_ms1_points_df.mz <= precursor_cuboid_d['mz_upper']) & (wide_ms1_points_df.scan >= precursor_cuboid_d['scan_lower']) & (wide_ms1_points_df.scan <= precursor_cuboid_d['scan_upper']) & (wide_ms1_points_df.retention_time_secs >= precursor_cuboid_d['ms1_rt_lower']) & (wide_ms1_points_df.retention_time_secs <= precursor_cuboid_d['ms1_rt_upper'])]

    # intensity descent
    raw_points_a = ms1_points_df[['mz','intensity']].to_numpy()
    peaks_a = intensity_descent(peaks_a=raw_points_a, peak_delta=MS1_PEAK_DELTA)

    # deconvolution
    ms1_peaks_l = list(map(tuple, peaks_a))
    deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, use_quick_charge=True, averagine=averagine.peptide, charge_range=(FEATURE_DETECTION_MIN_CHARGE,FEATURE_DETECTION_MAX_CHARGE), error_tolerance=5.0, scorer=scoring.MSDeconVFitter(MIN_SCORE_MS1_DECONVOLUTION_FEATURE), truncate_after=0.95)

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

    # load the ms2 data for the precursor
    if len(deconvolution_features_df) > 0:
        db_conn = sqlite3.connect(converted_db_name)
        ms2_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} and scan >= {} and scan <= {}".format(FRAME_TYPE_MS2, precursor_cuboid_d['ms2_rt_lower'], precursor_cuboid_d['ms2_rt_upper'], precursor_cuboid_d['scan_lower'], precursor_cuboid_d['scan_upper']), db_conn)
        db_conn.close()

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
        feature_d['envelope'] = json.dumps([tuple(e) for e in row.envelope])
        feature_d['isotope_count'] = len(row.envelope)
        feature_d['deconvolution_score'] = row.score
        # from the precursor cuboid
        feature_d['precursor_cuboid_id'] = precursor_cuboid_d['precursor_cuboid_id']
        peak_d = determine_mono_peak_characteristics(centre_mz=row.envelope[0][0], ms1_raw_points_df=wide_ms1_points_df)
        if peak_d is not None:
            feature_d['scan_apex'] = peak_d['scan_apex']
            feature_d['scan_lower'] = peak_d['scan_lower']
            feature_d['scan_upper'] = peak_d['scan_upper']
            feature_d['rt_apex'] = peak_d['rt_apex']
            feature_d['rt_lower'] = peak_d['rt_lower']
            feature_d['rt_upper'] = peak_d['rt_upper']
            feature_d['mono_mz_without_saturated_points'] = peak_d['mono_mz_without_saturated_points']
            feature_d['envelope_mono_peak_three_sigma_intensity'] = calculate_peak_intensity(peak_characteristics=peak_d, raw_points=wide_ms1_points_df)
            adj_d = calculate_phr_adjusted_intensity(peak_characteristics=peak_d, envelope=row.envelope, raw_points=wide_ms1_points_df)
            feature_d['envelope_phr_adjusted_intensity'] = adj_d['mono_intensity']
            feature_d['envelope_phr_adjusted_intensity_flag'] = adj_d['mono_inferred']
            feature_d['envelope_phr_adjusted_isotopes'] = adj_d['adjusted_isotopes']

        # resolve the feature's fragment ions
        fragment_ions_l = resolve_fragment_ions(feature_d, ms2_points_df)
        feature_d['fragment_ions_l'] = json.dumps(fragment_ions_l)
        # assign a unique identifier to this feature
        feature_d['feature_id'] = generate_feature_id(precursor_cuboid_d['precursor_cuboid_id'], idx+1)
        # add it to the list
        feature_l.append(feature_d)
    features_df = pd.DataFrame(feature_l)

    print("found {} features for precursor {}".format(len(features_df), precursor_cuboid_d['precursor_cuboid_id']))
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

# map the pasef cuboid coordinates to the common form
def get_common_cuboid_definition_from_pasef(precursor_cuboid_row):
    d = {}
    d['precursor_cuboid_id'] = precursor_cuboid_row.precursor_id  # the precursor_id from the isolation window table
    d['mz_lower'] = precursor_cuboid_row.window_mz_lower
    d['mz_upper'] = precursor_cuboid_row.window_mz_upper
    d['wide_mz_lower'] = precursor_cuboid_row.wide_mz_lower
    d['wide_mz_upper'] = precursor_cuboid_row.wide_mz_upper
    d['scan_lower'] = precursor_cuboid_row.fe_scan_lower
    d['scan_upper'] = precursor_cuboid_row.fe_scan_upper
    d['wide_scan_lower'] = precursor_cuboid_row.wide_scan_lower
    d['wide_scan_upper'] = precursor_cuboid_row.wide_scan_upper
    d['ms1_rt_lower'] = precursor_cuboid_row.fe_ms1_rt_lower
    d['ms1_rt_upper'] = precursor_cuboid_row.fe_ms1_rt_upper
    d['wide_ms1_rt_lower'] = precursor_cuboid_row.wide_ms1_rt_lower
    d['wide_ms1_rt_upper'] = precursor_cuboid_row.wide_ms1_rt_upper
    d['ms2_rt_lower'] = precursor_cuboid_row.fe_ms2_rt_lower
    d['ms2_rt_upper'] = precursor_cuboid_row.fe_ms2_rt_upper
    return d

# map the 3did cuboid coordinates to the common form
def get_common_cuboid_definition_from_3did(precursor_cuboid_row):
    d = {}
    d['precursor_cuboid_id'] = precursor_cuboid_row.precursor_cuboid_id  # a unique identifier for the precursor cuboid
    d['mz_lower'] = precursor_cuboid_row.mz_lower
    d['mz_upper'] = precursor_cuboid_row.mz_upper
    d['wide_mz_lower'] = precursor_cuboid_row.mz_lower
    d['wide_mz_upper'] = precursor_cuboid_row.mz_upper
    d['scan_lower'] = precursor_cuboid_row.scan_lower
    d['scan_upper'] = precursor_cuboid_row.scan_upper
    d['wide_scan_lower'] = precursor_cuboid_row.scan_lower
    d['wide_scan_upper'] = precursor_cuboid_row.scan_upper
    d['ms1_rt_lower'] = precursor_cuboid_row.rt_lower
    d['ms1_rt_upper'] = precursor_cuboid_row.rt_upper
    d['wide_ms1_rt_lower'] = precursor_cuboid_row.rt_lower
    d['wide_ms1_rt_upper'] = precursor_cuboid_row.rt_upper
    d['ms2_rt_lower'] = None
    d['ms2_rt_upper'] = None
    return d

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
parser.add_argument('-v','--visualise', action='store_true', help='Generate data for visualisation of the feature detection.')
parser.add_argument('-drd','--do_not_remove_duplicates', action='store_true', help='Do not remove duplicated features.')
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

# set up the indexes
print('setting up indexes on {}'.format(CONVERTED_DATABASE_NAME))
create_indexes(db_file_name=CONVERTED_DATABASE_NAME)

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
MIN_SCORE_MS1_DECONVOLUTION_FEATURE = cfg.getfloat('ms1', 'MIN_SCORE_MS1_DECONVOLUTION_FEATURE')
MIN_SCORE_MS2_DECONVOLUTION_FEATURE = cfg.getfloat('ms2', 'MIN_SCORE_MS2_DECONVOLUTION_FEATURE')
FEATURE_DETECTION_MIN_CHARGE = cfg.getint('ms1', 'FEATURE_DETECTION_MIN_CHARGE')
FEATURE_DETECTION_MAX_CHARGE = cfg.getint('ms1', 'FEATURE_DETECTION_MAX_CHARGE')
DUP_MZ_TOLERANCE_PPM = cfg.getint('ms1', 'DUP_MZ_TOLERANCE_PPM')
DUP_SCAN_TOLERANCE = cfg.getint('ms1', 'DUP_SCAN_TOLERANCE')
DUP_RT_TOLERANCE = cfg.getint('ms1', 'DUP_RT_TOLERANCE')
SATURATION_INTENSITY = cfg.getint('common', 'SATURATION_INTENSITY')

# input cuboids
CUBOIDS_DIR = "{}/precursor-cuboids-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-{}.pkl'.format(CUBOIDS_DIR, args.experiment_name, args.run_name, args.precursor_definition_method)

# output features
FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
FEATURES_FILE = '{}/exp-{}-run-{}-features-{}.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name, args.precursor_definition_method)

# check the cuboids file
if not os.path.isfile(CUBOIDS_FILE):
    print("The cuboids file is required but doesn't exist: {}".format(CUBOIDS_FILE))
    sys.exit(1)

# load the precursor cuboids
with open(CUBOIDS_FILE, 'rb') as handle:
    d = pickle.load(handle)
precursor_cuboids_df = d['coords_df']
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
if args.precursor_definition_method == 'pasef':
    features_l = ray.get([detect_features.remote(precursor_cuboid_d=get_common_cuboid_definition_from_pasef(row), converted_db_name=CONVERTED_DATABASE_NAME) for row in precursor_cuboids_df.itertuples()])
elif args.precursor_definition_method == '3did':
    features_l = ray.get([detect_features.remote(precursor_cuboid_d=get_common_cuboid_definition_from_3did(row), converted_db_name=CONVERTED_DATABASE_NAME) for row in precursor_cuboids_df.itertuples()])

# join the list of dataframes into a single dataframe
features_df = pd.concat(features_l, axis=0, sort=False)

# add the run name
features_df['run_name'] = args.run_name

# write out all the features
print("writing {} features to {}".format(len(features_df), FEATURES_FILE))
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
content_d = {'features_df':features_df, 'metadata':info}
with open(FEATURES_FILE, 'wb') as handle:
    pickle.dump(content_d, handle)

# de-dup the features
if not args.do_not_remove_duplicates:
    print('removing duplicates from {}'.format(FEATURES_FILE))
    dedup_start_run = time.time()

    # set up dup definitions
    MZ_TOLERANCE_PERCENT = DUP_MZ_TOLERANCE_PPM * 10**-4
    features_df['dup_mz'] = features_df['monoisotopic_mz']  # shorthand to reduce verbosity
    features_df['dup_mz_ppm_tolerance'] = features_df.dup_mz * MZ_TOLERANCE_PERCENT / 100
    features_df['dup_mz_lower'] = features_df.dup_mz - features_df.dup_mz_ppm_tolerance
    features_df['dup_mz_upper'] = features_df.dup_mz + features_df.dup_mz_ppm_tolerance
    features_df['dup_scan_lower'] = features_df.scan_apex - DUP_SCAN_TOLERANCE
    features_df['dup_scan_upper'] = features_df.scan_apex + DUP_SCAN_TOLERANCE
    features_df['dup_rt_lower'] = features_df.rt_apex - DUP_RT_TOLERANCE
    features_df['dup_rt_upper'] = features_df.rt_apex + DUP_RT_TOLERANCE

    # remove these after we're finished
    columns_to_drop_l = []
    columns_to_drop_l.append('dup_mz')
    columns_to_drop_l.append('dup_mz_ppm_tolerance')
    columns_to_drop_l.append('dup_mz_lower')
    columns_to_drop_l.append('dup_mz_upper')
    columns_to_drop_l.append('dup_scan_lower')
    columns_to_drop_l.append('dup_scan_upper')
    columns_to_drop_l.append('dup_rt_lower')
    columns_to_drop_l.append('dup_rt_upper')

    # see if any detections have a duplicate - if so, find the dup with the highest intensity and keep it
    keep_l = []
    for row in features_df.itertuples():
        dup_df = features_df[(features_df.dup_mz > row.dup_mz_lower) & (features_df.dup_mz < row.dup_mz_upper) & (features_df.scan_apex > row.dup_scan_lower) & (features_df.scan_apex < row.dup_scan_upper) & (features_df.rt_apex > row.dup_rt_lower) & (features_df.rt_apex < row.dup_rt_upper)].copy()
        # group the dups by charge - take the most intense for each charge
        for group_name,group_df in dup_df.groupby(['charge'], as_index=False):
            keep_l.append(group_df.iloc[0].feature_id)

    # remove any features that are not in the keep list
    dedup_df = features_df[features_df.feature_id.isin(keep_l)].copy()

    number_of_dups = len(features_df)-len(dedup_df)
    print('removed {} duplicates ({}% of the original detections)'.format(number_of_dups, round(number_of_dups/len(features_df)*100)))
    print('there are {} detected de-duplicated features'.format(len(dedup_df)))

    # remove the columns we added earlier
    dedup_df.drop(columns_to_drop_l, axis=1, inplace=True)

    FEATURES_DEDUP_FILE = '{}/exp-{}-run-{}-features-{}-dedup.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name, args.precursor_definition_method)

    # write out all the features
    print("writing {} de-duped features to {}".format(len(dedup_df), FEATURES_DEDUP_FILE))
    info.append(('dedup_running_time',round(time.time()-dedup_start_run,1)))
    content_d = {'features_df':dedup_df, 'metadata':info}
    with open(FEATURES_DEDUP_FILE, 'wb') as handle:
        pickle.dump(content_d, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
