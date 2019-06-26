import sqlite3
import pandas as pd
import numpy as np
import sys
import peakutils
from ms_deisotope import deconvolute_peaks, averagine, scoring
from ms_deisotope.deconvolution import peak_retention_strategy
from ms_peak_picker import simple_peak
from pyteomics import mgf
import os.path
import argparse
import ray
import time
import pickle

PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.
DELTA_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.

parser = argparse.ArgumentParser(description='Extract ms1 features from PASEF isolation windows.')
parser.add_argument('-cdbb','--converted_database_base', type=str, help='base path to the converted database.', required=True)
parser.add_argument('-rdbb','--raw_database_base', type=str, help='base path to the raw database.', required=True)
parser.add_argument('-mqtb','--maxquant_text_base', type=str, help='base path to MaxQuant text directory.', required=True)
parser.add_argument('-mgf','--mgf_filename', type=str, help='File name of the MGF to be generated.', required=True)
# parameters
parser.add_argument('-rtl','--rt_lower', type=float, help='The lower limit of retention time (secs).', required=True)
parser.add_argument('-rtu','--rt_upper', type=float, help='The upper limit of retention time (secs).', required=True)
parser.add_argument('-rtpw','--rt_base_peak_width_secs', type=float, default=30.0, help='How broad to look in RT for the peak apex (secs).', required=False)
parser.add_argument('-rtfe','--rt_fragment_event_delta_frames', type=int, default=2, help='Number of ms1 frames to look either side of the fragmentation event.', required=False)
parser.add_argument('-ms1ce','--ms1_collision_energy', type=float, default=10.0, help='Collision energy used in ms1 frames.', required=False)
parser.add_argument('-ms1bw','--ms1_bin_width', type=float, default=0.00001, help='Width of ms1 bins, in Thomsons.', required=False)
parser.add_argument('-ms2bw','--ms2_bin_width', type=float, default=0.001, help='Width of ms2 bins, in Thomsons.', required=False)
parser.add_argument('-ms1dt','--ms1_peak_delta', type=float, default=0.1, help='How far either side of a peak in ms1 to include when calculating its centroid and intensity, in Thomsons.', required=False)
parser.add_argument('-ms2dt','--ms2_peak_delta', type=float, default=0.01, help='How far either side of a peak in ms2 to include when calculating its centroid and intensity, in Thomsons.', required=False)
parser.add_argument('-ms1ddmz','--ms1_dedup_mz_tolerance_ppm', type=float, default=5.0, help='Tolerance in m/z ppm for de-duping features in ms1.', required=False)
parser.add_argument('-ms2l','--ms2_lower', type=float, default=90.0, help='Lower limit of m/z range in ms2.', required=False)
parser.add_argument('-ms2u','--ms2_upper', type=float, default=1750.0, help='Upper limit of m/z range in ms2.', required=False)
parser.add_argument('-nstddevmz','--number_of_std_dev_mz', type=float, default=3.0, help='Number of standard deviations to look either side of the expected isotopic spacing.', required=False)
parser.add_argument('-ms1phre','--max_ms1_peak_height_ratio_error', type=float, default=0.3, help='Maximum error for a feature\'s monoisotopic peak height ratio.', required=False)

# commands
parser.add_argument('-npbms2','--new_prebin_ms2', action='store_true', help='Create a new pre-bin file for ms2 frames.')
parser.add_argument('-pbms2fn','--pre_binned_ms2_filename', type=str, default='./pre_binned_ms2.pkl', help='File containing previously pre-binned ms2 frames.', required=False)
#
parser.add_argument('-nms1f','--new_ms1_features', action='store_true', help='Create a new ms1 features file.')
parser.add_argument('-ms1ffn','--ms1_features_filename', type=str, default='./ms1_df.pkl', help='File containing ms1 features.', required=False)
#
parser.add_argument('-cmms1','--check_ms1_mono_peak', action='store_true', help='Check the monoisotopic peak for each feature, moving it if necessary.')
parser.add_argument('-cmms1fn','--checked_ms1_mono_peak_filename', type=str, default='./checked_ms1_features.pkl', help='File containing mono-checked features.', required=False)
#
parser.add_argument('-nddms1','--new_dedup_ms1_features', action='store_true', help='Create a new de-duped ms1 features file.')
parser.add_argument('-ddms1fn','--dedup_ms1_filename', type=str, default='./ms1_deduped_df.pkl', help='File containing de-duped ms1 features.', required=False)
#
parser.add_argument('-nmgf','--new_mgf_spectra', action='store_true', help='Create a new mgf spectra file.')
parser.add_argument('-mgffn','--mgf_spectra_filename', type=str, default='./mgf_spectra.pkl', help='File containing mgf spectra.', required=False)
# modes
parser.add_argument('-cl','--cluster_mode', action='store_true', help='Run on a cluster.')
parser.add_argument('-tm','--test_mode', action='store_true', help='A small subset of the data for testing purposes.')
args = parser.parse_args()

# initialise Ray
if not ray.is_initialized():
    if args.cluster_mode:
        ray.init(redis_address="localhost:6379")
    else:
        ray.init(object_store_memory=100000000000)

start_run = time.time()

# Store the arguments as metadata for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

print("{} info: {}".format(parser.prog, info))

RAW_DATABASE_NAME = "{}/analysis.tdf".format(args.raw_database_base)
if not os.path.isfile(RAW_DATABASE_NAME):
    print("The raw database doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)

CONVERTED_DATABASE_NAME = '{}/HeLa_20KInt.sqlite'.format(args.converted_database_base)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

if args.new_prebin_ms2:
    if os.path.isfile(args.pre_binned_ms2_filename):
        os.remove(args.pre_binned_ms2_filename)
else:
    if not os.path.isfile(args.pre_binned_ms2_filename):
        print("The pre-binned ms2 file is required but doesn't exist: {}".format(args.pre_binned_ms2_filename))
        sys.exit(1)

if args.new_ms1_features:
    if os.path.isfile(args.ms1_features_filename):
        os.remove(args.ms1_features_filename)
else:
    if not os.path.isfile(args.ms1_features_filename):
        print("The ms1 features file is required but doesn't exist: {}".format(args.ms1_features_filename))
        sys.exit(1)

if args.new_dedup_ms1_features:
    if os.path.isfile(args.dedup_ms1_filename):
        os.remove(args.dedup_ms1_filename)
else:
    if not os.path.isfile(args.dedup_ms1_filename):
        print("The de-duped ms1 features file is required but doesn't exist: {}".format(args.dedup_ms1_filename))
        sys.exit(1)

if args.new_mgf_spectra:
    if os.path.isfile(args.mgf_spectra_filename):
        os.remove(args.mgf_spectra_filename)
else:
    if not os.path.isfile(args.mgf_spectra_filename):
        print("The mgf spectra file is required but doesn't exist: {}".format(args.mgf_spectra_filename))
        sys.exit(1)

ALLPEPTIDES_FILENAME = '{}/allPeptides.txt'.format(args.maxquant_text_base)
if not os.path.isfile(ALLPEPTIDES_FILENAME):
    print("The allPeptides file is required but doesn't exist: {}".format(ALLPEPTIDES_FILENAME))
    sys.exit(1)

PASEF_MSMS_SCANS_FILENAME = '{}/pasefMsmsScans.txt'.format(args.maxquant_text_base)
if not os.path.isfile(PASEF_MSMS_SCANS_FILENAME):
    print("The pasef msms scans file is required but doesn't exist: {}".format(PASEF_MSMS_SCANS_FILENAME))
    sys.exit(1)

# make sure the right indexes are created in the source database
print("Setting up indexes on {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
src_c = db_conn.cursor()
src_c.execute("create index if not exists idx_pasef_frames_1 on frames (frame_id, mz, intensity)")
src_c.execute("create index if not exists idx_pasef_frames_2 on frames (frame_id, mz, scan, retention_time_secs)")
db_conn.close()

# load the pasef msms scans
pasef_msms_scans_df = pd.read_csv(PASEF_MSMS_SCANS_FILENAME, sep='\t')

# load the allPeptides
allpeptides_df = pd.read_csv(ALLPEPTIDES_FILENAME, sep='\t')
allpeptides_df.rename(columns={'Number of isotopic peaks':'isotope_count', 'm/z':'mz', 'Number of data points':'number_data_points', 'Intensity':'intensity', 'Ion mobility index':'scan', 'Ion mobility index length':'scan_length', 'Ion mobility index length (FWHM)':'scan_length_fwhm', 'Retention time':'rt', 'Retention length':'rt_length', 'Retention length (FWHM)':'rt_length_fwhm', 'Charge':'charge_state', 'Number of pasef MS/MS':'number_pasef_ms2_ids', 'Pasef MS/MS IDs':'pasef_msms_ids', 'MS/MS scan number':'msms_scan_number', 'Isotope correlation':'isotope_correlation'}, inplace=True)
allpeptides_df = allpeptides_df[allpeptides_df.intensity.notnull() & allpeptides_df.pasef_msms_ids.notnull()].copy()
allpeptides_df.msms_scan_number = allpeptides_df.msms_scan_number.apply(lambda x: int(x))
allpeptides_df['pasef_msms_ids_list'] = allpeptides_df.pasef_msms_ids.str.split(";").apply(lambda x: [int(i) for i in x])

print("reading converted raw data from {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {} order by retention_time_secs".format(args.rt_lower, args.rt_upper, args.ms1_collision_energy), db_conn)
ms2_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy <> {} order by retention_time_secs".format(args.rt_lower, args.rt_upper, args.ms1_collision_energy), db_conn)
db_conn.close()

# get all the isolation windows
db_conn = sqlite3.connect(RAW_DATABASE_NAME)
isolation_window_df = pd.read_sql_query("select * from PasefFrameMsMsInfo", db_conn)
db_conn.close()

# add-in the retention time for the isolation windows and filter out the windows not in range
isolation_window_df = pd.merge(isolation_window_df, ms2_frame_properties_df, how='left', left_on=['Frame'], right_on=['frame_id'])
isolation_window_df.drop(['CollisionEnergy'], axis=1, inplace=True)
isolation_window_df.dropna(subset=['retention_time_secs'], inplace=True)
isolation_window_df['mz_lower'] = isolation_window_df.IsolationMz - (isolation_window_df.IsolationWidth / 2) - 0.7
isolation_window_df['mz_upper'] = isolation_window_df.IsolationMz + (isolation_window_df.IsolationWidth / 2) + 0.7
# filter out isolation windows that don't fit in the database subset we have loaded
isolation_window_df = isolation_window_df[(isolation_window_df.retention_time_secs >= (args.rt_lower - args.rt_base_peak_width_secs)) & (isolation_window_df.retention_time_secs <= (args.rt_upper + args.rt_base_peak_width_secs))]
print("loaded {} isolation windows from {}".format(len(isolation_window_df), RAW_DATABASE_NAME))

def bin_ms2_frames():
    # get the raw points for all ms2 frames
    ms2_frame_ids = tuple(ms2_frame_properties_df.frame_id)
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    ms2_raw_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} and mz >= {} and mz <= {} and intensity > 0".format(ms2_frame_ids, args.ms2_lower, args.ms2_upper), db_conn)
    db_conn.close()

    # arrange the points into bins
    ms2_bins = np.arange(start=args.ms2_lower, stop=args.ms2_upper+args.ms2_bin_width, step=args.ms2_bin_width)  # go slightly wider to accomodate the maximum value
    ms2_raw_points_df['bin_idx'] = np.digitize(ms2_raw_points_df.mz, ms2_bins).astype(int)
    return ms2_raw_points_df

def find_ms1_frames_for_ms2_frame_range(ms2_frame_ids, number_either_side):
    lower_ms2_frame = min(ms2_frame_ids)
    upper_ms2_frame = max(ms2_frame_ids)
    # calculate the deltas
    ms1_frame_properties_df['delta_l'] = abs(ms1_frame_properties_df.frame_id - lower_ms2_frame)
    ms1_frame_properties_df['delta_u'] = abs(ms1_frame_properties_df.frame_id - upper_ms2_frame)
    # find the closest
    closest_index_lower = ms1_frame_properties_df.delta_l.idxmin()
    closest_index_upper = ms1_frame_properties_df.delta_u.idxmin()
    # get the ms1 frames in the range
    ms1_frame_ids = tuple(ms1_frame_properties_df.loc[closest_index_lower-number_either_side:closest_index_upper+number_either_side].frame_id)
    # clean up
    ms1_frame_properties_df.drop('delta_l', axis=1, inplace=True)
    ms1_frame_properties_df.drop('delta_u', axis=1, inplace=True)
    return ms1_frame_ids

@ray.remote
def find_features(group_number, group_df):
    # find the ms1 features in this isolation window
    ms1_characteristics_l = []

    window = group_df.iloc[0]
    window_mz_lower = window.mz_lower
    window_mz_upper = window.mz_upper
    scan_width = int(window.ScanNumEnd - window.ScanNumBegin)
    wide_scan_lower = int(window.ScanNumBegin - scan_width)
    wide_scan_upper = int(window.ScanNumEnd + scan_width)
    fe_scan_lower = int(window.ScanNumBegin)
    fe_scan_upper = int(window.ScanNumEnd)
    precursor_id = int(window.Precursor)

    wide_rt_lower = group_df.retention_time_secs.min() - args.rt_base_peak_width_secs
    wide_rt_upper = group_df.retention_time_secs.max() + args.rt_base_peak_width_secs

    # get the ms1 frame IDs for the range of this precursor's ms2 frames
    isolation_window_ms1_frame_ids = find_ms1_frames_for_ms2_frame_range(ms2_frame_ids=list(group_df.Frame), number_either_side=args.rt_fragment_event_delta_frames)
    print("found ms1 frames {} for ms2 frames {}".format(isolation_window_ms1_frame_ids, list(group_df.Frame)))

    ms1_frame_ids = tuple(ms1_frame_properties_df.frame_id)

    # load the cube's raw ms1 points
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    ms1_raw_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {} and frame_id in {}".format(window_mz_lower, window_mz_upper, wide_scan_lower, wide_scan_upper, wide_rt_lower, wide_rt_upper, ms1_frame_ids), db_conn)
    db_conn.close()

    # get the raw points constrained to the fragmentation event's extent in ms1 frames
    fe_raw_points_df = ms1_raw_points_df[ms1_raw_points_df.isin(isolation_window_ms1_frame_ids)]

    ms1_bins = np.arange(start=window_mz_lower, stop=window_mz_upper+args.ms1_bin_width, step=args.ms1_bin_width)  # go slightly wider to accomodate the maximum value
    MZ_BIN_COUNT = len(ms1_bins)

    # initialise an array of lists to hold the m/z and intensity values allocated to each bin
    ms1_mz_values_array = np.empty(MZ_BIN_COUNT, dtype=np.object)
    for idx in range(MZ_BIN_COUNT):
        ms1_mz_values_array[idx] = []

    # gather the m/z values into bins
    for r in zip(fe_raw_points_df.mz, fe_raw_points_df.intensity):
        mz = r[0]
        intensity = int(r[1])
        if (mz >= window_mz_lower) and (mz <= window_mz_upper): # it should already but just to be sure
            mz_array_idx = int(np.digitize(mz, ms1_bins)) # in which bin should this mz go
            ms1_mz_values_array[mz_array_idx].append((mz, intensity))

    # compute the intensity-weighted m/z centroid and the summed intensity of the bins
    binned_ms1_l = []
    for bin_idx in range(MZ_BIN_COUNT):
        if len(ms1_mz_values_array[bin_idx]) > 0:
            mz_values_for_bin = np.array([ list[0] for list in ms1_mz_values_array[bin_idx]])
            intensity_values_for_bin = np.array([ list[1] for list in ms1_mz_values_array[bin_idx]]).astype(int)
            mz_centroid = peakutils.centroid(mz_values_for_bin, intensity_values_for_bin)
            summed_intensity = intensity_values_for_bin.sum()
            binned_ms1_l.append((mz_centroid,summed_intensity))

    binned_ms1_df = pd.DataFrame(binned_ms1_l, columns=['mz_centroid','summed_intensity'])
    raw_scratch_df = binned_ms1_df.copy() # take a copy because we're going to delete stuff
    if args.test_mode:
        binned_ms1_df.to_csv('./ms1-peaks-window-{}-before-intensity-descent.csv'.format(window_number), index=False, header=True)

    ms1_peaks_l = []
    while len(raw_scratch_df) > 0:
        # find the most intense point
        peak_df = raw_scratch_df.loc[raw_scratch_df.summed_intensity.idxmax()]
        peak_mz = peak_df.mz_centroid
        peak_mz_lower = peak_mz - args.ms1_peak_delta
        peak_mz_upper = peak_mz + args.ms1_peak_delta

        # get all the raw points within this m/z region
        peak_raw_points_df = raw_scratch_df[(raw_scratch_df.mz_centroid >= peak_mz_lower) & (raw_scratch_df.mz_centroid <= peak_mz_upper)]
        if len(peak_raw_points_df) > 0:
            mz_centroid = peakutils.centroid(peak_raw_points_df.mz_centroid, peak_raw_points_df.summed_intensity)
            summed_intensity = peak_raw_points_df.summed_intensity.sum()
            ms1_peaks_l.append((mz_centroid, summed_intensity))

            # remove the raw points assigned to this peak
            raw_scratch_df = raw_scratch_df[~raw_scratch_df.isin(peak_raw_points_df)].dropna(how = 'all')

    ms1_peaks_df = pd.DataFrame(ms1_peaks_l, columns=['mz','intensity'])
    if args.test_mode:
        ms1_peaks_df.to_csv('./ms1-peaks-window-{}-after-intensity-descent.csv'.format(window_number), index=False, header=True)

    # see https://github.com/mobiusklein/ms_deisotope/blob/ee4b083ad7ab5f77722860ce2d6fdb751886271e/ms_deisotope/deconvolution/api.py#L17
    deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, use_quick_charge=True, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.95)

    ms1_deconvoluted_peaks_l = []
    for peak in deconvoluted_peaks:
        # discard a monoisotopic peak that has either of the first two peaks as placeholders (indicated by intensity of 1)
        if ((len(peak.envelope) >= 3) and (peak.envelope[0][1] > 1) and (peak.envelope[1][1] > 1)):
            mono_peak_mz = peak.mz
            mono_intensity = peak.intensity
            second_peak_mz = peak.envelope[1][0]
            ms1_deconvoluted_peaks_l.append((mono_peak_mz, second_peak_mz, mono_intensity, peak.score, peak.signal_to_noise, peak.charge, peak.envelope))

    ms1_deconvoluted_peaks_df = pd.DataFrame(ms1_deconvoluted_peaks_l, columns=['mono_mz','second_peak_mz','intensity','score','SN','charge','envelope'])

    # For each monoisotopic peak found, find its apex in RT and mobility
    print("window {}, processing {} monoisotopics".format(window_number, len(ms1_deconvoluted_peaks_df)))
    for monoisotopic_idx in range(len(ms1_deconvoluted_peaks_df)):
        feature_monoisotopic_mz = ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].mono_mz
        feature_intensity = int(ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].intensity)
        second_peak_mz = ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].second_peak_mz
        feature_charge = int(ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].charge)
        feature_envelope = ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].envelope

        # Get the raw points for the monoisotopic peak (constrained by the fragmentation event)
        MZ_TOLERANCE_PPM = 20
        MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4

        monoisotopic_mz_ppm_tolerance = feature_monoisotopic_mz * MZ_TOLERANCE_PERCENT / 100
        monoisotopic_mz_lower = feature_monoisotopic_mz - monoisotopic_mz_ppm_tolerance
        monoisotopic_mz_upper = feature_monoisotopic_mz + monoisotopic_mz_ppm_tolerance

        monoisotopic_raw_points_df = fe_raw_points_df[(fe_raw_points_df.mz >= monoisotopic_mz_lower) & (fe_raw_points_df.mz <= monoisotopic_mz_upper)]
        if len(monoisotopic_raw_points_df) > 0:
            # collapsing the monoisotopic's summed points onto the mobility dimension
            scan_df = monoisotopic_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()

            mobility_curve_fit = False
            try:
                guassian_params = peakutils.peak.gaussian_fit(scan_df.scan, scan_df.intensity, center_only=False)
                scan_apex = guassian_params[1]
                scan_side_width = 2 * abs(guassian_params[2])  # number of standard deviations either side of the apex
                scan_lower = scan_apex - scan_side_width
                scan_upper = scan_apex + scan_side_width
                if (scan_apex >= wide_scan_lower) and (scan_apex <= wide_scan_upper):
                    mobility_curve_fit = True
            except:
                pass

            # if we couldn't fit a curve to the mobility dimension, take the intensity-weighted centroid
            if not mobility_curve_fit:
                scan_apex = peakutils.centroid(scan_df.scan, scan_df.intensity)
                scan_lower = wide_scan_lower
                scan_upper = wide_scan_upper

            # In the RT dimension, look wider to find the apex
            wide_rt_monoisotopic_raw_points_df = ms1_raw_points_df[(ms1_raw_points_df.mz >= monoisotopic_mz_lower) & (ms1_raw_points_df.mz <= monoisotopic_mz_upper)]
            rt_df = wide_rt_monoisotopic_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
            rt_curve_fit = False
            try:
                guassian_params = peakutils.peak.gaussian_fit(rt_df.retention_time_secs, rt_df.intensity, center_only=False)
                rt_apex = guassian_params[1]
                rt_side_width = 3 * abs(guassian_params[2])  # number of standard deviations either side of the apex
                rt_lower = rt_apex - rt_side_width
                rt_upper = rt_apex + rt_side_width
                if (rt_apex >= wide_rt_lower) and (rt_apex <= wide_rt_upper):
                    rt_curve_fit = True
            except:
                pass

            # if we couldn't fit a curve to the RT dimension, take the intensity-weighted centroid
            if not rt_curve_fit:
                rt_apex = peakutils.centroid(rt_df.retention_time_secs, rt_df.intensity)
                rt_lower = wide_rt_lower
                rt_upper = wide_rt_upper

            # find the isolation windows overlapping the feature's mono or second peak, plus scan and RT
            indexes = isolation_window_df.index[
                                        (((isolation_window_df.mz_upper >= feature_monoisotopic_mz) & (isolation_window_df.mz_lower <= feature_monoisotopic_mz)) |
                                        ((isolation_window_df.mz_upper >= second_peak_mz) & (isolation_window_df.mz_lower <= second_peak_mz))) &
                                        (isolation_window_df.ScanNumEnd >= scan_apex) &
                                        (isolation_window_df.ScanNumBegin <= scan_apex) &
                                        (isolation_window_df.retention_time_secs >= rt_lower) &
                                        (isolation_window_df.retention_time_secs <= rt_upper)
                                   ]
            isolation_windows_overlapping_feature_df = isolation_window_df.loc[indexes]

            if len(isolation_windows_overlapping_feature_df) > 0:
                ms2_frames = list(isolation_windows_overlapping_feature_df.Frame)
                ms2_scan_ranges = [tuple(x) for x in isolation_windows_overlapping_feature_df[['ScanNumBegin','ScanNumEnd']].values]
                ms1_characteristics_l.append((feature_monoisotopic_mz, feature_charge, feature_intensity, round(scan_apex,2), mobility_curve_fit, round(scan_lower,2), round(scan_upper,2), round(rt_apex,2), rt_curve_fit, round(rt_lower,2), round(rt_upper,2), precursor_id, ms2_frames, ms2_scan_ranges, feature_envelope))

    ms1_characteristics_df = pd.DataFrame(ms1_characteristics_l, columns=['monoisotopic_mz', 'charge', 'intensity', 'scan_apex', 'scan_curve_fit', 'scan_lower', 'scan_upper', 'rt_apex', 'rt_curve_fit', 'rt_lower', 'rt_upper', 'precursor_id', 'ms2_frames', 'ms2_scan_ranges','envelope'])
    return ms1_characteristics_df

def remove_ms1_duplicates(ms1_features_df):
    scratch_df = ms1_features_df.copy() # take a copy because we're going to delete stuff
    scratch_df.sort_values(by=['intensity'], ascending=False, inplace=True)
    MZ_TOLERANCE_PERCENT = args.ms1_dedup_mz_tolerance_ppm * 10**-4
    ms1_features_l = []
    while len(scratch_df) > 0:
        scratch_df.reset_index(drop=True, inplace=True)
        # take the first row
        row = scratch_df.iloc[0]
        mz = row.monoisotopic_mz
        scan = row.scan_apex
        rt = row.rt_apex

        # calculate the matching bounds
        mz_ppm_tolerance = mz * MZ_TOLERANCE_PERCENT / 100
        mz_lower = mz - mz_ppm_tolerance
        mz_upper = mz + mz_ppm_tolerance
        scan_lower = row.scan_lower
        scan_upper = row.scan_upper
        rt_lower = row.rt_lower
        rt_upper = row.rt_upper

        # find the matches within these tolerances
        cond_1 = (scratch_df.monoisotopic_mz >= mz_lower) & (scratch_df.monoisotopic_mz <= mz_upper) & (scratch_df.scan_apex >= scan_lower) & (scratch_df.scan_apex <= scan_upper) & (scratch_df.rt_apex >= rt_lower) & (scratch_df.rt_apex <= rt_upper)
        matching_rows = scratch_df.loc[cond_1, :].copy()
        # of those, find the most intense
        cond_2 = (matching_rows.intensity == matching_rows.intensity.max())
        most_intense_row = matching_rows.loc[cond_2, :].copy()
        most_intense_row['duplicates'] = len(matching_rows)
        # take the union of the ms2 frames
        # if len(matching_rows) > 1:
        #     most_intense_row.iloc[0].ms2_frames = list(np.unique(matching_rows.ms2_frames.to_list()))
        # add it to the list
        ms1_features_l.append(tuple(most_intense_row.iloc[0]))
        # drop the duplicates
        scratch_df.drop(matching_rows.index, inplace=True)

    cols = scratch_df.columns.append(pd.Index(['duplicates']))
    ms1_deduped_df = pd.DataFrame(ms1_features_l, columns=cols)
    ms1_deduped_df.sort_values(by=['intensity'], ascending=False, inplace=True)
    ms1_deduped_df["feature_id"] = np.arange(start=1, stop=len(ms1_deduped_df)+1)
    return ms1_deduped_df

INSTRUMENT_RESOLUTION = 40000.0

# The FWHM is the m/z / instrument resolution. Std dev is FWHM / 2.35482. See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
def standard_deviation(mz):
    FWHM = mz / INSTRUMENT_RESOLUTION
    return FWHM / 2.35482

# calculate the centroid, intensity of a bin
def calc_bin_centroid(bin_df):
    d = {}
    d['bin_idx'] = int(bin_df.iloc[0].bin_idx)
    d['mz_centroid'] = peakutils.centroid(bin_df.mz, bin_df.intensity)
    d['summed_intensity'] = int(bin_df.intensity.sum())
    d['point_count'] = len(bin_df)
    return pd.Series(d, index=d.keys())

# sum and centroid the bins
def find_ms1_peaks(binned_ms1_df):
    # calculate the bin centroid and summed intensity for the combined frames
    combined_ms1_df = binned_ms1_df.groupby(['bin_idx'], as_index=False).apply(calc_bin_centroid)
    # convert the bin attributes to the right type
    combined_ms1_df.summed_intensity = combined_ms1_df.summed_intensity.astype(int)
    combined_ms1_df.bin_idx = combined_ms1_df.bin_idx.astype(int)
    combined_ms1_df.point_count = combined_ms1_df.point_count.astype(int)
    return combined_ms1_df

# From "A Model-Based Method for the Prediction of the Isotopic Distribution of Peptides", Dirk Valkenborg,
# Ivy Jansen, and Tomasz Burzykowski, J Am Soc Mass Spectrom 2008, 19, 703–712

MAX_NUMBER_OF_SULPHUR_ATOMS = 3
MAX_NUMBER_OF_PREDICTED_RATIOS = 6

S0_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=object)
S0_r[1] = [-0.00142320578040, 0.53158267080224, 0.00572776591574, -0.00040226083326, -0.00007968737684]
S0_r[2] = [0.06258138406507, 0.24252967352808, 0.01729736525102, -0.00427641490976, 0.00038011211412]
S0_r[3] = [0.03092092306220, 0.22353930450345, -0.02630395501009, 0.00728183023772, -0.00073155573939]
S0_r[4] = [-0.02490747037406, 0.26363266501679, -0.07330346656184, 0.01876886839392, -0.00176688757979]
S0_r[5] = [-0.19423148776489, 0.45952477474223, -0.18163820209523, 0.04173579115885, -0.00355426505742]
S0_r[6] = [0.04574408690798, -0.05092121193598, 0.13874539944789, -0.04344815868749, 0.00449747222180]

S1_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=object)
S1_r[1] = [-0.01040584267474, 0.53121149663696, 0.00576913817747, -0.00039325152252, -0.00007954180489]
S1_r[2] = [0.37339166598255, -0.15814640001919, 0.24085046064819, -0.06068695741919, 0.00563606634601]
S1_r[3] = [0.06969331604484, 0.28154425636993, -0.08121643989151, 0.02372741957255, -0.00238998426027]
S1_r[4] = [0.04462649178239, 0.23204790123388, -0.06083969521863, 0.01564282892512, -0.00145145206815]
S1_r[5] = [-0.20727547407753, 0.53536509500863, -0.22521649838170, 0.05180965157326, -0.00439750995163]
S1_r[6] = [0.27169670700251, -0.37192045082925, 0.31939855191976, -0.08668833166842, 0.00822975581940]

S2_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=object)
S2_r[1] = [-0.01937823810470, 0.53084210514216, 0.00580573751882, -0.00038281138203, -0.00007958217070]
S2_r[2] = [0.68496829280011, -0.54558176102022, 0.44926662609767, -0.11154849560657, 0.01023294598884]
S2_r[3] = [0.04215807391059, 0.40434195078925, -0.15884974959493, 0.04319968814535, -0.00413693825139]
S2_r[4] = [0.14015578207913, 0.14407679007180, -0.01310480312503, 0.00362292256563, -0.00034189078786]
S2_r[5] = [-0.02549241716294, 0.32153542852101, -0.11409513283836, 0.02617210469576, -0.00221816103608]
S2_r[6] = [-0.14490868030324, 0.33629928307361, -0.08223564735018, 0.01023410734015, -0.00027717589598]

model_params = np.empty(MAX_NUMBER_OF_SULPHUR_ATOMS, dtype=object)
model_params[0] = S0_r
model_params[1] = S1_r
model_params[2] = S2_r

# Find the ratio of H(peak_number)/H(peak_number-1) for peak_number=1..6
# peak_number = 0 refers to the monoisotopic peak
# number_of_sulphur = number of sulphur atoms in the molecule
def peak_ratio(monoisotopic_mass, peak_number, number_of_sulphur):
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

# given a centre mz and a feature, calculate themz centroid and summed intensity from the raw points
def calculate_raw_peak_intensity_at_mz(centre_mz, feature):
    result = (-1, -1)

    mz_delta = standard_deviation(centre_mz) * args.number_of_std_dev_mz
    mz_lower = centre_mz - mz_delta
    mz_upper = centre_mz + mz_delta

    rt_lower = feature.rt_lower
    rt_upper = feature.rt_upper
    scan_lower = feature.scan_lower
    scan_upper = feature.scan_upper

    # find the ms1 frame ids for the feature's extent in RT
    ms1_frame_ids = tuple(ms1_frame_properties_df.frame_id)

    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    ms1_raw_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} and mz >= {} and mz <= {} and scan >= {} and scan <= {} and intensity > 0".format(ms1_frame_ids, mz_lower, mz_upper, scan_lower, scan_upper), db_conn)
    db_conn.close()

    if len(ms1_raw_points_df) > 0:
        # arrange the points into bins
        ms1_bins = np.arange(start=mz_lower, stop=mz_upper+args.ms1_bin_width, step=args.ms1_bin_width)  # go slightly wider to accomodate the maximum value
        ms1_raw_points_df['bin_idx'] = np.digitize(ms1_raw_points_df.mz, ms1_bins).astype(int)

        # find the peaks for each bin
        unresolved_peaks_df = find_ms1_peaks(ms1_raw_points_df)

        # centroid and sum all the bin peaks, as we are interested in a narrow band of m/z
        mz_centroid = peakutils.centroid(unresolved_peaks_df.mz_centroid, unresolved_peaks_df.summed_intensity)
        summed_intensity = unresolved_peaks_df.summed_intensity.sum()
        result = (mz_centroid, summed_intensity)

    return result

@ray.remote
def check_monoisotopic_peak(feature, idx, total):
    adjusted = False
    feature_d = feature.to_dict()
    candidate_phr_error = None

    # calculate the PHR error for peak 1 (first isotope) and peak 0 (what we think is the monoisotopic)
    observed_ratio = feature.envelope[1][1] / feature.envelope[0][1]
    monoisotopic_mass = feature.monoisotopic_mz * feature.charge
    expected_ratio = peak_ratio(monoisotopic_mass=monoisotopic_mass, peak_number=1, number_of_sulphur=0)
    original_phr_error = (observed_ratio - expected_ratio) / expected_ratio

    if abs(original_phr_error) > args.max_ms1_peak_height_ratio_error:
        print("feature {}/{}: PHR error {} - looking for the correct monoisotopic".format(idx+1,total,original_phr_error))
        # probably missed the monoisotopic - need to search for it
        expected_spacing_mz = DELTA_MZ / feature.charge
        centre_mz = feature.monoisotopic_mz - expected_spacing_mz
        candidate_mz_centroid, candidate_raw_intensity = calculate_raw_peak_intensity_at_mz(centre_mz, feature)
        if (candidate_mz_centroid != -1) and (candidate_raw_intensity != -1):
            # get the raw intensity for the original monoisotope so we can calculate an accurate ratio
            original_mz, original_raw_intensity = calculate_raw_peak_intensity_at_mz(feature.monoisotopic_mz, feature)
            if (original_mz != -1) and (original_raw_intensity != -1):
                candidate_ratio = original_raw_intensity / candidate_raw_intensity
                candidate_phr_error = (candidate_ratio - expected_ratio) / expected_ratio
                feature_d['candidate_phr_error'] = candidate_phr_error
                if (abs(candidate_phr_error) <= abs(original_phr_error)) and (abs(candidate_phr_error) <= args.max_ms1_peak_height_ratio_error):
                    # update the envelope with the adjusted monoisotopic peak
                    env_peak_0_intensity = feature.envelope[0][1]
                    new_env_peak_0_intensity = env_peak_0_intensity / candidate_ratio
                    updated_envelope = []
                    updated_envelope.append((candidate_mz_centroid, new_env_peak_0_intensity))
                    updated_envelope += feature.envelope
                    feature_d['envelope'] = updated_envelope
                    # the intensity is the sum of the envelope intensities
                    envelope_intensity = 0
                    for i in range(len(updated_envelope)):
                        envelope_intensity += updated_envelope[i][1]
                    feature_d['intensity'] = envelope_intensity
                    # the monoisotopic mz is the deisotoped mz
                    envelope_deisotoped_mzs = []
                    envelope_deisotoped_intensities = []
                    for isotope_number in range(len(updated_envelope)):
                        deisotoped_mz = updated_envelope[isotope_number][0] - (isotope_number * expected_spacing_mz)
                        envelope_deisotoped_mzs.append(deisotoped_mz)
                        envelope_deisotoped_intensities.append(updated_envelope[isotope_number][1])
                    # take the intensity-weighted centroid of the mzs in the list
                    feature_d['monoisotopic_mz'] = peakutils.centroid(np.array(envelope_deisotoped_mzs), np.array(envelope_deisotoped_intensities))
                    adjusted = True

    feature_d['mono_adjusted'] = adjusted
    feature_d['original_phr_error'] = original_phr_error
    feature_d['candidate_phr_error'] = candidate_phr_error
    feature_d['original_phr'] = observed_ratio
    return feature_d

def deconvolute_ms2_peaks_for_feature(feature_id, ms2_frame_id, binned_ms2_df):
    raw_scratch_df = binned_ms2_df.copy() # take a copy because we're going to delete stuff

    # do intensity descent to find the peaks
    ms2_peaks_l = []
    while len(raw_scratch_df) > 0:
        # find the most intense point
        peak_df = raw_scratch_df.loc[raw_scratch_df.summed_intensity.idxmax()]
        peak_mz = peak_df.mz_centroid
        peak_mz_lower = peak_mz - args.ms2_peak_delta
        peak_mz_upper = peak_mz + args.ms2_peak_delta

        # get all the raw points within this m/z region
        peak_raw_points_df = raw_scratch_df[(raw_scratch_df.mz_centroid >= peak_mz_lower) & (raw_scratch_df.mz_centroid <= peak_mz_upper)]
        if len(peak_raw_points_df) > 0:
            mz_centroid = peakutils.centroid(peak_raw_points_df.mz_centroid, peak_raw_points_df.summed_intensity)
            summed_intensity = peak_raw_points_df.summed_intensity.sum()
            ms2_peaks_l.append(simple_peak(mz=mz_centroid, intensity=summed_intensity))
            # remove the raw points assigned to this peak
            raw_scratch_df = raw_scratch_df[~raw_scratch_df.isin(peak_raw_points_df)].dropna(how = 'all')

    if args.test_mode:
        l = []
        for p in ms2_peaks_l:
            l.append((p.mz, p.intensity))
        ms2_peaks_df = pd.DataFrame(l, columns=['mz','intensity'])
        ms2_peaks_df.to_csv('./feature-{}-frame-{}-ms2-peaks-before-deconvolution.csv'.format(feature_id, ms2_frame_id), index=False, header=True)

    # deconvolute the peaks
    # see https://github.com/mobiusklein/ms_deisotope/blob/ee4b083ad7ab5f77722860ce2d6fdb751886271e/ms_deisotope/deconvolution/api.py#L17
    # see https://github.com/mobiusklein/ms_deisotope/blob/68e88e0ece3e76abdb2833ac82dca1800fe5bde1/ms_deisotope/deconvolution/peak_retention_strategy.py#L141
    ms2_deconvoluted_peaks, _ = deconvolute_peaks(ms2_peaks_l, use_quick_charge=True, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(minimum_score=8, mass_error_tolerance=0.1), error_tolerance=4e-5, truncate_after=0.8, retention_strategy=peak_retention_strategy.TopNRetentionStrategy(n_peaks=100, base_peak_coefficient=1e-6, max_mass=1800.0))

    ms2_deconvoluted_peaks_l = []
    for peak in ms2_deconvoluted_peaks:
        ms2_deconvoluted_peaks_l.append((round(peak.neutral_mass+PROTON_MASS, 4), int(peak.charge), peak.intensity, peak.score, peak.signal_to_noise))

    ms2_deconvoluted_peaks_df = pd.DataFrame(ms2_deconvoluted_peaks_l, columns=['mz','charge','intensity','score','SN'])
    if args.test_mode:
        ms2_deconvoluted_peaks_df.to_csv('./feature-{}-frame-{}-ms2-peaks-after-deconvolution.csv'.format(feature_id, ms2_frame_id), index=False, header=True)

    return ms2_deconvoluted_peaks_df

# sum and centroid the ms2 bins for this feature
def find_ms2_peaks_for_feature(feature_df, binned_ms2_for_feature_df):
    # calculate the bin centroid and summed intensity for the combined frames
    combined_ms2_df = binned_ms2_for_feature_df.groupby(['bin_idx'], as_index=False).apply(calc_bin_centroid)
    combined_ms2_df.summed_intensity = combined_ms2_df.summed_intensity.astype(int)
    combined_ms2_df.bin_idx = combined_ms2_df.bin_idx.astype(int)
    combined_ms2_df.point_count = combined_ms2_df.point_count.astype(int)
    return combined_ms2_df

def collate_spectra_for_feature(feature_df, ms2_deconvoluted_df):
    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    pairs_df = ms2_deconvoluted_df[['mz', 'intensity']].copy().sort_values(by=['mz'], ascending=True)
    spectrum = {}
    spectrum["m/z array"] = pairs_df.mz.values
    spectrum["intensity array"] = pairs_df.intensity.values
    params = {}
    params["TITLE"] = "RawFile: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(CONVERTED_DATABASE_NAME).split('.')[0], feature_df.charge, feature_df.intensity, feature_df.feature_id, round(feature_df.rt_apex,2))
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(feature_df.monoisotopic_mz,6), feature_df.intensity)
    params["CHARGE"] = "{}+".format(feature_df.charge)
    params["RTINSECONDS"] = "{}".format(round(feature_df.rt_apex,2))
    spectrum["params"] = params
    return spectrum

@ray.remote
def deconvolute_ms2(feature_df, binned_ms2_for_feature, idx, total):
    result = {}
    print("processing feature idx {} of {}".format(idx+1, total))
    ms2_frames_l = []
    for idx,ms2_frame_id in enumerate(feature_df.ms2_frames):
        # get the binned ms2 values for this frame and mobility range
        scan_lower = feature_df.ms2_scan_ranges[idx][0]
        scan_upper = feature_df.ms2_scan_ranges[idx][1]
        ms2_frames_l.append(binned_ms2_for_feature[(binned_ms2_for_feature.frame_id == ms2_frame_id) & (binned_ms2_for_feature.scan >= scan_lower) & (binned_ms2_for_feature.scan <= scan_upper)])

    # join the list of dataframes into a single dataframe
    ms2_frame_df = pd.concat(ms2_frames_l)

    # derive the spectra
    if len(ms2_frame_df) > 0:
        # detect peaks
        ms2_peaks_df = find_ms2_peaks_for_feature(feature_df, ms2_frame_df)
        if len(ms2_peaks_df) > 0:
            # deconvolve the peaks
            ms2_deconvoluted_df = deconvolute_ms2_peaks_for_feature(feature_df.feature_id, ms2_frame_id, ms2_peaks_df)
            if len(ms2_deconvoluted_df) >= 2:
                # package it up for the MGF
                result = collate_spectra_for_feature(feature_df, ms2_deconvoluted_df)
    return result


#########################################################
if args.new_ms1_features:
    # find ms1 features for each unique precursor ID
    print("finding ms1 features")
    if args.test_mode:
        isolation_window_df = isolation_window_df[:2]
    ms1_df_l = ray.get([find_features.remote(group_number=idx+1, group_df=group_df) for idx,group_df in isolation_window_df.groupby('Precursor')])
    ms1_df = pd.concat(ms1_df_l)  # combines a list of dataframes into a single dataframe
    ms1_df.to_pickle(args.ms1_features_filename)
    print("detected {} features".format(len(ms1_df)))
else:
    # load previously detected ms1 features
    print("loading ms1 features")
    ms1_df = pd.read_pickle(args.ms1_features_filename)
    print("loaded {} features".format(len(ms1_df)))

if args.check_ms1_mono_peak or args.new_ms1_features:
    print("checking ms1 monoisotopic peaks")
    ms1_df.reset_index(drop=True, inplace=True)
    if args.test_mode:
        ms1_df = ms1_df[:100]
    checked_features_l = ray.get([check_monoisotopic_peak.remote(feature=feature, idx=idx, total=len(ms1_df)) for idx,feature in ms1_df.iterrows()])
    checked_features_df = pd.DataFrame(checked_features_l)
    checked_features_df.to_pickle(args.checked_ms1_mono_peak_filename)
else:
    # load previously checked monoisotopic peaks
    print("loading checked ms1 monoisotopic peaks")
    checked_features_df = pd.read_pickle(args.checked_ms1_mono_peak_filename)

if args.new_dedup_ms1_features or args.check_ms1_mono_peak or args.new_ms1_features:
    # remove duplicates in ms1
    print("removing duplicates")
    ms1_deduped_df = remove_ms1_duplicates(checked_features_df)
    ms1_deduped_df.to_pickle(args.dedup_ms1_filename)
    print("removed {} duplicates - processing {} features".format(len(checked_features_df)-len(ms1_deduped_df), len(ms1_deduped_df)))
else:
    # load previously de-duped ms1 features
    print("loading de-duped ms1 features")
    ms1_deduped_df = pd.read_pickle(args.dedup_ms1_filename)
    print("loaded {} features".format(len(ms1_deduped_df)))

if args.new_prebin_ms2:
    # bin ms2 frames
    print("binning ms2 frames")
    binned_ms2_df = bin_ms2_frames()
    binned_ms2_df.to_pickle(args.pre_binned_ms2_filename)
    print("binned {} ms2 points".format(len(binned_ms2_df)))
else:
    # load previously binned ms2
    print("loading pre-binned ms2 frames")
    binned_ms2_df = pd.read_pickle(args.pre_binned_ms2_filename)
    print("loaded {} pre-binned ms2 points".format(len(binned_ms2_df)))

if args.new_mgf_spectra or args.check_ms1_mono_peak or args.new_dedup_ms1_features or args.new_ms1_features or args.new_prebin_ms2:
    # find ms2 peaks for each feature found in ms1, and collate the spectra for the MGF
    print("finding peaks in ms2 for each feature")
    ms1_deduped_df.reset_index(drop=True, inplace=True)
    if args.test_mode:
        ms1_deduped_df = ms1_deduped_df[ms1_deduped_df.feature_id == 822]
    # mgf_spectra_l is a list of dictionaries
    mgf_spectra_l = ray.get([deconvolute_ms2.remote(feature_df=feature_df, binned_ms2_for_feature=binned_ms2_df[binned_ms2_df.frame_id.isin(feature_df.ms2_frames)], idx=idx, total=len(ms1_deduped_df)) for idx,feature_df in ms1_deduped_df.iterrows()])
    # write out the results for analysis
    with open(args.mgf_spectra_filename, 'wb') as f:
        pickle.dump(mgf_spectra_l, f)
else:
    # load previously saved mgf spectra
    with open(args.mgf_spectra_filename, 'rb') as f:
        mgf_spectra_l = pickle.load(f)
    print("loaded the mgf spectra from {}".format(args.mgf_spectra_filename))

# generate the MGF for all the features
print("generating the MGF: {}".format(args.mgf_filename))
if os.path.isfile(args.mgf_filename):
    os.remove(args.mgf_filename)
mgf.write(output=args.mgf_filename, spectra=mgf_spectra_l)

stop_run = time.time()
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))
print("{} info: {}".format(parser.prog, info))

print("shutting down ray")
ray.shutdown()
