import sqlite3
import pandas as pd
import numpy as np
import sys
import peakutils
from ms_deisotope import deconvolute_peaks, averagine, scoring
from pyteomics import mgf
import os.path
import argparse
import ray
import time
import pickle

parser = argparse.ArgumentParser(description='Extract ms1 features from PASEF isolation windows.')
parser.add_argument('-cdbb','--converted_database_base', type=str, help='base path to the converted database.', required=True)
parser.add_argument('-rdbb','--raw_database_base', type=str, help='base path to the raw database.', required=True)
parser.add_argument('-mgf','--mgf_filename', type=str, help='File name of the MGF to be generated.', required=True)
parser.add_argument('-rtl','--rt_lower', type=float, help='The lower limit of retention time (secs).', required=True)
parser.add_argument('-rtu','--rt_upper', type=float, help='The upper limit of retention time (secs).', required=True)
parser.add_argument('-rtpw','--rt_base_peak_width_secs', type=float, default=30.0, help='How broad to look in RT for the peak apex (secs).', required=False)
parser.add_argument('-rtfe','--rt_fragment_event_delta_secs', type=float, default=3.5, help='How wide to look around the region of the fragmentation event (secs).', required=False)
parser.add_argument('-ms1ce','--ms1_collision_energy', type=float, default=10.0, help='Collision energy used in ms1 frames.', required=False)
parser.add_argument('-ms1bw','--ms1_bin_width', type=float, default=0.00001, help='Width of ms1 bins, in Thomsons.', required=False)
parser.add_argument('-ms2bw','--ms2_bin_width', type=float, default=0.001, help='Width of ms2 bins, in Thomsons.', required=False)
parser.add_argument('-ms1dt','--ms1_peak_delta', type=float, default=0.01, help='How far either side of a peak in ms1 to include when calculating its centroid and intensity, in Thomsons.', required=False)
parser.add_argument('-ms2dt','--ms2_peak_delta', type=float, default=0.01, help='How far either side of a peak in ms2 to include when calculating its centroid and intensity, in Thomsons.', required=False)
parser.add_argument('-ms2l','--ms2_lower', type=float, default=90.0, help='Lower limit of m/z range in ms2.', required=False)
parser.add_argument('-ms2u','--ms2_upper', type=float, default=1750.0, help='Upper limit of m/z range in ms2.', required=False)
parser.add_argument('-pbms2','--pre_binned_ms2_filename', type=str, default='./pre_binned_ms2.pkl', help='File containing previously pre-binned ms2 frames.', required=False)
parser.add_argument('-npbms2','--new_prebin_ms2', action='store_true', help='Create a new pre-bin file for ms2 frames.')
parser.add_argument('-ms1f','--ms1_features_filename', type=str, default='./ms1_df.pkl', help='File containing ms1 features.', required=False)
parser.add_argument('-nms1f','--new_ms1_features', action='store_true', help='Create a new ms1 features file.')
parser.add_argument('-ddms1','--dedup_ms1_filename', type=str, default='./ms1_deduped_df.pkl', help='File containing de-duped ms1 features.', required=False)
parser.add_argument('-nddms1','--new_dedup_ms1_features', action='store_true', help='Create a new de-duped ms1 features file.')
parser.add_argument('-cl','--cluster_mode', action='store_true', help='Run on a cluster.')
parser.add_argument('-tm','--test_mode', action='store_true', help='A small subset of the data for testing purposes.')
args = parser.parse_args()

# initialise Ray
if not ray.is_initialized():
    if args.cluster_mode:
        ray.init(redis_address="localhost:6379")
    else:
        ray.init()

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

PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

# ms1 duplicate tolerances
# +/- these amounts
MZ_TOLERANCE_PPM = 5
MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4
SCAN_TOLERANCE = 10
RT_TOLERANCE = 0.1

print("reading converted raw data from {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(args.rt_lower, args.rt_upper, args.ms1_collision_energy), db_conn)
ms2_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy <> {}".format(args.rt_lower, args.rt_upper, args.ms1_collision_energy), db_conn)
db_conn.close()

# get all the isolation windows
db_conn = sqlite3.connect(RAW_DATABASE_NAME)
isolation_window_df = pd.read_sql_query("select * from PasefFrameMsMsInfo", db_conn)
db_conn.close()

# add-in the retention time for the isolation windows and filter out the windows not in range
isolation_window_df = pd.merge(isolation_window_df, ms2_frame_properties_df, how='left', left_on=['Frame'], right_on=['frame_id'])
isolation_window_df.drop(['frame_id', 'CollisionEnergy'], axis=1, inplace=True)
isolation_window_df.dropna(subset=['retention_time_secs'], inplace=True)
isolation_window_df['mz_lower'] = isolation_window_df.IsolationMz - (isolation_window_df.IsolationWidth / 2) - 0.7
isolation_window_df['mz_upper'] = isolation_window_df.IsolationMz + (isolation_window_df.IsolationWidth / 2) + 0.7
isolation_window_df['wide_rt_lower'] = isolation_window_df.retention_time_secs - args.rt_base_peak_width_secs
isolation_window_df['wide_rt_upper'] = isolation_window_df.retention_time_secs + args.rt_base_peak_width_secs
isolation_window_df['fe_rt_lower'] = isolation_window_df.retention_time_secs - args.rt_fragment_event_delta_secs
isolation_window_df['fe_rt_upper'] = isolation_window_df.retention_time_secs + args.rt_fragment_event_delta_secs

# filter out isolation windows that don't fit in the database subset we have loaded
isolation_window_df = isolation_window_df[(isolation_window_df.wide_rt_lower >= args.rt_lower) & (isolation_window_df.wide_rt_upper <= args.rt_upper)]
if args.test_mode:
    isolation_window_df = isolation_window_df[:2]

print("There are {} precursor unique isolation windows.".format(isolation_window_df.Precursor.nunique()))

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

@ray.remote
def find_features(window_number, window_df):
    # find the ms1 features in this isolation window
    ms1_characteristics_l = []

    window_mz_lower = window_df.mz_lower
    window_mz_upper = window_df.mz_upper
    scan_width = int(window_df.ScanNumEnd - window_df.ScanNumBegin)
    wide_scan_lower = int(window_df.ScanNumBegin - scan_width)
    wide_scan_upper = int(window_df.ScanNumEnd + scan_width)
    fe_scan_lower = int(window_df.ScanNumBegin)
    fe_scan_upper = int(window_df.ScanNumEnd)
    wide_rt_lower = window_df.wide_rt_lower
    wide_rt_upper = window_df.wide_rt_upper
    fe_rt_lower = window_df.fe_rt_lower
    fe_rt_upper = window_df.fe_rt_upper
    precursor_id = int(window_df.Precursor)

    # get the ms1 frame IDs
    ms1_frame_ids = tuple(ms1_frame_properties_df.frame_id)

    # load the cube's raw ms1 points
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    ms1_raw_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {} and frame_id in {}".format(window_mz_lower, window_mz_upper, wide_scan_lower, wide_scan_upper, wide_rt_lower, wide_rt_upper, ms1_frame_ids), db_conn)
    db_conn.close()

    # get the raw points constrained to the fragmentation event's RT
    fe_raw_points_df = ms1_raw_points_df[(ms1_raw_points_df.retention_time_secs >= fe_rt_lower) & (ms1_raw_points_df.retention_time_secs <= fe_rt_upper)]

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

    # see https://github.com/mobiusklein/ms_deisotope/blob/ee4b083ad7ab5f77722860ce2d6fdb751886271e/ms_deisotope/deconvolution/api.py#L17
    deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.95)

    ms1_deconvoluted_peaks_l = []
    for peak in deconvoluted_peaks:
        # discard a monoisotopic peak that has either of the first two peaks as placeholders (indicated by intensity of 1)
        if ((len(peak.envelope) >= 3) and (peak.envelope[0][1] > 1) and (peak.envelope[1][1] > 1)):
            ms1_deconvoluted_peaks_l.append((peak.mz, peak.neutral_mass, peak.intensity, peak.score, peak.signal_to_noise, peak.envelope, peak.charge))

    ms1_deconvoluted_peaks_df = pd.DataFrame(ms1_deconvoluted_peaks_l, columns=['mz','neutral_mass','intensity','score','SN','envelope','charge'])
    # 'neutral mass' is the zero charge M, so we add the proton mass to get M+H (the monoisotopic mass)
    ms1_deconvoluted_peaks_df['m_plus_h'] = ms1_deconvoluted_peaks_df.neutral_mass + PROTON_MASS

    # For each monoisotopic peak found, find its apex in RT and mobility
    print("window {}, processing {} monoisotopics".format(window_number, len(ms1_deconvoluted_peaks_df)))
    for monoisotopic_idx in range(len(ms1_deconvoluted_peaks_df)):
        feature_monoisotopic_mz = ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].mz
        feature_charge = int(ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].charge)
        feature_intensity = int(ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].intensity)
        second_peak_mz = ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].envelope[1][0]
        feature_monoisotopic_mass = ms1_deconvoluted_peaks_df.iloc[monoisotopic_idx].m_plus_h

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
                scan_apex = peakutils.peak.gaussian_fit(scan_df.scan, scan_df.intensity, center_only=True)
                if (scan_apex >= wide_scan_lower) and (scan_apex <= wide_scan_upper):
                    mobility_curve_fit = True
            except:
                pass

            # if we couldn't fit a curve to the mobility dimension, take the intensity-weighted centroid
            if not mobility_curve_fit:
                scan_apex = peakutils.centroid(scan_df.scan, scan_df.intensity)

            feature_scan_apex = scan_apex
            feature_scan_lower = window_df.ScanNumBegin
            feature_scan_upper = window_df.ScanNumEnd

            # In the RT dimension, look wider to find the apex of the peak closest to the fragmentation event
            wide_rt_monoisotopic_raw_points_df = ms1_raw_points_df[(ms1_raw_points_df.mz >= monoisotopic_mz_lower) & (ms1_raw_points_df.mz <= monoisotopic_mz_upper)]
            rt_df = wide_rt_monoisotopic_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()

            rt_curve_fit = False
            peaks_threshold = 0.3
            peaks_idx = peakutils.indexes(rt_df.intensity.values, thres=peaks_threshold, min_dist=10)
            if len(peaks_idx) > 0:
                # get the peak closest to the fragmentation event
                peaks_df = rt_df.iloc[peaks_idx].copy()
                peaks_df['fragmentation_rt_delta'] = abs(window_df.retention_time_secs - peaks_df.retention_time_secs)
                peak_idx = peaks_df.fragmentation_rt_delta.idxmin()
                feature_rt_apex = peaks_df.loc[peak_idx].retention_time_secs
                rt_curve_fit = True
            else:
                # couldn't find a peak so just take the maximum
                peak_idx = rt_df.intensity.idxmax()
                feature_rt_apex = rt_df.loc[peak_idx].retention_time_secs

            valleys_idx = peakutils.indexes(-rt_df.intensity.values, thres=0.6, min_dist=args.rt_base_peak_width_secs/20)
            valleys_df = rt_df.iloc[valleys_idx].copy()

            # find the closest valley above the peak
            if (len(valleys_df) > 0) and (max(valleys_idx) > peak_idx):
                valley_idx_above = valleys_idx[valleys_idx > peak_idx].min()
                feature_rt_base_upper = valleys_df.loc[valley_idx_above].retention_time_secs
            else:
                feature_rt_base_upper = rt_df.retention_time_secs.max()

            # find the closest valley below the peak
            if (len(valleys_df) > 0) and (min(valleys_idx) < peak_idx):
                valley_idx_below = valleys_idx[valleys_idx < peak_idx].max()
                feature_rt_base_lower = valleys_df.loc[valley_idx_below].retention_time_secs
            else:
                feature_rt_base_lower = rt_df.retention_time_secs.min()

            # find the isolation windows overlapping the feature's mono or second peak, plus scan and RT
            indexes = isolation_window_df.index[
                                        (((isolation_window_df.mz_upper >= feature_monoisotopic_mz) & (isolation_window_df.mz_lower <= feature_monoisotopic_mz)) |
                                        ((isolation_window_df.mz_upper >= second_peak_mz) & (isolation_window_df.mz_lower <= second_peak_mz))) &
                                        (isolation_window_df.ScanNumEnd >= feature_scan_apex) &
                                        (isolation_window_df.ScanNumBegin <= feature_scan_apex) &
                                        (isolation_window_df.retention_time_secs >= feature_rt_base_lower) &
                                        (isolation_window_df.retention_time_secs <= feature_rt_base_upper)
                                   ]
            isolation_windows_overlapping_feature_df = isolation_window_df.loc[indexes]

            if len(isolation_windows_overlapping_feature_df) > 0:
                ms2_frames = list(isolation_windows_overlapping_feature_df.Frame)
                ms1_characteristics_l.append((round(feature_monoisotopic_mass,6), feature_charge, feature_monoisotopic_mz, feature_intensity, feature_scan_apex, mobility_curve_fit, round(feature_rt_apex,2), rt_curve_fit, precursor_id, ms2_frames))

    ms1_characteristics_df = pd.DataFrame(ms1_characteristics_l, columns=['monoisotopic_mass', 'charge', 'monoisotopic_mz', 'intensity', 'scan_apex', 'scan_curve_fit', 'rt_apex', 'rt_curve_fit', 'precursor_id', 'ms2_frames'])
    return ms1_characteristics_df

def remove_ms1_duplicates(ms1_features_df):
    scratch_df = ms1_features_df.copy() # take a copy because we're going to delete stuff
    ms1_peaks_l = []
    while len(scratch_df) > 0:
        # take the first row
        row = scratch_df.iloc[0]
        mz = row.monoisotopic_mz
        scan = row.scan_apex
        rt = row.rt_apex

        # calculate the matching bounds
        mz_ppm_tolerance = mz * MZ_TOLERANCE_PERCENT / 100
        mz_lower = mz - mz_ppm_tolerance
        mz_upper = mz + mz_ppm_tolerance
        scan_lower = scan - SCAN_TOLERANCE
        scan_upper = scan + SCAN_TOLERANCE
        rt_lower = rt - RT_TOLERANCE
        rt_upper = rt + RT_TOLERANCE

        # find the matches within these tolerances
        matches_df = scratch_df[(scratch_df.monoisotopic_mz >= mz_lower) & (scratch_df.monoisotopic_mz <= mz_upper) & (scratch_df.scan_apex >= scan_lower) & (scratch_df.scan_apex <= scan_upper) & (scratch_df.rt_apex >= rt_lower) & (scratch_df.rt_apex <= rt_upper)]
        print(matches_df.to_string())
        print(matches_df['intensity'])
        if len(matches_df) == 1:
            peak_df = matches_df
        else:
            peak_df = matches_df.loc[matches_df['intensity'].idxmax()].copy()
        peak_df['duplicates'] = len(matches_df)

        # add the most intense to the list
        ms1_peaks_l.append(tuple(peak_df))

        # remove the matches
        scratch_df = scratch_df[~scratch_df.isin(matches_df)].dropna(how = 'all')
        print("{} matches, {} remaining".format(len(matches_df), len(scratch_df)))

    ms1_deduped_df = pd.DataFrame(ms1_peaks_l, columns=['monoisotopic_mass', 'charge', 'monoisotopic_mz', 'intensity', 'scan_apex', 'scan_curve_fit', 'rt_apex', 'rt_curve_fit', 'precursor_id', 'duplicates'])
    ms1_deduped_df.sort_values(by=['intensity'], ascending=False, inplace=True)
    ms1_deduped_df["feature_id"] = np.arange(start=1, stop=len(ms1_deduped_df)+1)
    return ms1_deduped_df

def deconvolute_ms2_peaks_for_feature(binned_ms2_df):
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
            ms2_peaks_l.append((mz_centroid, summed_intensity))

            # remove the raw points assigned to this peak
            raw_scratch_df = raw_scratch_df[~raw_scratch_df.isin(peak_raw_points_df)].dropna(how = 'all')

    ms2_peaks_df = pd.DataFrame(ms2_peaks_l, columns=['mz','intensity'])

    print("{} ms2 peaks prior to deconvolution".format(len(ms2_peaks_df)))

    # deconvolute the peaks
    ms2_deconvoluted_peaks, _ = deconvolute_peaks(ms2_peaks_l, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.8)

    print("{} ms2 peaks after deconvolution".format(len(ms2_deconvoluted_peaks)))

    ms2_deconvoluted_peaks_l = []
    for peak in ms2_deconvoluted_peaks:
        # discard a monoisotopic peak that has either of the first two peaks as placeholders (indicated by intensity of 1)
        if ((len(peak.envelope) >= 3) and (peak.envelope[0][1] > 1) and (peak.envelope[1][1] > 1)):
            ms2_deconvoluted_peaks_l.append((round(peak.mz, 4), int(peak.charge), peak.neutral_mass, int(peak.intensity), peak.score, peak.signal_to_noise))

    ms2_deconvoluted_peaks_df = pd.DataFrame(ms2_deconvoluted_peaks_l, columns=['mz','charge','neutral_mass','intensity','score','SN'])
    print("{} peaks after quality filtering".format(len(ms2_deconvoluted_peaks_df)))
    # 'neutral mass' is the zero charge M, so we add the proton mass to get M+H (the monoisotopic mass)
    ms2_deconvoluted_peaks_df['m_plus_h'] = ms2_deconvoluted_peaks_df.neutral_mass + PROTON_MASS

    return ms2_deconvoluted_peaks_df

# calculate the centroid, intensity of a bin
def calc_centroid(bin_df):
    d = {}
    d['bin_idx'] = bin_df.iloc[0].bin_idx
    d['mz_centroid'] = peakutils.centroid(bin_df.mz, bin_df.intensity)
    d['summed_intensity'] = bin_df.intensity.sum()
    d['point_count'] = len(bin_df)
    return pd.Series(d, index=['bin_idx','mz_centroid','summed_intensity','point_count'])

# sum and centroid the ms2 bins for this feature
def find_ms2_peaks_for_feature(feature_df, binned_ms2_df):
    # calculate the bin centroid and summed intensity for the combined frames
    combined_ms2_df = ms2_raw_points_df.groupby(['bin_idx'], as_index=False).apply(calc_centroid)
    combined_ms2_df.summed_intensity = combined_ms2_df.summed_intensity.astype(int)
    combined_ms2_df.bin_idx = combined_ms2_df.bin_idx.astype(int)
    combined_ms2_df.point_count = combined_ms2_df.point_count.astype(int)
    return combined_ms2_df

def collate_spectra_for_feature(feature_df, ms2_deconvoluted_df):
    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    pairs_df = ms2_deconvoluted_df[['mz', 'intensity']].copy().sort_values(by=['intensity'], ascending=False)
    spectrum = {}
    spectrum["m/z array"] = pairs_df.mz.values
    spectrum["intensity array"] = pairs_df.intensity.values
    params = {}
    params["TITLE"] = "RawFile: {} Index: 10 precursor: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(CONVERTED_DATABASE_NAME).split('.')[0], feature_df.precursor_id, feature_df.charge, feature_df.intensity, feature_df.feature_id, round(feature_df.rt_apex,2))
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(feature_df.monoisotopic_mass,6), feature_df.intensity)
    params["CHARGE"] = "{}+".format(feature_df.charge)
    params["RTINSECONDS"] = "{}".format(round(feature_df.rt_apex,2))
    params["SCANS"] = "{}".format(int(feature_df.rt_apex))
    spectrum["params"] = params
    return spectrum


if args.new_ms1_features:
    # find ms1 features for each unique precursor ID
    print("finding ms1 features")
    ms1_df_l = ray.get([find_features.remote(window_number=idx+1, window_df=group_df.iloc[0]) for idx,group_df in isolation_window_df.groupby('Precursor')])
    ms1_df = pd.concat(ms1_df_l)  # combines a list of dataframes into a single dataframe
    ms1_df.to_pickle('./ms1_df.pkl')
    print("detected {} features".format(len(ms1_df)))
else:
    # load previously detected ms1 features
    print("loading ms1 features")
    ms1_df = pd.read_pickle(args.ms1_features_filename)
    print("loaded {} features".format(len(ms1_df)))

if args.new_dedup_ms1_features:
    # remove duplicates in ms1
    print("removing duplicates")
    ms1_deduped_df = remove_ms1_duplicates(ms1_df)
    ms1_deduped_df.to_pickle(args.dedup_ms1_filename)
    print("removed {} duplicates - processing {} features".format(len(ms1_df)-len(ms1_deduped_df), len(ms1_deduped_df)))
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
    print("binned {} points".format(len(binned_ms2_df)))
else:
    # load previously binned ms2
    print("loading pre-binned ms2 frames")
    binned_ms2_df = pd.read_pickle(args.pre_binned_ms2_filename)
    print("loaded {} pre-binned points".format(len(binned_ms2_df)))

# find ms2 peaks for each feature found in ms1, and collate the spectra for the MGF
print("finding peaks in ms2 for each feature")
mgf_spectra = []
ms2_spectra = []
ms1_deduped_df.reset_index(inplace=True)
for idx,feature_df in ms1_deduped_df.iterrows():
    print("processing feature {} of {}".format(idx+1, len(ms1_deduped_df)))
    # get the binned ms2 points for this feature
    ms2_frame_ids = feature_df.ms2_frames
    ms2_raw_points_df = binned_ms2_df[binned_ms2_df.frame_id.isin(ms2_frame_ids)]
    ms2_peaks_df = find_ms2_peaks_for_feature(feature_df, ms2_raw_points_df)
    ms2_deconvoluted_df = deconvolute_ms2_peaks_for_feature(ms2_peaks_df)
    feature_spectra = collate_spectra_for_feature(feature_df, ms2_deconvoluted_df)
    mgf_spectra.append(feature_spectra)
    ms2_spectra.append((feature_df, feature_spectra))

# generate the MGF for all the features
print("generating the MGF: {}".format(args.mgf_filename))
if os.path.isfile(args.mgf_filename):
    os.remove(args.mgf_filename)
mgf.write(output=args.mgf_filename, spectra=mgf_spectra)

with open('./ms2_spectra.pkl', 'wb') as f:
    pickle.dump(ms2_spectra, f)

stop_run = time.time()
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))
print("{} info: {}".format(parser.prog, info))

print("shutting down ray")
ray.shutdown()
