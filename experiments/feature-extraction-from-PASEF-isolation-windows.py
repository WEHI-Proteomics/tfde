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


parser = argparse.ArgumentParser(description='Extract features from PASEF isolation windows.')
parser.add_argument('-cdbb','--converted_database_base', type=str, help='base path to the converted database.', required=True)
parser.add_argument('-rdbb','--raw_database_base', type=str, help='base path to the raw database.', required=True)
parser.add_argument('-rtl','--rt_lower', type=float, help='The lower limit of retention time (secs).', required=True)
parser.add_argument('-rtu','--rt_upper', type=float, help='The upper limit of retention time (secs).', required=True)
parser.add_argument('-rtpw','--rt_base_peak_width_secs', type=float, default=30.0, help='How broad to look in RT for the peak apex (secs).', required=False)
parser.add_argument('-rtfe','--rt_fragment_event_delta_secs', type=float, default=3.5, help='How wide to look around the region of the fragmentation event (secs).', required=False)
parser.add_argument('-ms1ce','--ms1_collision_energy', type=float, default=10.0, help='Collision energy used in ms1 frames.', required=False)
parser.add_argument('-ms1bw','--ms1_bin_width', type=float, default=0.00001, help='Width of ms1 bins, in Thomsons.', required=False)
parser.add_argument('-ms2bw','--ms2_bin_width', type=float, default=0.001, help='Width of ms2 bins, in Thomsons.', required=False)
parser.add_argument('-ms1dt','--ms1_peak_delta', type=float, default=0.01, help='How far either side of a peak in ms1 to include when calculating its centroid and intensity, in Thomsons.', required=False)
parser.add_argument('-ms2dt','--ms2_peak_delta', type=float, default=0.01, help='How far either side of a peak in ms2 to include when calculating its centroid and intensity, in Thomsons.', required=False)
parser.add_argument('-cl','--cluster_mode', action='store_true', help='Run on a cluster.')
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

CONVERTED_DATABASE_NAME = '{}/HeLa_20KInt.sqlite'.format(args.converted_database_base)
MGF_FILENAME = '{}/HeLa_20KInt-features.mgf'.format(args.converted_database_base)

RAW_DATABASE_NAME = "{}/analysis.tdf".format(args.raw_database_base)

PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

print("reading converted raw data from {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(args.rt_lower, args.rt_upper, args.ms1_collision_energy), db_conn)
ms2_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy <> {}".format(args.rt_lower, args.rt_upper, args.ms1_collision_energy), db_conn)
db_conn.close()

if os.path.isfile(MGF_FILENAME):
    os.remove(MGF_FILENAME)

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

print("There are {} precursor unique isolation windows.".format(isolation_window_df.Precursor.nunique()))

@ray.remote
def analyse_isolation_window(window_number, window_df):
    print("processing precursor window {}".format(window_number))

    feature_id = 0
    mgf_spectra = []

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
    ms1_frame_ids = tuple(ms1_frame_properties_df.astype(int).values[:,0])

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
        # discard a monoisotopic peak that has a second isotope with intensity of 1 (rubbish value)
        if ((len(peak.envelope) > 1) and (peak.envelope[1][1] > 1)):
            ms1_deconvoluted_peaks_l.append((peak.mz, peak.neutral_mass, peak.intensity, peak.score, peak.signal_to_noise, peak.envelope, peak.charge))

    ms1_deconvoluted_peaks_df = pd.DataFrame(ms1_deconvoluted_peaks_l, columns=['mz','neutral_mass','intensity','score','SN','envelope','charge'])
    # 'neutral mass' is the zero charge M, so we add the proton mass to get M+H (the monoisotopic mass)
    ms1_deconvoluted_peaks_df['m_plus_h'] = ms1_deconvoluted_peaks_df.neutral_mass + PROTON_MASS

    print("\twindow {}: there are {} monoisotopics in the deconvolution".format(window_number, len(ms1_deconvoluted_peaks_df)))

    # For each monoisotopic peak found, find its apex in RT and mobility
    for monoisotopic_idx in range(len(ms1_deconvoluted_peaks_df)):
        print("\twindow {}, processing monoisotopic {}".format(window_number, monoisotopic_idx+1))
        feature_id += 1
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
            centroid_scan = peakutils.centroid(scan_df.scan, scan_df.intensity)

            feature_scan_centroid = centroid_scan
            feature_scan_lower = window_df.ScanNumBegin
            feature_scan_upper = window_df.ScanNumEnd

            # In the RT dimension, look wider to find the apex of the peak closest to the fragmentation event
            wide_rt_monoisotopic_raw_points_df = ms1_raw_points_df[(ms1_raw_points_df.mz >= monoisotopic_mz_lower) & (ms1_raw_points_df.mz <= monoisotopic_mz_upper)]
            rt_df = wide_rt_monoisotopic_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()

            peaks_threshold = 0.3
            peaks_idx = peakutils.indexes(rt_df.intensity.values, thres=peaks_threshold, min_dist=10)
            if len(peaks_idx) > 0:
                # get the peak closest to the fragmentation event
                peaks_df = rt_df.iloc[peaks_idx].copy()
                peaks_df['fragmentation_rt_delta'] = abs(window_df.retention_time_secs - peaks_df.retention_time_secs)
                peak_idx = peaks_df.fragmentation_rt_delta.idxmin()
                feature_rt_apex = peaks_df.loc[peak_idx].retention_time_secs
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
                                        (isolation_window_df.ScanNumEnd >= feature_scan_centroid) &
                                        (isolation_window_df.ScanNumBegin <= feature_scan_centroid) &
                                        (isolation_window_df.retention_time_secs >= feature_rt_base_lower) &
                                        (isolation_window_df.retention_time_secs <= feature_rt_base_upper)
                                   ]
            isolation_windows_overlapping_feature_df = isolation_window_df.loc[indexes]

            if len(isolation_windows_overlapping_feature_df) > 0:
                print("\t\twindow {}, there are {} overlapping isolation windows - finding the ms2 peaks".format(window_number, len(isolation_windows_overlapping_feature_df)))

                # extract the raw ms2 points from the overlapping isolation windows
                ms2_raw_points_df = pd.DataFrame()
                for idx in range(len(isolation_windows_overlapping_feature_df)):
                    isolation_window_scan_lower = int(isolation_windows_overlapping_feature_df.iloc[idx].ScanNumBegin)
                    isolation_window_scan_upper = int(isolation_windows_overlapping_feature_df.iloc[idx].ScanNumEnd)
                    isolation_window_frame_id = int(isolation_windows_overlapping_feature_df.iloc[idx].Frame)

                    # get the raw ms2 points from the fragmentation frame within the scan constraints
                    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
                    df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_id == {} and scan >= {} and scan <= {}".format(isolation_window_frame_id,isolation_window_scan_lower,isolation_window_scan_upper), db_conn)
                    db_conn.close()

                    # add these to the collection
                    ms2_raw_points_df = ms2_raw_points_df.append(df, ignore_index=True)

                # bin the data
                MS2_MZ_MAX = ms2_raw_points_df.mz.max()
                MS2_MZ_MIN = ms2_raw_points_df.mz.min()

                ms2_bins = np.arange(start=MS2_MZ_MIN, stop=MS2_MZ_MAX+args.ms2_bin_width, step=args.ms2_bin_width)  # go slightly wider to accomodate the maximum value
                MS2_MZ_BIN_COUNT = len(ms2_bins)

                # initialise an array of lists to hold the m/z and intensity values allocated to each bin
                ms2_mz_values_array = np.empty(MS2_MZ_BIN_COUNT, dtype=np.object)
                for idx in range(MS2_MZ_BIN_COUNT):
                    ms2_mz_values_array[idx] = []

                # gather the m/z values into bins
                for r in zip(ms2_raw_points_df.mz, ms2_raw_points_df.intensity):
                    mz = r[0]
                    intensity = int(r[1])
                    if (mz >= MS2_MZ_MIN) and (mz <= MS2_MZ_MAX): # it should already but just to be sure
                        mz_array_idx = int(np.digitize(mz, ms2_bins)) # in which bin should this mz go
                        ms2_mz_values_array[mz_array_idx].append((mz, intensity))

                # compute the intensity-weighted m/z centroid and the summed intensity of the bins
                binned_ms2_l = []
                for bin_idx in range(MS2_MZ_BIN_COUNT):
                    if len(ms2_mz_values_array[bin_idx]) > 0:
                        mz_values_for_bin = np.array([ list[0] for list in ms2_mz_values_array[bin_idx]])
                        intensity_values_for_bin = np.array([ list[1] for list in ms2_mz_values_array[bin_idx]]).astype(int)
                        mz_centroid = peakutils.centroid(mz_values_for_bin, intensity_values_for_bin)
                        summed_intensity = intensity_values_for_bin.sum()
                        binned_ms2_l.append((mz_centroid,summed_intensity))

                binned_ms2_df = pd.DataFrame(binned_ms2_l, columns=['mz_centroid','summed_intensity'])

                # now do intensity descent to find the peaks
                raw_scratch_df = binned_ms2_df.copy() # take a copy because we're going to delete stuff

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

                # deconvolute the peaks
                ms2_deconvoluted_peaks, _ = deconvolute_peaks(ms2_peaks_l, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.95)

                ms2_deconvoluted_peaks_l = []
                for peak in ms2_deconvoluted_peaks:
                    # discard a monoisotopic peak that has a second isotope with intensity of 1 (rubbish value)
                    if ((len(peak.envelope) > 1) and (peak.envelope[1][1] > 1)):
                        ms2_deconvoluted_peaks_l.append((round(peak.mz, 4), int(peak.charge), peak.neutral_mass, int(peak.intensity), peak.score, peak.signal_to_noise))

                ms2_deconvoluted_peaks_df = pd.DataFrame(ms2_deconvoluted_peaks_l, columns=['mz','charge','neutral_mass','intensity','score','SN'])
                # 'neutral mass' is the zero charge M, so we add the proton mass to get M+H (the monoisotopic mass)
                ms2_deconvoluted_peaks_df['m_plus_h'] = ms2_deconvoluted_peaks_df.neutral_mass + PROTON_MASS

                print("\t\twindow {}, building the MGF".format(window_number))

                # append the monoisotopic and the ms2 fragments to the list for MGF creation
                pairs_df = ms2_deconvoluted_peaks_df[['mz', 'intensity']].copy().sort_values(by=['intensity'], ascending=False)
                spectra = []
                spectrum = {}
                spectrum["m/z array"] = pairs_df.mz.values
                spectrum["intensity array"] = pairs_df.intensity.values
                params = {}
                params["TITLE"] = "RawFile: {} Index: 0 precursor: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(CONVERTED_DATABASE_NAME).split('.')[0], precursor_id, feature_charge, feature_intensity, feature_id, round(feature_rt_apex,2))
                params["INSTRUMENT"] = "ESI-QUAD-TOF"
                params["PEPMASS"] = "{} {}".format(round(feature_monoisotopic_mass,6), feature_intensity)
                params["CHARGE"] = "{}+".format(feature_charge)
                params["RTINSECONDS"] = "{}".format(round(feature_rt_apex,2))
                params["SCANS"] = "{}".format(int(feature_rt_apex))
                spectrum["params"] = params
                spectra.append(spectrum)

                # add it to the list of spectra
                mgf_spectra.append(spectra)
            else:
                print("\t\twindow {}, there were no overlapping isolation windows".format(window_number))
        else:
            print("\t\twindow {}, found no raw points in this monoisotopic's region - skipping".format(window_number))
    return mgf_spectra

# run the analysis for each unique precursor ID
spectra_l = ray.get([analyse_isolation_window.remote(window_number=idx+1, window_df=group_df.iloc[0]) for idx,group_df in isolation_window_df.groupby('Precursor')])

# write out the MGF
print("generating the MGF at {}".format(MGF_FILENAME))
for spectra in spectra_l:
    for spec in spectra:
        mgf.write(output=MGF_FILENAME, spectra=spec, file_mode='a')

stop_run = time.time()
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))
print("{} info: {}".format(parser.prog, info))

print("shutting down ray")
ray.shutdown()
