import sqlite3
import pandas as pd
import numpy as np
import sys
import peakutils
from ms_deisotope import deconvolute_peaks, averagine, scoring
from pyteomics import mgf
import os.path


RT_LIMIT_LOWER = 4340  # RT range in the database
RT_LIMIT_UPPER = 4580
RT_BASE_PEAK_WIDTH_SECS = 30.0  # assumption about base peak width in RT
RT_FRAGMENT_EVENT_DELTA_SECS = 3.5  # use this window for constraining RT to focus on the fragmentation event
MS1_CE = 10

BASE_NAME = "/Users/darylwilding-mcbride/Downloads/HeLa_20KInt-rt-{}-{}-denoised".format(RT_LIMIT_LOWER,RT_LIMIT_UPPER)
BASE_MAXQUANT_TXT_DIR = '/Users/darylwilding-mcbride/Downloads/maxquant_results/txt'
ALLPEPTIDES_FILENAME = '{}/allPeptides.txt'.format(BASE_MAXQUANT_TXT_DIR)
PASEF_MSMS_SCANS_FILENAME = '{}/pasefMsmsScans.txt'.format(BASE_MAXQUANT_TXT_DIR)
CONVERTED_DATABASE_NAME = '{}/HeLa_20KInt.sqlite'.format(BASE_NAME)
MGF_FILENAME = '{}/HeLa_20KInt-features.mgf'.format(BASE_NAME)

feature_id = 0


db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(RT_LIMIT_LOWER,RT_LIMIT_UPPER,MS1_CE), db_conn)
ms2_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy <> {}".format(RT_LIMIT_LOWER,RT_LIMIT_UPPER,MS1_CE), db_conn)
db_conn.close()

if os.path.isfile(MGF_FILENAME):
    os.remove(mgf_filename)

# get all the isolation windows
isolation_window_df = pd.read_pickle('/Users/darylwilding-mcbride/Downloads/isolation_window_df.pkl')

# add-in the retention time for the isolation windows and filter out the windows not in range
isolation_window_df = pd.merge(isolation_window_df, ms2_frame_properties_df, how='left', left_on=['Frame'], right_on=['frame_id'])
isolation_window_df.drop(['frame_id', 'CollisionEnergy'], axis=1, inplace=True)
isolation_window_df.dropna(subset=['retention_time_secs'], inplace=True)
isolation_window_df['mz_lower'] = isolation_window_df.IsolationMz - (isolation_window_df.IsolationWidth / 2) - 0.7
isolation_window_df['mz_upper'] = isolation_window_df.IsolationMz + (isolation_window_df.IsolationWidth / 2) + 0.7
isolation_window_df['wide_rt_lower'] = isolation_window_df.retention_time_secs - RT_BASE_PEAK_WIDTH_SECS
isolation_window_df['wide_rt_upper'] = isolation_window_df.retention_time_secs + RT_BASE_PEAK_WIDTH_SECS
isolation_window_df['fe_rt_lower'] = isolation_window_df.retention_time_secs - RT_FRAGMENT_EVENT_DELTA_SECS
isolation_window_df['fe_rt_upper'] = isolation_window_df.retention_time_secs + RT_FRAGMENT_EVENT_DELTA_SECS

# filter out isolation windows that don't fit in the database subset we have loaded
isolation_window_df = isolation_window_df[(isolation_window_df.wide_rt_lower >= RT_LIMIT_LOWER) & (isolation_window_df.wide_rt_upper <= RT_LIMIT_UPPER)]

print("There are {} precursor isolation windows.".format(len(isolation_window_df)))

# Analyse all the isolation windows reported
for isolation_window_idx in range(len(isolation_window_df)):
    print("processing precursor window {} of {}".format(isolation_window_idx+1, len(isolation_window_df)))
    window_df = isolation_window_df.iloc[isolation_window_idx]

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
    fe_raw_points_df.to_csv('/Users/darylwilding-mcbride/Downloads/fe_raw_points_ms1.csv')

    MS1_MZ_BIN_WIDTH = 0.1

    ms1_bins = np.arange(start=window_mz_lower, stop=window_mz_upper+MS1_MZ_BIN_WIDTH, step=MS1_MZ_BIN_WIDTH)  # go slightly wider to accomodate the maximum value

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
    ms1_peaks = []
    for bin_idx in range(MZ_BIN_COUNT):
        if len(ms1_mz_values_array[bin_idx]) > 0:
            mz_values_for_bin = np.array([ list[0] for list in ms1_mz_values_array[bin_idx]])
            intensity_values_for_bin = np.array([ list[1] for list in ms1_mz_values_array[bin_idx]]).astype(int)
            mz_centroid = peakutils.centroid(mz_values_for_bin, intensity_values_for_bin)
            summed_intensity = intensity_values_for_bin.sum()
            ms1_peaks.append((mz_centroid,summed_intensity))

    ms1_peaks_df = pd.DataFrame(ms1_peaks, columns=['mz_centroid','summed_intensity'])

    MS1_MZ_BIN_WIDTH = 1e-5

    ms1_bins = np.arange(start=window_mz_lower, stop=window_mz_upper+MS1_MZ_BIN_WIDTH, step=MS1_MZ_BIN_WIDTH)  # go slightly wider to accomodate the maximum value

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

    MZ_DELTA = 0.01

    ms1_peaks_l = []
    while len(raw_scratch_df) > 0:
        # find the most intense point
        peak_df = raw_scratch_df.loc[raw_scratch_df.summed_intensity.idxmax()]
        peak_mz = peak_df.mz_centroid
        peak_mz_lower = peak_mz - MZ_DELTA
        peak_mz_upper = peak_mz + MZ_DELTA

        # get all the raw points within this m/z region
        peak_raw_points_df = raw_scratch_df[(raw_scratch_df.mz_centroid >= peak_mz_lower) & (raw_scratch_df.mz_centroid <= peak_mz_upper)]
        mz_centroid = peakutils.centroid(peak_raw_points_df.mz_centroid, peak_raw_points_df.summed_intensity)
        summed_intensity = peak_raw_points_df.summed_intensity.sum()
        ms1_peaks_l.append((mz_centroid, summed_intensity))

        # remove the raw points assigned to this peak
        raw_scratch_df = raw_scratch_df[~raw_scratch_df.isin(peak_raw_points_df)].dropna(how = 'all')

    ms1_peaks_df = pd.DataFrame(ms1_peaks_l, columns=['mz','intensity'])

    PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

    # see https://github.com/mobiusklein/ms_deisotope/blob/ee4b083ad7ab5f77722860ce2d6fdb751886271e/ms_deisotope/deconvolution/api.py#L17
    deconvoluted_peaks, _priority_targets = deconvolute_peaks(ms1_peaks_l, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.95)

    peaks_l = []
    for peak in deconvoluted_peaks:
        # discard a monoisotopic peak that has a second isotope with intensity of 1 (rubbish value)
        if ((len(peak.envelope) > 1) and (peak.envelope[1][1] > 1)):
            peaks_l.append((peak.mz, peak.neutral_mass, peak.intensity, peak.score, peak.signal_to_noise, peak.envelope, peak.charge))

    deconvoluted_peaks_df = pd.DataFrame(peaks_l, columns=['mz','neutral_mass','intensity','score','SN','envelope','charge'])
    # 'neutral mass' is the zero charge M, so we add the proton mass to get M+H (the monoisotopic mass)
    deconvoluted_peaks_df['m_plus_h'] = deconvoluted_peaks_df.neutral_mass + PROTON_MASS

    print("there are {} monoisotopics in the deconvolution".format(len(deconvoluted_peaks_df)))

    # For each monoisotopic peak found, find its apex in RT and mobility
    for monoisotopic_idx in range(len(deconvoluted_peaks_df)):
        feature_id += 1
        feature_monoisotopic_mz = deconvoluted_peaks_df.iloc[monoisotopic_idx].mz
        feature_charge = int(deconvoluted_peaks_df.iloc[monoisotopic_idx].charge)
        feature_intensity = int(deconvoluted_peaks_df.iloc[monoisotopic_idx].intensity)
        second_peak_mz = deconvoluted_peaks_df.iloc[monoisotopic_idx].envelope[1][0]
        feature_monoisotopic_mass = deconvoluted_peaks_df.iloc[monoisotopic_idx].m_plus_h

        # Get the raw points for the monoisotopic peak (constrained by the fragmentation event)
        MZ_TOLERANCE_PPM = 20
        MZ_TOLERANCE_PERCENT = MZ_TOLERANCE_PPM * 10**-4

        monoisotopic_mz_ppm_tolerance = feature_monoisotopic_mz * MZ_TOLERANCE_PERCENT / 100
        monoisotopic_mz_lower = feature_monoisotopic_mz - monoisotopic_mz_ppm_tolerance
        monoisotopic_mz_upper = feature_monoisotopic_mz + monoisotopic_mz_ppm_tolerance

        monoisotopic_raw_points_df = fe_raw_points_df[(fe_raw_points_df.mz >= monoisotopic_mz_lower) & (fe_raw_points_df.mz <= monoisotopic_mz_upper)]

        # collapsing the monoisotopic's summed points onto the mobility dimension
        scan_df = monoisotopic_raw_points_df.groupby(['scan'], as_index=False).intensity.sum()
        centroid_scan = peakutils.centroid(scan_df.scan, scan_df.intensity)

        feature_scan_centroid = centroid_scan
        feature_scan_lower = window_df.ScanNumBegin
        feature_scan_upper = window_df.ScanNumEnd

        # ### In the RT dimension, look wider to find the apex of the peak closest to the fragmentation event
        wide_rt_monoisotopic_raw_points_df = ms1_raw_points_df[(ms1_raw_points_df.mz >= monoisotopic_mz_lower) & (ms1_raw_points_df.mz <= monoisotopic_mz_upper)]

        rt_df = wide_rt_monoisotopic_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()

        peaks_threshold = 0.3
        peaks_idx = peakutils.indexes(rt_df.intensity.values, thres=peaks_threshold, min_dist=10)
        if len(peaks_idx) == 0:
            peaks_threshold -= 0.1
            peaks_idx = peakutils.indexes(rt_df.intensity.values, thres=peaks_threshold, min_dist=10)
        peaks_df = rt_df.iloc[peaks_idx].copy()

        peaks_df['fragmentation_rt_delta'] = abs(window_df.retention_time_secs - peaks_df.retention_time_secs)

        # get the peak closest to the fragmentation event
        peak_idx = peaks_df.fragmentation_rt_delta.idxmin()

        feature_rt_apex = peaks_df.loc[peak_idx].retention_time_secs

        valleys_idx = peakutils.indexes(-rt_df.intensity.values, thres=0.6, min_dist=RT_BASE_PEAK_WIDTH_SECS/20)
        valleys_df = rt_df.iloc[valleys_idx].copy()

        # find the closest valley above the peak
        if max(valleys_idx) > peak_idx:
            valley_idx_above = valleys_idx[valleys_idx > peak_idx].min()
        else:
            valley_idx_above = -1

        # find the closest valley below the peak
        if min(valleys_idx) < peak_idx:
            valley_idx_below = valleys_idx[valleys_idx < peak_idx].max()
        else:
            valley_idx_below = -1

        feature_rt_base_lower = valleys_df.loc[valley_idx_below].retention_time_secs
        feature_rt_base_upper = valleys_df.loc[valley_idx_above].retention_time_secs

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
        MS2_MZ_BIN_WIDTH = 1e-5

        ms2_bins = np.arange(start=MS2_MZ_MIN, stop=MS2_MZ_MAX+MS2_MZ_BIN_WIDTH, step=MS2_MZ_BIN_WIDTH)  # go slightly wider to accomodate the maximum value

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

        MZ_DELTA = 0.01

        ms2_peaks_l = []
        while len(raw_scratch_df) > 0:
            # find the most intense point
            peak_df = raw_scratch_df.loc[raw_scratch_df.summed_intensity.idxmax()]
            peak_mz = peak_df.mz_centroid
            peak_mz_lower = peak_mz - MZ_DELTA
            peak_mz_upper = peak_mz + MZ_DELTA

            # get all the raw points within this m/z region
            peak_raw_points_df = raw_scratch_df[(raw_scratch_df.mz_centroid >= peak_mz_lower) & (raw_scratch_df.mz_centroid <= peak_mz_upper)]
            mz_centroid = peakutils.centroid(peak_raw_points_df.mz_centroid, peak_raw_points_df.summed_intensity)
            summed_intensity = peak_raw_points_df.summed_intensity.sum()
            ms2_peaks_l.append((mz_centroid, summed_intensity))

            # remove the raw points assigned to this peak
            raw_scratch_df = raw_scratch_df[~raw_scratch_df.isin(peak_raw_points_df)].dropna(how = 'all')

        ms2_peaks_df = pd.DataFrame(ms2_peaks_l, columns=['mz','intensity'])

        # deconvolute the peaks
        ms2_deconvoluted_peaks, _ = deconvolute_peaks(ms2_peaks_l, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(10.0), truncate_after=0.95)

        ms2_peaks_l = []
        for peak in ms2_deconvoluted_peaks:
            # discard a monoisotopic peak that has a second isotope with intensity of 1 (rubbish value)
            if ((len(peak.envelope) > 1) and (peak.envelope[1][1] > 1)):
                ms2_peaks_l.append((round(peak.mz, 4), int(peak.charge), peak.neutral_mass, int(peak.intensity), peak.score, peak.signal_to_noise))

        ms2_deconvoluted_peaks_df = pd.DataFrame(ms2_peaks_l, columns=['mz','charge','neutral_mass','intensity','score','SN'])
        # 'neutral mass' is the zero charge M, so we add the proton mass to get M+H (the monoisotopic mass)
        ms2_deconvoluted_peaks_df['m_plus_h'] = ms2_deconvoluted_peaks_df.neutral_mass + PROTON_MASS

        # append the monoisotopic and the ms2 fragments to the list for MGF creation
        pairs_df = ms2_deconvoluted_peaks_df[['mz', 'intensity']].copy().sort_values(by=['intensity'], ascending=False)
        spectra = []
        spectrum = {}
        spectrum["m/z array"] = pairs_df.centroid_mz.values
        spectrum["intensity array"] = pairs_df.intensity.values
        params = {}
        params["TITLE"] = "RawFile: {} Index: 1318 precursor: 1 Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(CONVERTED_DATABASE_NAME).split('.')[0], feature_charge, feature_intensity, feature_id, feature_rt_apex)
        params["INSTRUMENT"] = "ESI-QUAD-TOF"
        params["PEPMASS"] = "{} {}".format(round(feature_monoisotopic_mass,6), feature_intensity)
        params["CHARGE"] = "{}+".format(feature_charge)
        params["RTINSECONDS"] = "{}".format(feature_rt_apex)
        params["SCANS"] = "{}".format(int(feature_rt_apex))
        spectrum["params"] = params
        spectra.append(spectrum)

        # write out the MGF file
        mgf.write(output=mgf_filename, spectra=spectra, file_mode='a')
