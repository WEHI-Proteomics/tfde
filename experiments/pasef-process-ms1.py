from numba import njit
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
from sys import getsizeof
import configparser
from configparser import ExtendedInterpolation
import warnings
from scipy.optimize.minpack import OptimizeWarning

# so we can use profiling without removing @profile
try:
    profile
except NameError:
    profile = lambda x: x


parser = argparse.ArgumentParser(description='Extract ms1 features from PASEF isolation windows.')
parser.add_argument('-ini','--ini_file', type=str, help='Path to the config file.', required=True)
parser.add_argument('-os','--operating_system', type=str, choices=['linux','macos'], help='Operating system name.', required=True)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster','join'], help='The Ray mode to use.', required=True)
parser.add_argument('-ra','--redis_address', type=str, help='Address of the cluster to join.', required=False)
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.')
parser.add_argument('-idm','--interim_data_mode', action='store_true', help='Write out interim data for debugging.')
args = parser.parse_args()

if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.ini_file)

MASS_DEFECT_WINDOW_DA_MIN = config.getint('common', 'MASS_DEFECT_WINDOW_DA_MIN')
MASS_DEFECT_WINDOW_DA_MAX = config.getint('common', 'MASS_DEFECT_WINDOW_DA_MAX')
PROTON_MASS = config.getfloat('common', 'PROTON_MASS')
RT_LOWER = config.getfloat('common', 'RT_LOWER')
RT_UPPER = config.getfloat('common', 'RT_UPPER')
RT_BASE_PEAK_WIDTH_SECS = config.getfloat('common', 'RT_BASE_PEAK_WIDTH_SECS')
MS1_COLLISION_ENERGY = config.getfloat('common', 'MS1_COLLISION_ENERGY')
SATURATION_INTENSITY = config.getint('common', 'SATURATION_INTENSITY')
INSTRUMENT_RESOLUTION = config.getfloat('common', 'INSTRUMENT_RESOLUTION')
MS2_MZ_ISOLATION_WINDOW_EXTENSION = config.getfloat('ms2', 'MS2_MZ_ISOLATION_WINDOW_EXTENSION')
MS1_PEAK_DELTA = config.getfloat('ms1', 'MS1_PEAK_DELTA')
CARBON_MASS_DIFFERENCE = config.getfloat('common', 'CARBON_MASS_DIFFERENCE')

CONVERTED_DATABASE_NAME = config.get(args.operating_system, 'CONVERTED_DATABASE_NAME')
RAW_DATABASE_NAME = config.get(args.operating_system, 'RAW_DATABASE_NAME')
MS1_PEAK_PKL = config.get(args.operating_system, 'MS1_PEAK_PKL')

RT_FRAGMENT_EVENT_DELTA_FRAMES = config.getint('ms1', 'RT_FRAGMENT_EVENT_DELTA_FRAMES')
MS1_BIN_WIDTH = config.getfloat('ms1', 'MS1_BIN_WIDTH')
NUMBER_OF_STD_DEV_MZ = config.getint('ms1', 'NUMBER_OF_STD_DEV_MZ')
MAX_MS1_PEAK_HEIGHT_RATIO_ERROR = config.getfloat('ms1', 'MAX_MS1_PEAK_HEIGHT_RATIO_ERROR')

# initialise Ray
if not ray.is_initialized():
    if args.ray_mode == "join":
        if args.redis_address is not None:
            ray.init(redis_address=args.redis_address)
        else:
            print("Argument error: a redis_address is needed for join mode")
            sys.exit(1)
    elif args.ray_mode == "cluster":
        ray.init(object_store_memory=40000000000,
                    redis_max_memory=25000000000)
    else:
        ray.init(local_mode=True)

start_run = time.time()

if not os.path.isfile(RAW_DATABASE_NAME):
    print("The raw database doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)

if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

# make sure the right indexes are created in the source database
print("Setting up indexes on {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
src_c = db_conn.cursor()
src_c.execute("create index if not exists idx_pasef_frames_1 on frames (frame_id, mz, scan, intensity)")
src_c.execute("create index if not exists idx_pasef_frames_2 on frames (frame_id, mz, scan, retention_time_secs)")
src_c.execute("create index if not exists idx_pasef_frame_properties_1 on frame_properties (retention_time_secs, collision_energy)")
db_conn.close()

print("reading converted raw data from {}".format(CONVERTED_DATABASE_NAME))
# get all the isolation windows
db_conn = sqlite3.connect(RAW_DATABASE_NAME)
isolation_window_df = pd.read_sql_query("select * from PasefFrameMsMsInfo", db_conn)
db_conn.close()

# get the ms2 frames
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {} order by retention_time_secs".format(RT_LOWER, RT_UPPER, MS1_COLLISION_ENERGY), db_conn)
ms2_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy <> {} order by retention_time_secs".format(RT_LOWER, RT_UPPER, MS1_COLLISION_ENERGY), db_conn)
db_conn.close()

# add-in the retention time for the isolation windows
isolation_window_df = pd.merge(isolation_window_df, ms2_frame_properties_df, how='left', left_on=['Frame'], right_on=['frame_id'])
isolation_window_df.drop(['CollisionEnergy'], axis=1, inplace=True)
isolation_window_df.dropna(subset=['retention_time_secs'], inplace=True)
isolation_window_df['mz_lower'] = isolation_window_df.IsolationMz - (isolation_window_df.IsolationWidth / 2) - MS2_MZ_ISOLATION_WINDOW_EXTENSION
isolation_window_df['mz_upper'] = isolation_window_df.IsolationMz + (isolation_window_df.IsolationWidth / 2) + MS2_MZ_ISOLATION_WINDOW_EXTENSION
# filter out isolation windows that don't fit in the database subset we have loaded
isolation_window_df = isolation_window_df[(isolation_window_df.retention_time_secs >= (RT_LOWER - RT_BASE_PEAK_WIDTH_SECS)) & (isolation_window_df.retention_time_secs <= (RT_UPPER + RT_BASE_PEAK_WIDTH_SECS))]
print("loaded {} isolation windows from {}".format(len(isolation_window_df), RAW_DATABASE_NAME))
isolation_window_df.sort_values(by=['Precursor'], ascending=False, inplace=True)

def time_this(f):
    def timed_wrapper(*args, **kw):
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()

        # Time taken = end_time - start_time
        print('| func:%r args:[%r, %r] took: %2.4f seconds |' % \
              (f.__name__, args, kw, end_time - start_time))
        return result
    return timed_wrapper

# The FWHM is the m/z / instrument resolution. Std dev is FWHM / 2.35482. See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
@njit(fastmath=True)
def standard_deviation(mz):
    FWHM = mz / INSTRUMENT_RESOLUTION
    return FWHM / 2.35482

# takes a numpy array of intensity, and another of mz
@njit(fastmath=True)
def mz_centroid(_int_f, _mz_f):
    return ((_int_f/_int_f.sum()) * _mz_f).sum()

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

# returns a tuple with the characteristics of the feature in the specified row
@njit(fastmath=True)
def collate_feature_characteristics(row, group_df, fe_raw_points_df, ms1_raw_points_df):
    result = None

    feature_monoisotopic_mz = row.mono_mz
    feature_intensity_isolation_window = int(row.intensity)
    second_peak_mz = row.second_peak_mz
    feature_charge = int(row.charge)
    feature_envelope = row.envelope

    window = group_df.iloc[0]
    window_mz_lower = window.mz_lower
    window_mz_upper = window.mz_upper
    scan_width = int(window.ScanNumEnd - window.ScanNumBegin)
    wide_scan_lower = int(window.ScanNumBegin - scan_width)
    wide_scan_upper = int(window.ScanNumEnd + scan_width)
    fe_scan_lower = int(window.ScanNumBegin)
    fe_scan_upper = int(window.ScanNumEnd)
    precursor_id = int(window.Precursor)
    wide_rt_lower = group_df.retention_time_secs.min() - RT_BASE_PEAK_WIDTH_SECS
    wide_rt_upper = group_df.retention_time_secs.max() + RT_BASE_PEAK_WIDTH_SECS

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
            scan_apex = mz_centroid(scan_df.intensity.to_numpy(), scan_df.scan.to_numpy())
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
            rt_apex = mz_centroid(rt_df.intensity.to_numpy(), rt_df.retention_time_secs.to_numpy())
            rt_lower = wide_rt_lower
            rt_upper = wide_rt_upper

        # now that we have the full extent of the feature in RT, recalculate the feature m/z to gain the most mass accuracy (without points in saturation)
        raw_mono_points_without_saturation_df = wide_rt_monoisotopic_raw_points_df[wide_rt_monoisotopic_raw_points_df.intensity < SATURATION_INTENSITY]
        updated_feature_monoisotopic_mz = mz_centroid(raw_mono_points_without_saturation_df.intensity.to_numpy(), raw_mono_points_without_saturation_df.mz.to_numpy())
        feature_intensity_full_rt_extent = int(wide_rt_monoisotopic_raw_points_df.intensity.sum())

        result = (updated_feature_monoisotopic_mz, feature_charge, feature_intensity_isolation_window, feature_intensity_full_rt_extent, round(scan_apex,2), mobility_curve_fit, round(scan_lower,2), round(scan_upper,2), round(rt_apex,2), rt_curve_fit, round(rt_lower,2), round(rt_upper,2), precursor_id, feature_envelope)

    return result

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
    wide_rt_lower = group_df.retention_time_secs.min() - RT_BASE_PEAK_WIDTH_SECS
    wide_rt_upper = group_df.retention_time_secs.max() + RT_BASE_PEAK_WIDTH_SECS

    # get the ms1 frame IDs for the range of this precursor's ms2 frames
    isolation_window_ms1_frame_ids = find_ms1_frames_for_ms2_frame_range(ms2_frame_ids=list(group_df.Frame), number_either_side=RT_FRAGMENT_EVENT_DELTA_FRAMES)
    # all the ms1 frames
    ms1_frame_ids = tuple(ms1_frame_properties_df.frame_id)

    # load the cube's raw ms1 points
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    ms1_raw_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where mz >= {} and mz <= {} and scan >= {} and scan <= {} and retention_time_secs >= {} and retention_time_secs <= {} and frame_id in {}".format(window_mz_lower, window_mz_upper, wide_scan_lower, wide_scan_upper, wide_rt_lower, wide_rt_upper, ms1_frame_ids), db_conn)
    db_conn.close()

    # get the raw points constrained to the fragmentation event's extent in ms1 frames
    fe_raw_points_df = ms1_raw_points_df[ms1_raw_points_df.frame_id.isin(isolation_window_ms1_frame_ids)]

    ms1_bins = np.arange(start=window_mz_lower, stop=window_mz_upper+MS1_BIN_WIDTH, step=MS1_BIN_WIDTH)  # go slightly wider to accomodate the maximum value
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
            mz_cent = mz_centroid(intensity_values_for_bin, mz_values_for_bin)
            summed_intensity = intensity_values_for_bin.sum()
            binned_ms1_l.append((mz_cent,summed_intensity))

    binned_ms1_df = pd.DataFrame(binned_ms1_l, columns=['mz_centroid','summed_intensity'])

    if args.interim_data_mode:
        binned_ms1_df.to_csv('./ms1-peaks-group-{}-before-intensity-descent.csv'.format(group_number), index=False, header=True)

    # intensity descent
    mzs = binned_ms1_df.mz_centroid.to_numpy()
    intensities = binned_ms1_df.summed_intensity.to_numpy()
    ms1_peaks_l = []
    while len(mzs) > 0:
        # find the most intense point
        max_intensity_index = np.argmax(intensities)
        peak_mz = mzs[max_intensity_index]
        peak_mz_lower = peak_mz - MS1_PEAK_DELTA
        peak_mz_upper = peak_mz + MS1_PEAK_DELTA

        # get all the raw points within this m/z region
        peak_indexes = np.where((mzs >= peak_mz_lower) & (mzs <= peak_mz_upper))
        if len(peak_indexes) > 0:
            mz_cent = mz_centroid(intensities[peak_indexes], mzs[peak_indexes])
            summed_intensity = intensities[peak_indexes].sum()
            ms1_peaks_l.append(simple_peak(mz=mz_cent, intensity=summed_intensity))
            # remove the raw points assigned to this peak
            intensities = np.delete(intensities, peak_indexes)
            mzs = np.delete(mzs, peak_indexes)

    if args.interim_data_mode:
        l = []
        for p in ms1_peaks_l:
            l.append((p.mz, p.intensity))
        ms1_peaks_df = pd.DataFrame(l, columns=['mz','intensity'])
        ms1_peaks_df.to_csv('./ms1-peaks-group-{}-after-intensity-descent.csv'.format(group_number), index=False, header=True)

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
    ms1_characteristics_l = list(ms1_deconvoluted_peaks_df.apply(lambda row: collate_feature_characteristics(row, group_df, fe_raw_points_df, ms1_raw_points_df), axis=1).values)
    ms1_characteristics_l = [x for x in ms1_characteristics_l if x != None]  # clean up empty rows
    if len(ms1_characteristics_l) > 0:
        ms1_characteristics_df = pd.DataFrame(ms1_characteristics_l, columns=['monoisotopic_mz', 'charge', 'intensity', 'intensity_full_rt_extent', 'scan_apex', 'scan_curve_fit', 'scan_lower', 'scan_upper', 'rt_apex', 'rt_curve_fit', 'rt_lower', 'rt_upper', 'precursor_id', 'envelope'])
    else:
        ms1_characteristics_df = None

    return ms1_characteristics_df

# calculate the centroid, intensity of a bin
def calc_bin_centroid(bin_df):
    return pd.Series(np.array([bin_df.iloc[0].bin_idx, mz_centroid(bin_df.intensity.to_numpy(), bin_df.mz.to_numpy()), bin_df.intensity.sum()]), ['bin_idx', 'mz_centroid', 'summed_intensity'])

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

# given a centre mz and a feature, calculate the mz centroid and summed intensity from the raw points
def calculate_raw_peak_intensity_at_mz(centre_mz, feature):
    result = (-1, -1)

    mz_delta = standard_deviation(centre_mz) * NUMBER_OF_STD_DEV_MZ
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
        ms1_bins = np.arange(start=mz_lower, stop=mz_upper+MS1_BIN_WIDTH, step=MS1_BIN_WIDTH)  # go slightly wider to accomodate the maximum value
        ms1_raw_points_df['bin_idx'] = np.digitize(ms1_raw_points_df.mz, ms1_bins).astype(int)

        # find the peaks for each bin
        unresolved_peaks_df = ms1_raw_points_df.groupby(['bin_idx'], as_index=False).apply(calc_bin_centroid)

        # centroid and sum all the bin peaks, as we are interested in a narrow band of m/z
        mz_cent = mz_centroid(unresolved_peaks_df.summed_intensity.to_numpy(), unresolved_peaks_df.mz_centroid.to_numpy())
        summed_intensity = unresolved_peaks_df.summed_intensity.sum()
        result = (mz_cent, summed_intensity)

    return result

@ray.remote
def check_monoisotopic_peak(feature, idx, total):
    adjusted = False
    feature_d = feature.to_dict()
    original_phr_error = None
    candidate_phr_error = None
    observed_ratio = None

    # calculate the PHR error for peak 1 (first isotope) and peak 0 (what we think is the monoisotopic)
    observed_ratio = feature.envelope[1][1] / feature.envelope[0][1]
    monoisotopic_mass = feature.monoisotopic_mz * feature.charge
    expected_ratio = peak_ratio(monoisotopic_mass=monoisotopic_mass, peak_number=1, number_of_sulphur=0)
    if expected_ratio is not None:
        original_phr_error = (observed_ratio - expected_ratio) / expected_ratio
        if abs(original_phr_error) > MAX_MS1_PEAK_HEIGHT_RATIO_ERROR:
            # probably missed the monoisotopic - need to search for it
            expected_spacing_mz = CARBON_MASS_DIFFERENCE / feature.charge
            centre_mz = feature.monoisotopic_mz - expected_spacing_mz
            candidate_mz_centroid, candidate_raw_intensity = calculate_raw_peak_intensity_at_mz(centre_mz, feature)
            if (candidate_mz_centroid != -1) and (candidate_raw_intensity != -1):
                # get the raw intensity for the original monoisotope so we can calculate an accurate ratio
                original_mz, original_raw_intensity = calculate_raw_peak_intensity_at_mz(feature.monoisotopic_mz, feature)
                if (original_mz != -1) and (original_raw_intensity != -1):
                    candidate_ratio = original_raw_intensity / candidate_raw_intensity
                    candidate_phr_error = (candidate_ratio - expected_ratio) / expected_ratio
                    feature_d['candidate_phr_error'] = candidate_phr_error
                    if (abs(candidate_phr_error) <= abs(original_phr_error)) and (abs(candidate_phr_error) <= MAX_MS1_PEAK_HEIGHT_RATIO_ERROR):
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
                        feature_d['monoisotopic_mz'] = mz_centroid(np.array(envelope_deisotoped_intensities), np.array(envelope_deisotoped_mzs))
                        adjusted = True

    feature_d['mono_adjusted'] = adjusted
    feature_d['original_phr_error'] = original_phr_error
    feature_d['candidate_phr_error'] = candidate_phr_error
    feature_d['original_phr'] = observed_ratio
    return feature_d

# create the bins for mass defect windows in Da space
def generate_mass_defect_windows():
    bin_edges_l = []
    for nominal_mass in range(MASS_DEFECT_WINDOW_DA_MIN, MASS_DEFECT_WINDOW_DA_MAX):
        mass_centre = nominal_mass * 1.00048  # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3184890/
        width = 0.19 + (0.0001 * nominal_mass)
        lower_mass = mass_centre - (width / 2)
        upper_mass = mass_centre + (width / 2)
        bin_edges_l.append(lower_mass)
        bin_edges_l.append(upper_mass)
    bins = np.asarray(bin_edges_l)
    return bins

vectorise_mz = np.vectorize(lambda obj: obj.mz)
vectorise_intensity = np.vectorize(lambda obj: obj.intensity)

# return a point if, in its imaginary charge-3, charge-2, or charge-1 de-charged state, it fits inside at least one mass defect window
def remove_points_outside_mass_defect_windows(ms2_peaks_a, mass_defect_window_bins):
    mz_a = vectorise_mz(ms2_peaks_a)
    inside_mass_defect_window_a = np.full((len(mz_a)), False)
    for charge in [3,2,1]:
        decharged_mass_a = (mz_a * charge) - (PROTON_MASS * charge)
        decharged_mass_bin_indexes = np.digitize(decharged_mass_a, mass_defect_window_bins)  # an odd index means the point is inside a mass defect window
        mass_defect_window_indexes = (decharged_mass_bin_indexes % 2) == 1  # odd bin indexes are mass defect windows
        inside_mass_defect_window_a[mass_defect_window_indexes] = True
    result = ms2_peaks_a[inside_mass_defect_window_a]
    return result

#########################################################

# find ms1 features for each unique precursor ID
print("finding ms1 features")
start_time = time.time()
if args.small_set_mode:
    isolation_window_df = isolation_window_df[:20]
ms1_df_l = ray.get([find_features.remote(group_number=group_name, group_df=group_df) for group_name,group_df in isolation_window_df.groupby('Precursor')])
ms1_df = pd.concat(ms1_df_l)  # combines a list of dataframes into a single dataframe
# assign an ID to each feature
ms1_df.sort_values(by=['intensity'], ascending=False, inplace=True)
ms1_df["feature_id"] = np.arange(start=1, stop=len(ms1_df)+1)
stop_time = time.time()
print("new_ms1_features: {} seconds".format(round(stop_time-start_time,1)))
print("detected {} features".format(len(ms1_df)))

print("checking ms1 monoisotopic peaks")
start_time = time.time()
ms1_df.reset_index(drop=True, inplace=True)
checked_features_l = ray.get([check_monoisotopic_peak.remote(feature=feature, idx=idx, total=len(ms1_df)) for idx,feature in ms1_df.iterrows()])
checked_features_df = pd.DataFrame(checked_features_l)
checked_features_df.to_pickle(MS1_PEAK_PKL)
stop_time = time.time()
print("check_ms1_mono_peak: {} seconds".format(round(stop_time-start_time,1)))

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))

print("shutting down ray")
ray.shutdown()
