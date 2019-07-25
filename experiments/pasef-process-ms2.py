import sqlite3
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from ms_deisotope import deconvolute_peaks, averagine, scoring
from ms_deisotope.deconvolution import peak_retention_strategy
from numba import njit
import configparser
from configparser import ExtendedInterpolation
import os.path
import ray
import time
import argparse

# A refined approach for processing ms2 spectra independently of the features, associating them in a subsequent step. See 'associating ms2 spectra with features'.

# so we can use profiling without removing @profile
try:
    profile
except NameError:
    profile = lambda x: x

parser = argparse.ArgumentParser(description='Deconvolute ms2 spectra for PASEF isolation windows.')
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

MS2_PEAK_DELTA = config.getfloat('ms2', 'MS2_PEAK_DELTA')
MS2_MZ_ISOLATION_WINDOW_EXTENSION = config.getfloat('ms2', 'MS2_MZ_ISOLATION_WINDOW_EXTENSION')

CONVERTED_DATABASE_NAME = config.get(args.operating_system, 'CONVERTED_DATABASE_NAME')
RAW_DATABASE_NAME = config.get(args.operating_system, 'RAW_DATABASE_NAME')
DECONVOLUTED_MS2_PKL = config.get(args.operating_system, 'DECONVOLUTED_MS2_PKL')

# create the bins for mass defect windows in Da space
# @njit(fastmath=True)
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

# return a point if, in its imaginary charge-3, charge-2, or charge-1 de-charged state, it fits inside at least one mass defect window
# ms2_peaks_a is a numpy array of [mz,intensity]
# @njit(fastmath=True)
def remove_points_outside_mass_defect_windows(ms2_peaks_a, mass_defect_window_bins):
    mz_a = ms2_peaks_a[:,0]
    inside_mass_defect_window_a = np.full((len(mz_a)), False)
    for charge in [3,2,1]:
        decharged_mass_a = (mz_a * charge) - (PROTON_MASS * charge)
        decharged_mass_bin_indexes = np.digitize(decharged_mass_a, mass_defect_window_bins)  # an odd index means the point is inside a mass defect window
        mass_defect_window_indexes = (decharged_mass_bin_indexes % 2) == 1  # odd bin indexes are mass defect windows
        inside_mass_defect_window_a[mass_defect_window_indexes] = True
    result = ms2_peaks_a[inside_mass_defect_window_a]
    return result

@njit(fastmath=True)
def mz_centroid(_int_f, _mz_f):
    return ((_int_f/_int_f.sum()) * _mz_f).sum()

# ms2_peaks_a is a numpy array of [mz,intensity]
# returns a nunpy array of [mz_centroid,summed_intensity]
@njit(fastmath=True)
def ms2_intensity_descent(ms2_peaks_a):
    # intensity descent
    ms2_peaks_l = []
    while len(ms2_peaks_a) > 0:
        # find the most intense point
        max_intensity_index = np.argmax(ms2_peaks_a[:,1])
        peak_mz = ms2_peaks_a[max_intensity_index,0]
        peak_mz_lower = peak_mz - MS2_PEAK_DELTA
        peak_mz_upper = peak_mz + MS2_PEAK_DELTA

        # get all the raw points within this m/z region
        peak_indexes = np.where((ms2_peaks_a[:,0] >= peak_mz_lower) & (ms2_peaks_a[:,0] <= peak_mz_upper))
        if len(peak_indexes) > 0:
            mz_cent = mz_centroid(ms2_peaks_a[peak_indexes,1], ms2_peaks_a[peak_indexes,0])
            summed_intensity = ms2_peaks_a[peak_indexes,1].sum()
            ms2_peaks_l.append((mz_cent, summed_intensity))
            # remove the raw points assigned to this peak
            ms2_peaks_a = np.delete(ms2_peaks_a, peak_indexes, axis=0)
    return np.array(ms2_peaks_l)

# return a list of deconvoluted spectra for this precursor
@ray.remote
def process_ms2(precursor_id, precursor_group_df):
    deconvoluted_peaks_l = []
    # determine the target raw data
    scan_lower = precursor_group_df.iloc[0].ScanNumBegin
    scan_upper = precursor_group_df.iloc[0].ScanNumEnd
    ms2_frame_ids = tuple(precursor_group_df.Frame)
    if len(ms2_frame_ids) == 1:
        ms2_frame_ids = "({})".format(ms2_frame_ids[0])
    # extract the raw data
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    ms2_raw_points_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} and scan >= {} and scan <= {} and intensity > 0 order by mz".format(ms2_frame_ids, scan_lower, scan_upper), db_conn)
    db_conn.close()
    # remove the points that are not within a mass defect window
    raw_points_a = ms2_raw_points_df[['mz','intensity']].to_numpy()
    filtered_raw_points_a = remove_points_outside_mass_defect_windows(raw_points_a, mass_defect_window_bins)
    # perform intensity descent to resolve peaks
    peaks_a = ms2_intensity_descent(filtered_raw_points_a)
    # deconvolute the spectra
    peaks_l = list(map(tuple, peaks_a))
    deconvoluted_peaks, _ = deconvolute_peaks(peaks_l, use_quick_charge=True, averagine=averagine.peptide, charge_range=(1,5), scorer=scoring.MSDeconVFitter(minimum_score=8, mass_error_tolerance=0.1), error_tolerance=4e-5, truncate_after=0.8, retention_strategy=peak_retention_strategy.TopNRetentionStrategy(n_peaks=100, base_peak_coefficient=1e-6, max_mass=1800.0))
    # package the spectra as a list
    for peak in deconvoluted_peaks:
        deconvoluted_peaks_l.append((precursor_id, round(peak.neutral_mass+PROTON_MASS, 4), int(peak.charge), peak.intensity, peak.score, peak.signal_to_noise))
    return deconvoluted_peaks_l


#############################
# check we have the required files
if not os.path.isfile(RAW_DATABASE_NAME):
    print("The raw database doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

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

# make sure the right indexes are created in the source database
print("Setting up indexes on {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
src_c = db_conn.cursor()
src_c.execute("create index if not exists idx_pasef_ms2_frames_1 on frames (frame_id, scan, intensity)")
src_c.execute("create index if not exists idx_pasef_frame_properties_1 on frame_properties (retention_time_secs, collision_energy)")
db_conn.close()

start_run = time.time()

# get all the isolation windows
db_conn = sqlite3.connect(RAW_DATABASE_NAME)
isolation_window_df = pd.read_sql_query("select * from PasefFrameMsMsInfo", db_conn)
db_conn.close()

# get the ms2 frames
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
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

# generate the mass defect windows
mass_defect_window_bins = generate_mass_defect_windows()

# process the ms2 spectra
ms2_df_l = ray.get([process_ms2.remote(precursor_id=precursor_id, precursor_group_df=precursor_group_df) for precursor_id,precursor_group_df in isolation_window_df.groupby('Precursor')])
flattened_ms2_df_l = [item for sublist in ms2_df_l for item in sublist]

print("writing {} peaks to {}".format(len(flattened_ms2_df_l), DECONVOLUTED_MS2_PKL))
ms2_deconvoluted_peaks_df = pd.DataFrame(flattened_ms2_df_l, columns=['precursor','mz','charge','intensity','score','SN'])
ms2_deconvoluted_peaks_df.to_pickle(DECONVOLUTED_MS2_PKL)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))

print("shutting down ray")
ray.shutdown()
