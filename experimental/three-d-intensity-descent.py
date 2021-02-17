import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import peakutils
from scipy import signal
import math
import os
import time
import argparse
import ray
import sqlite3
import shutil
import sys
import multiprocessing as mp
sys.path.append('/home/ubuntu/open-path/pda/packaged/')
from process_precursor_cuboid_ms1 import ms1
import configparser
from configparser import ExtendedInterpolation
from argparse import Namespace

# define a straight line to exclude the charge-1 cloud
def scan_coords_for_single_charge_region(mz_lower, mz_upper):
    scan_for_mz_lower = -1 * ((1.2 * mz_lower) - 1252)
    scan_for_mz_upper = -1 * ((1.2 * mz_upper) - 1252)
    return (scan_for_mz_lower,scan_for_mz_upper)

# a distance metric for points within an isotope
def point_metric(r1, r2):
    mz_1 = r1[0]
    scan_1 = r1[1]
    mz_2 = r2[0]
    scan_2 = r2[1]
    if (abs(mz_1 - mz_2) <= 0.1) and (abs(scan_1 - scan_2) <= 5):
        result = 0.5
    else:
        result = 10
    return result

# a distance metric for isotopes within a series
def isotope_metric(r1, r2):
    mz_1 = r1[0]
    scan_1 = r1[1]
    mz_2 = r2[0]
    scan_2 = r2[1]
    if (abs(mz_1 - mz_2) <= 0.8) and (abs(mz_1 - mz_2) > 0.1) and (abs(scan_1 - scan_2) <= 10):
        result = 0.5
    else:
        result = 10
    # print('r1={}, r2={}, result={}'.format(r1,r2,result))
    return result

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(args.proportion_of_cores_to_use * number_of_cores)
    return number_of_workers

# set up the indexes we need for queries
def create_indexes(db_file_name):
    db_conn = sqlite3.connect(db_file_name)
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_three_d_1 on frames (frame_type, mz, retention_time_secs)")
    db_conn.close()

# returns a dataframe with the frame properties
def load_frame_properties(converted_db_name):
    # get all the isolation windows
    db_conn = sqlite3.connect(converted_db_name)
    frames_properties_df = pd.read_sql_query("select * from frame_properties order by Id ASC;", db_conn)
    db_conn.close()

    print("loaded {} frame_properties from {}".format(len(frames_properties_df), converted_db_name))
    return frames_properties_df

# find the closest lower ms1 frame_id, and the closest upper ms1 frame_id
def find_closest_ms1_frame_to_rt(frames_properties_df, retention_time_secs):
    # find the frame ids within this range of RT
    df = frames_properties_df[(frames_properties_df.Time > retention_time_secs) & (frames_properties_df.MsMsType == FRAME_TYPE_MS1)]
    if len(df) > 0:
        closest_ms1_frame_above_rt = df.Id.min()
    else:
        # couldn't find an ms1 frame above this RT, so just use the last one
        closest_ms1_frame_above_rt = frames_properties_df[(frames_properties_df.MsMsType == FRAME_TYPE_MS1)].Id.max()
    df = frames_properties_df[(frames_properties_df.Time < retention_time_secs) & (frames_properties_df.MsMsType == FRAME_TYPE_MS1)]
    if len(df) > 0:
        closest_ms1_frame_below_rt = df.Id.max()
    else:
        # couldn't find an ms1 frame below this RT, so just use the first one
        closest_ms1_frame_below_rt = frames_properties_df[(frames_properties_df.MsMsType == FRAME_TYPE_MS1)].Id.min()
    result = {}
    result['below'] = closest_ms1_frame_below_rt
    result['above'] = closest_ms1_frame_above_rt
    return result

# process a segment of this run's data, and return a list of precursor cuboids
@ray.remote
def find_precursor_cuboids(segment_mz_lower, segment_mz_upper):
    isotope_cluster_retries = 0
    point_cluster_retries = 0

    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    raw_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and mz >= {} and mz <= {} and retention_time_secs >= {} and retention_time_secs <= {} and intensity >= {}".format(FRAME_TYPE_MS1, segment_mz_lower, segment_mz_upper, args.rt_lower, args.rt_upper, INTENSITY_THRESHOLD), db_conn)
    db_conn.close()

    raw_df.reset_index(drop=True, inplace=True)

    # assign each point a unique identifier
    raw_df['point_id'] = raw_df.index

    # load the frame properties
    frame_properties_df = load_frame_properties(CONVERTED_DATABASE_NAME)

    precursor_cuboids_l = []
    anchor_point_s = raw_df.loc[raw_df.intensity.idxmax()]
    while anchor_point_s.intensity >= MIN_ANCHOR_POINT_INTENSITY:
        mz_lower = anchor_point_s.mz - ANCHOR_POINT_MZ_LOWER_OFFSET
        mz_upper = anchor_point_s.mz + ANCHOR_POINT_MZ_UPPER_OFFSET
        scan_lower = anchor_point_s.scan - ANCHOR_POINT_SCAN_LOWER_OFFSET
        scan_upper = anchor_point_s.scan + ANCHOR_POINT_SCAN_UPPER_OFFSET

        candidate_region_df = raw_df[(raw_df.intensity >= INTENSITY_THRESHOLD) & (raw_df.frame_id == anchor_point_s.frame_id) & (raw_df.mz >= mz_lower) & (raw_df.mz <= mz_upper) & (raw_df.scan >= scan_lower) & (raw_df.scan <= scan_upper)].copy()

        peak_mz_lower = anchor_point_s.mz-MS1_PEAK_DELTA
        peak_mz_upper = anchor_point_s.mz+MS1_PEAK_DELTA

        peak_df = candidate_region_df[(candidate_region_df.mz >= peak_mz_lower) & (candidate_region_df.mz <= peak_mz_upper)]

        scan_0_df = peak_df.groupby(['scan'], as_index=False).intensity.sum()
        scan_0_df.sort_values(by=['scan'], ascending=True, inplace=True)

        # filter the points
        scan_0_df['filtered_intensity'] = scan_0_df.intensity  # set the default
        window_length = 21
        if len(scan_0_df) > window_length:
            try:
                scan_0_df['filtered_intensity'] = signal.savgol_filter(scan_0_df.intensity, window_length=window_length, polyorder=2)
                filtered = True
            except:
                filtered = False
        else:
            filtered = False

        peak_idxs = peakutils.indexes(scan_0_df.filtered_intensity.values, thres=0.05, min_dist=10/2, thres_abs=False)
        peak_x_l = scan_0_df.iloc[peak_idxs].scan.to_list()
        peaks_df = scan_0_df[scan_0_df.scan.isin(peak_x_l)]

        valley_idxs = peakutils.indexes(-scan_0_df.filtered_intensity.values, thres=0.05, min_dist=10/2, thres_abs=False)
        valley_x_l = scan_0_df.iloc[valley_idxs].scan.to_list()
        valleys_df = scan_0_df[scan_0_df.scan.isin(valley_x_l)]

        upper_x = valleys_df[valleys_df.scan > anchor_point_s.scan].scan.min()
        if math.isnan(upper_x):
            upper_x = scan_0_df.scan.max()
        lower_x = valleys_df[valleys_df.scan < anchor_point_s.scan].scan.max()
        if math.isnan(lower_x):
            lower_x = scan_0_df.scan.min()

        scan_lower = lower_x
        scan_upper = upper_x

        # trim the candidate region to account for the selected peak in mobility
        candidate_region_df = candidate_region_df[(candidate_region_df.scan >= lower_x) & (candidate_region_df.scan <= upper_x)]

        # segment the raw data to reveal the isotopes in the feature
        X = candidate_region_df[['mz','scan']].values

        # cluster the points
        dbscan = DBSCAN(eps=1, min_samples=3, metric=point_metric)
        clusters = dbscan.fit_predict(X)
        candidate_region_df['cluster'] = clusters
        anchor_point_cluster = candidate_region_df[candidate_region_df.point_id == anchor_point_s.point_id].iloc[0].cluster

        number_of_point_clusters = len(candidate_region_df[candidate_region_df.cluster >= 0].cluster.unique())

        if (number_of_point_clusters > 0) and (anchor_point_cluster >= 0):

            # calculate the cluster centroids
            centroids_l = []
            for group_name,group_df in candidate_region_df.groupby(['cluster'], as_index=False):
                if group_name >= 0:
                    mz_centroid = peakutils.centroid(group_df.mz, group_df.intensity)
                    scan_centroid = peakutils.centroid(group_df.scan, group_df.intensity)
                    centroids_l.append((group_name, mz_centroid, scan_centroid))
            centroids_df = pd.DataFrame(centroids_l, columns=['cluster','mz','scan'])

            X = centroids_df[['mz','scan']].values

            # cluster the isotopes
            dbscan = DBSCAN(eps=1, min_samples=2, metric=isotope_metric)  # minimum isotopes to form a series
            clusters = dbscan.fit_predict(X)
            centroids_df['isotope_cluster'] = clusters

            number_of_isotope_clusters = len(centroids_df[centroids_df.isotope_cluster >= 0].isotope_cluster.unique())
            anchor_point_isotope_cluster = centroids_df[(centroids_df.cluster == anchor_point_cluster)].iloc[0].isotope_cluster

            if (number_of_isotope_clusters > 0) and (anchor_point_isotope_cluster >= 0):
                candidate_region_df = pd.merge(candidate_region_df, centroids_df[['cluster','isotope_cluster']], how='left', left_on=['cluster'], right_on=['cluster'])
                candidate_region_df.fillna(value=-1, inplace=True)
                candidate_region_df.isotope_cluster = candidate_region_df.isotope_cluster.astype(int)

                # we now have the 2D extent of the feature - take that extent through time and see if we can cluster the centroids in time

                # get the extent of the isotope cluster in m/z and mobility
                points_in_cluster_df = candidate_region_df[(candidate_region_df.isotope_cluster == anchor_point_isotope_cluster)]
                mz_lower = points_in_cluster_df.mz.min()
                mz_upper = points_in_cluster_df.mz.max()
                scan_lower = points_in_cluster_df.scan.min()
                scan_upper = points_in_cluster_df.scan.max()

                # get the left-most peak in the isotope cluster
                monoisotopic_cluster_s = centroids_df.loc[centroids_df[(centroids_df.isotope_cluster == anchor_point_isotope_cluster)].mz.idxmin()]
                mono_raw_points_df = raw_df[(raw_df.mz >= monoisotopic_cluster_s.mz-MS1_PEAK_DELTA) & (raw_df.mz <= monoisotopic_cluster_s.mz+MS1_PEAK_DELTA) & (raw_df.scan >= scan_lower) & (raw_df.scan <= scan_upper) & (raw_df.retention_time_secs >= anchor_point_s.retention_time_secs-RT_BASE_PEAK_WIDTH) & (raw_df.retention_time_secs <= anchor_point_s.retention_time_secs+RT_BASE_PEAK_WIDTH)]
                rt_0_df = mono_raw_points_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
                rt_0_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)

                # filter the points
                rt_0_df['filtered_intensity'] = rt_0_df.intensity  # set the default
                window_length = 11
                if len(rt_0_df) > window_length:
                    try:
                        rt_0_df['filtered_intensity'] = signal.savgol_filter(rt_0_df.intensity, window_length=window_length, polyorder=3)
                        filtered = True
                    except:
                        filtered = False
                else:
                    filtered = False

                peak_idxs = peakutils.indexes(rt_0_df.filtered_intensity.values, thres=0.05, min_dist=10/2, thres_abs=False)
                peak_x_l = rt_0_df.iloc[peak_idxs].retention_time_secs.to_list()
                peaks_df = rt_0_df[rt_0_df.retention_time_secs.isin(peak_x_l)]

                valley_idxs = peakutils.indexes(-rt_0_df.filtered_intensity.values, thres=0.05, min_dist=10/8, thres_abs=False)
                valley_x_l = rt_0_df.iloc[valley_idxs].retention_time_secs.to_list()
                valleys_df = rt_0_df[rt_0_df.retention_time_secs.isin(valley_x_l)]

                upper_x = valleys_df[valleys_df.retention_time_secs > anchor_point_s.retention_time_secs].retention_time_secs.min()
                if math.isnan(upper_x):
                    upper_x = rt_0_df.retention_time_secs.max()
                lower_x = valleys_df[valleys_df.retention_time_secs < anchor_point_s.retention_time_secs].retention_time_secs.max()
                if math.isnan(lower_x):
                    lower_x = rt_0_df.retention_time_secs.min()

                rt_lower = lower_x
                rt_upper = upper_x

                # make sure the RT extent isn't too extreme
                if (rt_upper - rt_lower) > RT_BASE_PEAK_WIDTH:
                    rt_lower = anchor_point_s.retention_time_secs - (RT_BASE_PEAK_WIDTH / 2)
                    rt_upper = anchor_point_s.retention_time_secs + (RT_BASE_PEAK_WIDTH / 2)

                # hold on to the region for the next step
                candidate_region_d = candidate_region_df.to_dict('records')

                # add this cuboid to the list
                precursor_cuboids_l.append((mz_lower, mz_upper, scan_lower, scan_upper, rt_lower, rt_upper, candidate_region_d))
                # print('.', end='', flush=True)
                isotope_cluster_retries = 0

                # get the point ids
                points_to_remove_l = raw_df[(raw_df.mz >= mz_lower) & (raw_df.mz <= mz_upper) & (raw_df.scan >= scan_lower) & (raw_df.scan <= scan_upper) & (raw_df.retention_time_secs >= rt_lower) & (raw_df.retention_time_secs <= rt_upper)].point_id.tolist()
                # set the intensity so we don't process them again
                raw_df.loc[raw_df.point_id.isin(points_to_remove_l), 'intensity'] = PROCESSED_INTENSITY_INDICATOR

                # # remove the points in each isotope of the series, and the other points through time in the same isotope
                # clusters_to_remove_l = centroids_df[(centroids_df.isotope_cluster == anchor_point_isotope_cluster)].cluster.tolist()
                # for c in clusters_to_remove_l:
                #     points_df = candidate_region_df[(candidate_region_df.cluster == c)]
                #     # find the bounds of the points in this cluster; we need to search for them because the points in other frames haven't been clustered
                #     p_mz_lower = points_df.mz.min()
                #     p_mz_upper = points_df.mz.max()
                #     p_scan_lower = points_df.scan.min()
                #     p_scan_upper = points_df.scan.max()
                #     p_rt_lower = rt_lower
                #     p_rt_upper = rt_upper
                #     # get the point ids
                #     points_to_remove_l = raw_df[(raw_df.mz >= p_mz_lower) & (raw_df.mz <= p_mz_upper) & (raw_df.scan >= p_scan_lower) & (raw_df.scan <= p_scan_upper) & (raw_df.retention_time_secs >= p_rt_lower) & (raw_df.retention_time_secs <= p_rt_upper)].point_id.tolist()
                #     # print('removing isotope cluster {}, cluster {}, {} points'.format(anchor_point_isotope_cluster, c, len(points_to_remove_l)))
                #     # set the intensity so we don't process them again
                #     raw_df.loc[raw_df.point_id.isin(points_to_remove_l), 'intensity'] = PROCESSED_INTENSITY_INDICATOR
            else:
                # just remove the anchor point's cluster because we could not form a series
                # print('number_of_isotope_clusters: {}, anchor_point_isotope_cluster: {}, anchor_point_cluster: {}'.format(number_of_isotope_clusters, anchor_point_isotope_cluster, anchor_point_cluster))
                # print(centroids_df)
                # mark the points assigned to the anchor point's cluster so we don't process them again
                clusters_to_remove_l = [anchor_point_cluster]
                points_to_remove_l = candidate_region_df[candidate_region_df.cluster.isin(clusters_to_remove_l)].point_id.tolist()
                # print('removing clusters {}, {} points'.format(clusters_to_remove_l, len(points_to_remove_l)))
                raw_df.loc[raw_df.point_id.isin(points_to_remove_l), 'intensity'] = PROCESSED_INTENSITY_INDICATOR
                # print('_', end='', flush=True)
                isotope_cluster_retries += 1
                if isotope_cluster_retries >= MAX_ISOTOPE_CLUSTER_RETRIES:
                    # print('max isotope cluster retries reached for mz={} to {}'.format(segment_mz_lower, segment_mz_upper))
                    break
        else:
            points_to_remove_l = [anchor_point_s.point_id]
            raw_df.loc[raw_df.point_id.isin(points_to_remove_l), 'intensity'] = PROCESSED_INTENSITY_INDICATOR
            # print('x', end='', flush=True)
            point_cluster_retries += 1
            if point_cluster_retries >= MAX_POINT_CLUSTER_RETRIES:
                # print('max point cluster retries reached for mz={} to {}'.format(segment_mz_lower, segment_mz_upper))
                break

        # find the next anchor point
        anchor_point_s = raw_df.loc[raw_df.intensity.idxmax()]

    # return what we found in this segment
    print('found {} cuboids for mz={} to {}'.format(len(precursor_cuboids_l), segment_mz_lower, segment_mz_upper))
    return precursor_cuboids_l



# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

MS1_PEAK_DELTA = 0.1
RT_BASE_PEAK_WIDTH = 10

MIN_ANCHOR_POINT_INTENSITY = 200

ANCHOR_POINT_MZ_LOWER_OFFSET = 0.6   # one isotope for charge-2 plus a little bit more
ANCHOR_POINT_MZ_UPPER_OFFSET = 3.0   # six isotopes for charge-2 plus a little bit more

ANCHOR_POINT_SCAN_LOWER_OFFSET = 100
ANCHOR_POINT_SCAN_UPPER_OFFSET = 100

INTENSITY_THRESHOLD = 50
PROCESSED_INTENSITY_INDICATOR = -1

MAX_ISOTOPE_CLUSTER_RETRIES = 1000
MAX_POINT_CLUSTER_RETRIES = 10

# constrain the data to re-run the same feature for debugging
MZ_MIN_DEBUG, MZ_MAX_DEBUG = (764.4201958278368, 765.9385489808168)
SCAN_MIN_DEBUG, SCAN_MAX_DEBUG = (400, 460)
RT_LOWER_DEBUG, RT_UPPER_DEBUG = (2103.4829066671086, 2110.362298691853)


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
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], help='The Ray mode to use.', required=True)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.9, help='Proportion of the machine\'s cores to use for this program.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./open-path/pda/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-fdo', '--feature_detect_only', action='store_true', help='Use an existing cuboids file.')
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

CUBOIDS_DIR = "{}/precursor-cuboids-3did".format(EXPERIMENT_DIR)
CUBOIDS_FILE = '{}/exp-{}-run-{}-mz-{}-{}-precursor-cuboids.pkl'.format(CUBOIDS_DIR, args.experiment_name, args.run_name, args.mz_lower, args.mz_upper)

if args.feature_detect_only:
    # make sure we have what we need
    if not os.path.exists(CUBOIDS_DIR):
        print("The cuboids directory is required but doesn't exist: {}".format(CUBOIDS_DIR))
        sys.exit(1)
    if not os.path.isfile(CUBOIDS_FILE):
        print("The cuboids file is required but doesn't exist: {}".format(CUBOIDS_FILE))
        sys.exit(1)
    precursor_cuboids_df = pd.read_pickle(CUBOIDS_FILE)
    print('loaded {} precursor cuboids from {}'.format(len(precursor_cuboids_df), CUBOIDS_FILE))
else:
    # set up the output directory
    if os.path.exists(CUBOIDS_DIR):
        shutil.rmtree(CUBOIDS_DIR)
    os.makedirs(CUBOIDS_DIR)

    # set up the output file
    if os.path.isfile(CUBOIDS_FILE):
        os.remove(CUBOIDS_FILE)

    # set up Ray
    print("setting up Ray")
    if not ray.is_initialized():
        if args.ray_mode == "cluster":
            ray.init(num_cpus=number_of_workers())
        else:
            ray.init(local_mode=True)

    print('setting up indexes on {}'.format(CONVERTED_DATABASE_NAME))
    create_indexes(CONVERTED_DATABASE_NAME)

    mz_range = args.mz_upper - args.mz_lower
    NUMBER_OF_MZ_SEGMENTS = (mz_range // args.mz_width_per_segment) + (mz_range % args.mz_width_per_segment > 0)  # thanks to https://stackoverflow.com/a/23590097/1184799

    print('finding precursor cuboids')
    cuboids_l = ray.get([find_precursor_cuboids.remote(segment_mz_lower=args.mz_lower+(i*args.mz_width_per_segment), segment_mz_upper=args.mz_lower+(i*args.mz_width_per_segment)+args.mz_width_per_segment) for i in range(NUMBER_OF_MZ_SEGMENTS)])
    cuboids_l = [item for sublist in cuboids_l for item in sublist]  # cuboids_l is a list of lists, so we need to flatten it

    # assign each cuboid a unique identifier
    precursor_cuboids_df = pd.DataFrame(cuboids_l, columns=['mz_lower', 'mz_upper', 'scan_lower', 'scan_upper', 'rt_lower', 'rt_upper', 'candidate_region_d'])
    precursor_cuboids_df['precursor_cuboid_id'] = precursor_cuboids_df.index

    # ... and save them in a file
    print()
    print('saving {} precursor cuboids to {}'.format(len(precursor_cuboids_df), CUBOIDS_FILE))
    precursor_cuboids_df.to_pickle(CUBOIDS_FILE)

# parse the config file
config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read(args.ini_file)

# load the frame properties
frames_properties_df = load_frame_properties(CONVERTED_DATABASE_NAME)

# use the ms1 function to perform the feature detection step
ms1_args = Namespace()
ms1_args.experiment_name = 'P3856T'
ms1_args.run_name = args.run_name
ms1_args.MS1_PEAK_DELTA = config.getfloat('ms1', 'MS1_PEAK_DELTA')
ms1_args.SATURATION_INTENSITY = config.getfloat('common', 'SATURATION_INTENSITY')
ms1_args.MAX_MS1_PEAK_HEIGHT_RATIO_ERROR = config.getfloat('ms1', 'MAX_MS1_PEAK_HEIGHT_RATIO_ERROR')
ms1_args.PROTON_MASS = config.getfloat('common', 'PROTON_MASS')
ms1_args.INSTRUMENT_RESOLUTION = config.getfloat('common', 'INSTRUMENT_RESOLUTION')
ms1_args.NUMBER_OF_STD_DEV_MZ = config.getfloat('ms1', 'NUMBER_OF_STD_DEV_MZ')
ms1_args.FEATURES_DIR = '{}/features-3did/{}'.format(args.experiment_base_dir, args.run_name)

# set up the output directory
if os.path.exists(ms1_args['FEATURES_DIR']):
    shutil.rmtree(ms1_args['FEATURES_DIR'])
os.makedirs(ms1_args['FEATURES_DIR'])

# for each cuboid, find the features
for row in precursor_cuboids_df.itertuples():
    # create the metadata record
    cuboid_metadata = {}
    cuboid_metadata['precursor_id'] = row.precursor_cuboid_id
    cuboid_metadata['window_mz_lower'] = row.mz_lower
    cuboid_metadata['window_mz_upper'] = row.mz_upper
    cuboid_metadata['wide_mz_lower'] = row.mz_lower
    cuboid_metadata['wide_mz_upper'] = row.mz_upper
    cuboid_metadata['window_scan_width'] = row.scan_upper - row.scan_lower
    cuboid_metadata['fe_scan_lower'] = row.scan_lower
    cuboid_metadata['fe_scan_upper'] = row.scan_upper
    cuboid_metadata['wide_scan_lower'] = row.scan_lower
    cuboid_metadata['wide_scan_upper'] = row.scan_upper
    cuboid_metadata['wide_rt_lower'] = row.rt_lower
    cuboid_metadata['wide_rt_upper'] = row.rt_upper
    cuboid_metadata['fe_ms1_frame_lower'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=row.rt_lower)['below']
    cuboid_metadata['fe_ms1_frame_upper'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=row.rt_upper)['above']
    cuboid_metadata['fe_ms2_frame_lower'] = None
    cuboid_metadata['fe_ms2_frame_upper'] = None
    cuboid_metadata['wide_frame_lower'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=row.rt_lower)['below']
    cuboid_metadata['wide_frame_upper'] = find_closest_ms1_frame_to_rt(frames_properties_df=frames_properties_df, retention_time_secs=row.rt_upper)['above']
    cuboid_metadata['number_of_windows'] = 1

    # load the raw points
    ms1_points_df = pd.DataFrame.from_dict(row.candidate_region_d)

    # adjust the args
    ms1_args.precursor_id = row.precursor_cuboid_id
    ms1_args.FEATURES_FILE = "{}/exp-{}-run-{}-features-precursor-{}.pkl".format(ms1_args['FEATURES_DIR'], ms1_args['experiment_name'], ms1_args['run_name'], ms1_args['precursor_id'])

    # find the features in this precursor cuboid
    _ = ms1(precursor_metadata=cuboid_metadata, ms1_points_df=ms1_points_df, args=ms1_args)


stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
