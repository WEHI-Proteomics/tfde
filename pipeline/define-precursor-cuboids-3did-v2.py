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
import sys
import multiprocessing as mp
import pickle
import configparser
from configparser import ExtendedInterpolation

# set up the indexes we need for queries
def create_indexes(db_file_name):
    db_conn = sqlite3.connect(db_file_name)
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_extract_cuboids_1 on frames (frame_type,retention_time_secs,scan,mz)")
    db_conn.close()

# a distance metric for points within an isotope
def point_metric(r1, r2):
    # mz_1 = r1[0]
    # scan_1 = r1[1]
    # mz_2 = r2[0]
    # scan_2 = r2[1]
    return 0.5 if ((abs(r1[0] - r2[0]) <= 0.1) and (abs(r1[1] - r2[1]) <= 5)) else 10

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

# determine the maximum filter length for the number of points
def find_filter_length(number_of_points):
    filter_lengths = [51,11,5]  # must be a positive odd number, greater than the polynomial order, and less than the number of points to be filtered
    return filter_lengths[next(x[0] for x in enumerate(filter_lengths) if x[1] < number_of_points)]

# find the closest frame id for a point in time
def frame_id_for_rt(voxel_df, rt):
    df = voxel_df.copy()
    df['rt_delta'] = abs(voxel_df.retention_time_secs - rt)
    df.sort_values(by=['rt_delta'], ascending=True, inplace=True)
    return df.iloc[0].frame_id

# define a straight line to exclude the charge-1 cloud
def scan_coords_for_single_charge_region(mz_lower, mz_upper):
    scan_for_mz_lower = max(int(-1 * ((1.2 * mz_lower) - 1252)), 0)
    scan_for_mz_upper = max(int(-1 * ((1.2 * mz_upper) - 1252)), 0)
    return {'scan_for_mz_lower':scan_for_mz_lower, 'scan_for_mz_upper':scan_for_mz_upper}

# process a segment of this run's data, and return a list of precursor cuboids
# @ray.remote
def find_precursor_cuboids(segment_mz_lower, segment_mz_upper):
    precursor_cuboids_l = []

    # find out where the charge-1 cloud ends and only include points below it (i.e. include points with a higher scan)
    scan_limit = scan_coords_for_single_charge_region(mz_lower=segment_mz_lower, mz_upper=segment_mz_upper)['scan_for_mz_upper']

    # load the raw points for this m/z segment
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    raw_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} and scan >= {} and mz >= {} and mz <= {}".format(FRAME_TYPE_MS1, args.rt_lower, args.rt_upper, scan_limit, segment_mz_lower, segment_mz_upper), db_conn)
    db_conn.close()

    if len(raw_df) > 0:
        # assign each point a unique identifier
        raw_df.reset_index(drop=True, inplace=True)  # just in case
        raw_df['point_id'] = raw_df.index

        # define bins
        rt_bins = pd.interval_range(start=raw_df.retention_time_secs.min(), end=raw_df.retention_time_secs.max()+RT_BIN_SIZE, freq=RT_BIN_SIZE)
        scan_bins = pd.interval_range(start=raw_df.scan.min(), end=raw_df.scan.max()+SCAN_BIN_SIZE, freq=SCAN_BIN_SIZE)
        mz_bins = pd.interval_range(start=raw_df.mz.min(), end=raw_df.mz.max()+MZ_BIN_SIZE, freq=MZ_BIN_SIZE)

        # assign raw points to their bins
        raw_df['rt_bin'] = pd.cut(raw_df.retention_time_secs, bins=rt_bins)
        raw_df['scan_bin'] = pd.cut(raw_df.scan, bins=scan_bins)
        raw_df['mz_bin'] = pd.cut(raw_df.mz, bins=mz_bins)

        # sum the intensities in each bin
        summary_df = raw_df.groupby(['mz_bin','scan_bin','rt_bin'], as_index=False, sort=False).intensity.sum()
        summary_df.dropna(subset = ['intensity'], inplace=True)
        summary_df.sort_values(by=['intensity'], ascending=False, inplace=True)

        if args.visualise:
            summary_df = summary_df.head(n=1000).sample(n=1)

        # process each voxel by decreasing intensity
        for row in summary_df.itertuples():
            # get the attributes of this voxel
            voxel_mz_lower = row.mz_bin.left
            voxel_mz_upper = row.mz_bin.right
            voxel_mz_midpoint = row.mz_bin.mid
            voxel_scan_lower = row.scan_bin.left
            voxel_scan_upper = row.scan_bin.right
            voxel_scan_midpoint = row.scan_bin.mid
            voxel_rt_lower = row.rt_bin.left
            voxel_rt_upper = row.rt_bin.right
            voxel_rt_midpoint = row.rt_bin.mid
            voxel_df = raw_df[(raw_df.mz >= voxel_mz_lower) & (raw_df.mz <= voxel_mz_upper) & (raw_df.scan >= voxel_scan_lower) & (raw_df.scan <= voxel_scan_upper) & (raw_df.retention_time_secs >= voxel_rt_lower) & (raw_df.retention_time_secs <= voxel_rt_upper)]

            # find the frame ID of the voxel's highpoint
            voxel_rt_df = voxel_df.groupby(['frame_id','retention_time_secs'], as_index=False).intensity.sum()
            voxel_rt_highpoint = voxel_rt_df.loc[voxel_rt_df.intensity.idxmax()].retention_time_secs
            voxel_rt_highpoint_frame_id = voxel_rt_df.loc[voxel_rt_df.intensity.idxmax()].frame_id

            # keep information aside for debugging
            visualisation_d = {}
            visualisation_d['voxel'] = {'voxel_mz_lower':voxel_mz_lower, 'voxel_mz_upper':voxel_mz_upper, 'voxel_scan_lower':voxel_scan_lower, 'voxel_scan_upper':voxel_scan_upper, 'voxel_rt_lower':voxel_rt_lower, 'voxel_rt_upper':voxel_rt_upper, 'voxel_rt_highpoint':voxel_rt_highpoint, 'voxel_rt_highpoint_frame_id':voxel_rt_highpoint_frame_id}

            # define the search area in the m/z and scan dimensions
            region_mz_lower = voxel_mz_midpoint - ANCHOR_POINT_MZ_LOWER_OFFSET
            region_mz_upper = voxel_mz_midpoint + ANCHOR_POINT_MZ_UPPER_OFFSET
            region_scan_lower = voxel_scan_midpoint - ANCHOR_POINT_SCAN_LOWER_OFFSET
            region_scan_upper = voxel_scan_midpoint + ANCHOR_POINT_SCAN_UPPER_OFFSET
            visualisation_d['region_2d'] = {'region_mz_lower':region_mz_lower, 'region_mz_upper':region_mz_upper, 'region_scan_lower':region_scan_lower, 'region_scan_upper':region_scan_upper}

            # constrain the raw points to the search area for this voxel
            region_2d_df = raw_df[(raw_df.frame_id == voxel_rt_highpoint_frame_id) & (raw_df.mz >= region_mz_lower) & (raw_df.mz <= region_mz_upper) & (raw_df.scan >= region_scan_lower) & (raw_df.scan <= region_scan_upper)].copy()

            # segment the raw data to reveal the isotopes in the feature
            X = region_2d_df[['mz','scan']].values
            dbscan = DBSCAN(eps=1, min_samples=3, metric=point_metric)
            clusters = dbscan.fit_predict(X)

            # assign each point to a cluster (will be -1 it doesn't belong to a cluster)
            region_2d_df['cluster'] = clusters

            # find the centroid of each point cluster
            isotope_centroids_l = []
            for group_name,group_df in region_2d_df.groupby('cluster'):
                if group_name >= 0:
                    mz_centroid = peakutils.centroid(group_df.mz, group_df.intensity)
                    scan_centroid = peakutils.centroid(group_df.scan, group_df.intensity)
                    isotope_centroids_l.append((group_name, mz_centroid, scan_centroid))
            isotope_centroids_df = pd.DataFrame(isotope_centroids_l, columns=['cluster','mz','scan'])

            # if there's more than one isotope...
            if len(isotope_centroids_df) > 0:
                # cluster the centroids into isotopic series
                X = isotope_centroids_df[['mz','scan']].values
                dbscan = DBSCAN(eps=1, min_samples=2, metric=isotope_metric)  # minimum isotopes to form a series
                clusters = dbscan.fit_predict(X)

                # assign each isotope to a cluster, which is a series of isotopes (will be -1 if it doesn't belong to a cluster)
                isotope_centroids_df['isotope_cluster'] = clusters

                # assign each point in the 2D region to an isotope series
                region_2d_df = pd.merge(region_2d_df, isotope_centroids_df[['cluster','isotope_cluster']], how='left', left_on=['cluster'], right_on=['cluster'])
                region_2d_df.replace(to_replace=np.nan, value=-1, inplace=True)
                region_2d_df.isotope_cluster = region_2d_df.isotope_cluster.astype(int)

                visualisation_d['region_2d_df'] = region_2d_df.to_dict('records')

                # only consider the points that are part of an isotope series
                region_2d_df = region_2d_df[(region_2d_df.isotope_cluster >= 0)]

                # each isotopic series is a candidate precursor cuboid for feature detection
                number_of_isotope_series_from_voxel = 0
                for group_name,group_df in region_2d_df.groupby('isotope_cluster'):
                    # get the extent of the isotope cluster in m/z and mobility
                    cuboid_mz_lower = group_df.mz.min()
                    cuboid_mz_upper = group_df.mz.max()
                    cuboid_scan_lower = group_df.scan.min()
                    cuboid_scan_upper = group_df.scan.max()

                    # only handle isotope series that contain the voxel; other voxels will pick up the other features
                    if (voxel_mz_midpoint >= cuboid_mz_lower) and (voxel_mz_midpoint <= cuboid_mz_upper) and (voxel_scan_midpoint >= cuboid_scan_lower) and (voxel_scan_midpoint <= cuboid_scan_upper):
                        region_rt_lower = voxel_rt_highpoint - RT_BASE_PEAK_WIDTH
                        region_rt_upper = voxel_rt_highpoint + RT_BASE_PEAK_WIDTH

                        # get the points
                        region_3d_df = raw_df[(raw_df.mz >= cuboid_mz_lower) & (raw_df.mz <= cuboid_mz_upper) & (raw_df.scan >= cuboid_scan_lower) & (raw_df.scan <= cuboid_scan_upper) & (raw_df.retention_time_secs >= region_rt_lower) & (raw_df.retention_time_secs <= region_rt_upper)].copy()

                        # find the extent of the peak in the RT dimension
                        rt_df = region_3d_df.groupby(['retention_time_secs'], as_index=False).intensity.sum()
                        rt_df.sort_values(by=['retention_time_secs'], ascending=True, inplace=True)

                        # filter the points
                        rt_df['filtered_intensity'] = rt_df.intensity  # set the default
                        try:
                            rt_df['filtered_intensity'] = signal.savgol_filter(rt_df.intensity, window_length=find_filter_length(number_of_points=len(rt_df)), polyorder=RT_FILTER_POLY_ORDER)
                        except:
                            pass

                        # find the valleys nearest the highpoint
                        valley_idxs = peakutils.indexes(-rt_df.filtered_intensity.values, thres=VALLEYS_THRESHOLD_RT, min_dist=VALLEYS_MIN_DIST_RT, thres_abs=False)
                        valley_x_l = rt_df.iloc[valley_idxs].retention_time_secs.to_list()
                        valleys_df = rt_df[rt_df.retention_time_secs.isin(valley_x_l)]

                        upper_x = valleys_df[valleys_df.retention_time_secs > voxel_rt_highpoint].retention_time_secs.min()
                        if math.isnan(upper_x):
                            upper_x = rt_df.retention_time_secs.max()
                        lower_x = valleys_df[valleys_df.retention_time_secs < voxel_rt_highpoint].retention_time_secs.max()
                        if math.isnan(lower_x):
                            lower_x = rt_df.retention_time_secs.min()

                        # the extent of the precursor cuboid in RT
                        cuboid_rt_lower = lower_x
                        cuboid_rt_upper = upper_x

                        visualisation_d['rt_df'] = rt_df.to_dict('records')

                        # add this cuboid to the list
                        precursor_coordinates_d = {
                            'mz_lower':cuboid_mz_lower, 
                            'mz_upper':cuboid_mz_upper, 
                            'wide_mz_lower':cuboid_mz_lower - (CARBON_MASS_DIFFERENCE / 1), # just in case we missed the monoisotopic
                            'wide_mz_upper':cuboid_mz_upper, 
                            'scan_lower':int(cuboid_scan_lower),
                            'scan_upper':int(cuboid_scan_upper), 
                            'wide_scan_lower':int(cuboid_scan_lower), # same because we've already resolved its extent
                            'wide_scan_upper':int(cuboid_scan_upper), 
                            'rt_lower':cuboid_rt_lower, 
                            'rt_upper':cuboid_rt_upper, 
                            'wide_rt_lower':cuboid_rt_lower, # same because we've already resolved its extent
                            'wide_rt_upper':cuboid_rt_upper
                            }

                        number_of_isotope_series_from_voxel += 1

                        if args.visualise:
                            precursor_coordinates_d['visualisation_d'] = visualisation_d
                            
                        precursor_cuboids_l.append(precursor_coordinates_d)
                        print('+', end='', flush=True)

                # if we couldn't form an isotope series around this voxel, it's time to stop
                if number_of_isotope_series_from_voxel == 0:
                    print('.', end='', flush=True)
                    break
            else:
                # if we couldn't form an isotope around this voxel, it's time to stop
                print('-', end='', flush=True)
                break

    # return what we found in this segment
    print('\nfound {} cuboids for mz={} to {}'.format(len(precursor_cuboids_l), segment_mz_lower, segment_mz_upper))
    return precursor_cuboids_l



# move these constants to the INI file
ANCHOR_POINT_MZ_LOWER_OFFSET = 0.6   # one isotope for charge-2 plus a little bit more
ANCHOR_POINT_MZ_UPPER_OFFSET = 3.0   # six isotopes for charge-2 plus a little bit more

ANCHOR_POINT_SCAN_LOWER_OFFSET = 40  # twice the base peak width
ANCHOR_POINT_SCAN_UPPER_OFFSET = 40

# filter and peak detection parameters
VALLEYS_THRESHOLD_RT = 0.5    # only consider valleys that drop more than this proportion of the normalised maximum
VALLEYS_THRESHOLD_SCAN = 0.5

VALLEYS_MIN_DIST_RT = 2.0     # seconds
VALLEYS_MIN_DIST_SCAN = 10.0  # scans

SCAN_FILTER_POLY_ORDER = 3
RT_FILTER_POLY_ORDER = 3

# bin sizes
RT_BIN_SIZE = 5
SCAN_BIN_SIZE = 20
MZ_BIN_SIZE = 0.1


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
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-rm','--ray_mode', type=str, choices=['local','cluster'], help='The Ray mode to use.', required=True)
parser.add_argument('-pc','--proportion_of_cores_to_use', type=float, default=0.9, help='Proportion of the machine\'s cores to use for this program.', required=False)
parser.add_argument('-v','--visualise', action='store_true', help='Generate data for visualisation of the segmentation.')
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
CARBON_MASS_DIFFERENCE = cfg.getfloat('common','CARBON_MASS_DIFFERENCE')

# set up the indexes
print('setting up indexes on {}'.format(CONVERTED_DATABASE_NAME))
create_indexes(db_file_name=CONVERTED_DATABASE_NAME)

# set up the precursor cuboids
CUBOIDS_DIR = '{}/precursor-cuboids-3did'.format(EXPERIMENT_DIR)
if not os.path.exists(CUBOIDS_DIR):
    os.makedirs(CUBOIDS_DIR)

CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-3did.pkl'.format(CUBOIDS_DIR, args.experiment_name, args.run_name)

# set up Ray
# print("setting up Ray")
# if not ray.is_initialized():
#     if args.ray_mode == "cluster":
#         ray.init(num_cpus=number_of_workers())
#     else:
#         ray.init(local_mode=True)

# calculate the segments
mz_range = args.mz_upper - args.mz_lower
NUMBER_OF_MZ_SEGMENTS = (mz_range // args.mz_width_per_segment) + (mz_range % args.mz_width_per_segment > 0)  # thanks to https://stackoverflow.com/a/23590097/1184799

# find the precursors
print('finding precursor cuboids')
# cuboids_l = ray.get([find_precursor_cuboids.remote(segment_mz_lower=args.mz_lower+(i*args.mz_width_per_segment), segment_mz_upper=args.mz_lower+(i*args.mz_width_per_segment)+args.mz_width_per_segment) for i in range(NUMBER_OF_MZ_SEGMENTS)])
cuboids_l = [find_precursor_cuboids(segment_mz_lower=args.mz_lower+(i*args.mz_width_per_segment), segment_mz_upper=args.mz_lower+(i*args.mz_width_per_segment)+args.mz_width_per_segment) for i in range(NUMBER_OF_MZ_SEGMENTS)]
cuboids_l = [item for sublist in cuboids_l for item in sublist]  # cuboids_l is a list of lists, so we need to flatten it

# assign each cuboid a unique identifier
coords_df = pd.DataFrame(cuboids_l)
coords_df['precursor_cuboid_id'] = coords_df.index

# ... and save them in a file
print()
print('saving {} precursor cuboids to {}'.format(len(coords_df), CUBOIDS_FILE))
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
content_d = {'coords_df':coords_df, 'metadata':info}
with open(CUBOIDS_FILE, 'wb') as handle:
    pickle.dump(content_d, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
