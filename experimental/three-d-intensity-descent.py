import pandas as pd
import numpy as np
from matplotlib import colors, cm, text, pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import peakutils
from scipy import signal
import math
import os
import time

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
    if (abs(mz_1 - mz_2) <= 0.1) and (abs(scan_1 - scan_2) <= 2):
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
    if (abs(mz_1 - mz_2) <= 1.0) and (abs(mz_1 - mz_2) > 0.1) and (abs(scan_1 - scan_2) <= 5):
        result = 0.5
    else:
        result = 10
    # print('r1={}, r2={}, result={}'.format(r1,r2,result))
    return result


MZ_MIN = 748
MZ_MAX = 766

MS1_PEAK_DELTA = 0.1
RT_BASE_PEAK_WIDTH = 10

MIN_ANCHOR_POINT_INTENSITY = 500

# constrain the data to re-run the same feature for debugging
MZ_MIN_DEBUG, MZ_MAX_DEBUG = (764.4201958278368, 765.9385489808168)
SCAN_MIN_DEBUG, SCAN_MAX_DEBUG = (400, 460)
RT_LOWER_DEBUG, RT_UPPER_DEBUG = (2103.4829066671086, 2110.362298691853)

start_run = time.time()

# determine the maximum scan for charge-1 features in this m/z range
charge_one_scan_max = max(scan_coords_for_single_charge_region(mz_lower=MZ_MIN, mz_upper=MZ_MAX))

raw_df = pd.read_pickle('/Users/darylwilding-mcbride/Downloads/YHE211_1-mz-748-766-rt-2000-2200.pkl')
raw_df = raw_df[(raw_df.frame_type == 0) & (raw_df.intensity >= 50) & (raw_df.scan >= charge_one_scan_max)]
# raw_df = raw_df[(raw_df.mz >= MZ_MIN_DEBUG) & (raw_df.mz <= MZ_MAX_DEBUG) & (raw_df.scan >= SCAN_MIN_DEBUG) & (raw_df.scan <= SCAN_MAX_DEBUG) & (raw_df.retention_time_secs >= RT_LOWER_DEBUG) & (raw_df.retention_time_secs <= RT_UPPER_DEBUG)]
raw_df.reset_index(drop=True, inplace=True)

# assign each point a unique identifier
raw_df['point_id'] = raw_df.index

CUBOIDS_FILE = '/Users/darylwilding-mcbride/Downloads/precursor-cuboids.pkl'
precursor_cuboids_l = []
# remove the cuboids file
if os.path.isfile(CUBOIDS_FILE):
    os.remove(CUBOIDS_FILE)
# a unique id for each precursor cuboid
precursor_cuboid_id = 1


anchor_point_s = raw_df.loc[raw_df.intensity.idxmax()]
while anchor_point_s.intensity >= MIN_ANCHOR_POINT_INTENSITY:
    mz_lower = anchor_point_s.mz - 1.0
    mz_upper = anchor_point_s.mz + 7.0
    scan_lower = anchor_point_s.scan - 100
    scan_upper = anchor_point_s.scan + 100

    candidate_region_df = raw_df[(raw_df.frame_id == anchor_point_s.frame_id) & (raw_df.mz >= mz_lower) & (raw_df.mz <= mz_upper) & (raw_df.scan >= scan_lower) & (raw_df.scan <= scan_upper)].copy()

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

    scan_0_extract_df = scan_0_df[(scan_0_df.scan >= lower_x) & (scan_0_df.scan <= upper_x)]

    # trim the candidate region to account for the selected peak in mobility
    candidate_region_df = candidate_region_df[(candidate_region_df.scan >= lower_x) & (candidate_region_df.scan <= upper_x)]

    # segment the raw data to reveal the isotopes in the feature
    X = candidate_region_df[['mz','scan']].values

    # cluster the points
    dbscan = DBSCAN(eps=1, min_samples=3, metric=point_metric)
    clusters = dbscan.fit_predict(X)
    candidate_region_df['cluster'] = clusters

    number_of_point_clusters = len(candidate_region_df[candidate_region_df.cluster >= 0].cluster.unique())

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
    dbscan = DBSCAN(eps=1, min_samples=1, metric=isotope_metric)
    clusters = dbscan.fit_predict(X)
    centroids_df['isotope_cluster'] = clusters

    number_of_isotope_clusters = len(centroids_df[centroids_df.isotope_cluster >= 0].isotope_cluster.unique())

    candidate_region_df = pd.merge(candidate_region_df, centroids_df[['cluster','isotope_cluster']], how='left', left_on=['cluster'], right_on=['cluster'])
    candidate_region_df.fillna(value=-1, inplace=True)
    candidate_region_df.isotope_cluster = candidate_region_df.isotope_cluster.astype(int)

    # we now have the 2D extent of the feature - take that extent through time and see if we can cluster the centroids in time

    # which cluster contains the anchor point?
    anchor_point_s = candidate_region_df[(candidate_region_df.point_id == anchor_point_s.point_id)].iloc[0]

    # get the extent of the isotope cluster in m/z and mobility
    points_in_cluster_df = candidate_region_df[(candidate_region_df.isotope_cluster == anchor_point_s.isotope_cluster)]
    mz_lower = points_in_cluster_df.mz.min()
    mz_upper = points_in_cluster_df.mz.max()
    scan_lower = points_in_cluster_df.scan.min()
    scan_upper = points_in_cluster_df.scan.max()

    # get the left-most peak in the isotope cluster
    monoisotopic_cluster_s = centroids_df.loc[centroids_df[(centroids_df.isotope_cluster == anchor_point_s.isotope_cluster)].mz.idxmin()]
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

    rt_0_extract_df = rt_0_df[(rt_0_df.retention_time_secs >= lower_x) & (rt_0_df.retention_time_secs <= upper_x)]

    # set the intensity to zero for all points in the candidate region points
    raw_df.loc[(raw_df.retention_time_secs >= rt_lower) & (raw_df.retention_time_secs <= rt_upper) & (raw_df.mz >= mz_lower) & (raw_df.mz <= mz_upper) & (raw_df.scan >= scan_lower) & (raw_df.scan <= scan_upper), 'intensity'] = 10

    # add this cuboid to the list
    precursor_cuboids_l.append((precursor_cuboid_id, mz_lower, mz_upper, scan_lower, scan_upper, rt_lower, rt_upper))

    # find the next anchor point
    anchor_point_s = raw_df.loc[raw_df.intensity.idxmax()]

    print('.', end='', flush=True)

# save the precursor cuboids
precursor_cuboids_df = pd.DataFrame(precursor_cuboids_l, columns=['precursor_cuboid_id', 'mz_lower', 'mz_upper', 'scan_lower', 'scan_upper', 'rt_lower', 'rt_upper'])
print()
print('saving {} precursor cuboids to {}'.format(len(precursor_cuboids_df), CUBOIDS_FILE))
precursor_cuboids_df.to_pickle(CUBOIDS_FILE)

stop_run = time.time()
# print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
print("total running time: {} seconds".format(round(stop_run-start_run,1)))
