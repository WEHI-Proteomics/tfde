import sqlite3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import signal
import sys

# cluster array indices
CLUSTER_FRAME_ID_IDX = 0
CLUSTER_ID_IDX = 1
CLUSTER_CHARGE_STATE_IDX = 2
CLUSTER_BASE_SCAN_STD_DEV_IDX = 3
CLUSTER_BASE_MAX_POINT_MZ_IDX = 4
CLUSTER_BASE_MAX_POINT_SCAN_IDX = 5
CLUSTER_INTENSITY_SUM_IDX = 6
CLUSTER_FEATURE_ID_IDX = 7

def find_nearest_low_index_below_threshold(values, threshold):
    max_index = np.argmax(values)
    values_below_threshold = (values - threshold) < 0
    change_indices = np.where(np.roll(values_below_threshold,1) != values_below_threshold)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
    if len(change_indices) > 0:
        distance_from_base = change_indices - max_index
        distance_from_base[distance_from_base>0] = -sys.maxint
        idx = change_indices[distance_from_base.argmax()]
    else:
        idx = None
    return idx

def find_nearest_high_index_below_threshold(values, threshold):
    max_index = np.argmax(values)
    values_below_threshold = (values - threshold) < 0
    change_indices = np.where(np.roll(values_below_threshold,1) != values_below_threshold)[0]     # for when there is more than one cluster found in a frame, the first cluster will be the most intense
    if len(change_indices) > 0:
        distance_from_base = change_indices - max_index
        distance_from_base[distance_from_base<0] = sys.maxint
        idx = change_indices[distance_from_base.argmin()]
    else:
        idx = None
    return idx

parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fi','--feature_id', type=int, help='The ID of the feature.', required=True)
args = parser.parse_args()

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)
c = source_conn.cursor()
# c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum, feature_id from clusters where base_peak_max_point_mz>=560 and base_peak_max_point_mz<=564 order by frame_id, cluster_id asc;")
c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum, feature_id from clusters where feature_id={} order by frame_id, cluster_id asc;".format(args.feature_id))

clusters_v = np.array(c.fetchall(), dtype=np.float32)
clusters_v_feature_indices = np.where(clusters_v[:,CLUSTER_FEATURE_ID_IDX] > 0)[0]
source_conn.close()

filtered = signal.savgol_filter(clusters_v[clusters_v_feature_indices, CLUSTER_INTENSITY_SUM_IDX], window_length=41, polyorder=10)
low_index = find_nearest_low_index_below_threshold(filtered, 50000)
high_index = find_nearest_high_index_below_threshold(filtered, 50000)

f1 = plt.figure()
ax1 = f1.add_subplot(111)
# ax1.plot(clusters_v[:,CLUSTER_FRAME_ID_IDX], clusters_v[:,CLUSTER_INTENSITY_SUM_IDX], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
ax1.plot(clusters_v[clusters_v_feature_indices,CLUSTER_FRAME_ID_IDX], clusters_v[clusters_v_feature_indices,CLUSTER_INTENSITY_SUM_IDX], 'o', markeredgewidth=0.0, markersize=6)
ax1.plot(clusters_v[clusters_v_feature_indices, CLUSTER_FRAME_ID_IDX], filtered, '-', markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
ax1.plot(clusters_v[clusters_v_feature_indices[low_index], CLUSTER_FRAME_ID_IDX], filtered[low_index], 'x', markerfacecolor='green', markeredgecolor='green', markeredgewidth=2.0, markersize=20, alpha=0.5)
ax1.plot(clusters_v[clusters_v_feature_indices[high_index], CLUSTER_FRAME_ID_IDX], filtered[high_index], 'x', markerfacecolor='green', markeredgecolor='green', markeredgewidth=2.0, markersize=20, alpha=0.5)

plt.title("Features")

plt.xlabel('frame')
plt.ylabel('intensity')
# plt.yscale('log')

f2 = plt.figure()
plt.gca().invert_yaxis()
ax2 = f2.add_subplot(111)
ax2.plot(clusters_v[:,CLUSTER_BASE_MAX_POINT_MZ_IDX], clusters_v[:,CLUSTER_BASE_MAX_POINT_SCAN_IDX], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
ax2.plot(clusters_v[clusters_v_feature_indices,CLUSTER_BASE_MAX_POINT_MZ_IDX], clusters_v[clusters_v_feature_indices,CLUSTER_BASE_MAX_POINT_SCAN_IDX], 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=6)

plt.xlabel('m/z')
plt.ylabel('scan')

plt.margins(0.02)
plt.show()
