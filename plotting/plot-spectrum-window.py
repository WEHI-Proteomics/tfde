import sqlite3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import signal

# cluster array indices
CLUSTER_FRAME_ID_IDX = 0
CLUSTER_ID_IDX = 1
CLUSTER_CHARGE_STATE_IDX = 2
CLUSTER_BASE_SCAN_STD_DEV_IDX = 3
CLUSTER_BASE_MAX_POINT_MZ_IDX = 4
CLUSTER_BASE_MAX_POINT_SCAN_IDX = 5
CLUSTER_INTENSITY_SUM_IDX = 6
CLUSTER_FEATURE_ID_IDX = 7

parser = argparse.ArgumentParser(description='A method for tracking features through frames.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ml','--mz_lower', type=float, help='The low end of the m/z range.', required=False)
parser.add_argument('-mu','--mz_upper', type=float, help='The high end of the m/z range.', required=False)
parser.add_argument('-fl','--frame_lower', type=int, help='The low end of the frame range.', required=False)
parser.add_argument('-fu','--frame_upper', type=int, help='The high end of the frame range.', required=False)
parser.add_argument('-sl','--scan_lower', type=int, help='The low end of the scan range.', required=False)
parser.add_argument('-su','--scan_upper', type=int, help='The high end of the scan range.', required=False)
args = parser.parse_args()

# Connect to the database file
source_conn = sqlite3.connect(args.database_name)
c = source_conn.cursor()

if args.frame_lower is None:
    q = c.execute("SELECT value FROM summing_info WHERE item=\"frame_lower\"")
    row = q.fetchone()
    args.frame_lower = int(row[0])
    print("lower frame_id set to {} from the data".format(args.frame_lower))

if args.frame_upper is None:
    q = c.execute("SELECT value FROM summing_info WHERE item=\"frame_upper\"")
    row = q.fetchone()
    args.frame_upper = int(row[0])
    print("upper frame_id set to {} from the data".format(args.frame_upper))

if args.mz_lower is None:
    q = c.execute("SELECT value FROM summing_info WHERE item=\"mz_lower\"")
    row = q.fetchone()
    args.mz_lower = float(row[0])
    print("lower m/z set to {} from the data".format(args.mz_lower))

if args.mz_upper is None:
    q = c.execute("SELECT value FROM summing_info WHERE item=\"mz_upper\"")
    row = q.fetchone()
    args.mz_upper = float(row[0])
    print("upper m/z set to {} from the data".format(args.mz_upper))

if args.scan_lower is None:
    q = c.execute("SELECT value FROM summing_info WHERE item=\"scan_lower\"")
    row = q.fetchone()
    args.scan_lower = int(row[0])
    print("lower scan set to {} from the data".format(args.scan_lower))

if args.scan_upper is None:
    q = c.execute("SELECT value FROM summing_info WHERE item=\"scan_upper\"")
    row = q.fetchone()
    args.scan_upper = int(row[0])
    print("upper scan set to {} from the data".format(args.scan_upper))

c.execute("select frame_id, cluster_id, charge_state, base_peak_scan_std_dev, base_peak_max_point_mz, base_peak_max_point_scan, intensity_sum, feature_id from clusters where frame_id>={} and frame_id<={} and base_peak_max_point_mz>={} and base_peak_max_point_mz<={} and base_peak_max_point_scan>={} and base_peak_max_point_scan<={} order by frame_id, cluster_id asc;".format(args.frame_lower, args.frame_upper, args.mz_lower, args.mz_upper, args.scan_lower, args.scan_upper))

clusters_v = np.array(c.fetchall(), dtype=np.float32)
clusters_v_feature_indices = np.where(clusters_v[:,CLUSTER_FEATURE_ID_IDX] > 0)[0]
clusters_v_not_feature_indices = np.where(clusters_v[:,CLUSTER_FEATURE_ID_IDX] == 0)[0]
source_conn.close()

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(clusters_v[clusters_v_not_feature_indices,CLUSTER_FRAME_ID_IDX], clusters_v[clusters_v_not_feature_indices,CLUSTER_INTENSITY_SUM_IDX], 'o', markerfacecolor='green', markeredgewidth=0.0, markersize=6)
ax1.plot(clusters_v[clusters_v_feature_indices,CLUSTER_FRAME_ID_IDX], clusters_v[clusters_v_feature_indices,CLUSTER_INTENSITY_SUM_IDX], 'o', markerfacecolor='orange', markeredgewidth=0.0, markersize=6)

plt.title("Detected Features, frames {}-{}, m/z {}-{}, scan {}-{} ({})".format(args.frame_lower, args.frame_upper, args.mz_lower, args.mz_upper, args.scan_lower, args.scan_upper, args.database_name))

plt.xlabel('frame')
plt.ylabel('intensity')
plt.yscale('log')

f2 = plt.figure()
plt.gca().invert_yaxis()
ax2 = f2.add_subplot(111)
ax2.plot(clusters_v[clusters_v_not_feature_indices,CLUSTER_BASE_MAX_POINT_MZ_IDX], clusters_v[clusters_v_not_feature_indices,CLUSTER_BASE_MAX_POINT_SCAN_IDX], 'o', markerfacecolor='green', markeredgewidth=0.0, markersize=6)
ax2.plot(clusters_v[clusters_v_feature_indices,CLUSTER_BASE_MAX_POINT_MZ_IDX], clusters_v[clusters_v_feature_indices,CLUSTER_BASE_MAX_POINT_SCAN_IDX], 'o', markerfacecolor='orange', markeredgewidth=0.0, markersize=6)

plt.xlabel('m/z')
plt.ylabel('scan')

plt.margins(0.02)
plt.show()
