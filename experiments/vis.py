import sys
import sqlite3
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Visualise ')
parser.add_argument('-f','--frame', type=int, help='The frame number.', required=True)
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
parser.add_argument('-q','--query_only', help='Just find out the attributes of the frames in the database.', required=False, action='store_true')
scan_group1 = parser.add_mutually_exclusive_group(required=False)
scan_group1.add_argument('-slp','--scan_lower_proportional', type=float, help='Scan lower value (proportional). E.g. 0.5 means the scan half-way through.')
scan_group1.add_argument('-sla','--scan_lower_absolute', type=int, help='The (absolute) lower scan number.')
scan_group2 = parser.add_mutually_exclusive_group(required=False)
scan_group2.add_argument('-sup','--scan_upper_proportional', type=float, help='Scan upper value (proportional).')
scan_group2.add_argument('-sua','--scan_upper_absolute', type=int, help='The (absolute) upper scan number.')
mz_group1 = parser.add_mutually_exclusive_group(required=False)
mz_group1.add_argument('-mlp','--mz_lower_proportional', type=float, help='m/z lower value (proportional). E.g. 0.5 means the m/z half-way through.')
mz_group1.add_argument('-mla','--mz_lower_absolute', type=float, help='The (absolute) lower m/z.')
mz_group2 = parser.add_mutually_exclusive_group(required=False)
mz_group2.add_argument('-mup','--mz_upper_proportional', type=float, help='m/z upper value (proportional).')
mz_group2.add_argument('-mua','--mz_upper_absolute', type=float, help='The (absolute) upper m/z number.')

args = parser.parse_args()

print args

source_conn = sqlite3.connect(args.database_name)
frame_df = pd.read_sql_query("select scan from frames where frame_id={};".format(args.frame), source_conn)
scan_min = frame_df.scan.min()
scan_max = frame_df.scan.max()
frame_df = pd.read_sql_query("select mz from frames where frame_id={};".format(args.frame), source_conn)
mz_min = frame_df.mz.min()
mz_max = frame_df.mz.max()

if args.query_only:
    frame_df = pd.read_sql_query("select distinct frame_id from frames;".format(args.frame), source_conn)
    frame_min = frame_df.frame_id.min()
    frame_max = frame_df.frame_id.max()
    print("frames {} to {}, scans from {} to {}, m/z from {} to {}".format(frame_min, frame_max, scan_min, scan_max, mz_min, mz_max))

if args.scan_lower_proportional != None:
    scan_lower = int(args.scan_lower_proportional * (scan_max-scan_min) + scan_min)
elif args.scan_lower_absolute != None:
    scan_lower = args.scan_lower_absolute
else:
    scan_lower = scan_min
if args.scan_upper_proportional != None:
    scan_upper = int(args.scan_upper_proportional * (scan_max-scan_min) + scan_min)
elif args.scan_upper_absolute != None:
    scan_upper = args.scan_upper_absolute
else:
    scan_upper = scan_max

if args.mz_lower_proportional != None:
    mz_lower = int(args.mz_lower_proportional * (mz_max-mz_min) + mz_min)
elif args.mz_lower_absolute != None:
    mz_lower = args.mz_lower_absolute
else:
    mz_lower = mz_min
if args.mz_upper_proportional != None:
    mz_upper = int(args.mz_upper_proportional * (mz_max-mz_min) + mz_min)
elif args.mz_upper_absolute != None:
    mz_upper = args.mz_upper_absolute
else:
    mz_upper = mz_max

print("scan from {} to {}".format(scan_lower, scan_upper))
print("mz from {} to {}".format(mz_lower, mz_upper))

frame_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id={} AND scan>={} and scan<={} and mz>={} and mz<={};".format(args.frame, scan_lower, scan_upper, mz_lower, mz_upper), source_conn)

# plot along the m/z axis
f = plt.figure()
ax1 = f.add_subplot(111)
plt.title("Database {}, frame {}, scan {} to {}, m/z {} to {}".format(args.database_name, args.frame, scan_lower, scan_upper, mz_lower, mz_upper))
plt.xlabel('m/z')
plt.ylabel('intensity')
plt.margins(0.02)
# plt.xlim(1200, 1300)
# plt.ylim(0, 675)
for scan in range(scan_lower, scan_upper+1):
    points_df = frame_df[(frame_df.scan == scan)].sort_values('mz', ascending=True)
    ax1.plot(points_df.mz, points_df.intensity, 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)
plt.show()
