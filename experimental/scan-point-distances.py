import sys
import sqlite3
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Display a histogram of distances between points along a scan.')
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
parser.add_argument('-f','--frame', type=int, help='The frame number.', required=True)
parser.add_argument('-s','--scan', type=int, help='The scan number.', required=True)
parser.add_argument('-q','--query', help='Display the attributes of the frames in the database.', required=False, action='store_true')
args = parser.parse_args()

source_conn = sqlite3.connect(args.database_name)

if args.query:
    frame_df = pd.read_sql_query("select distinct scan from frames where frame_id={};".format(args.frame), source_conn)
    scan_min = frame_df.scan.min()
    scan_max = frame_df.scan.max()
    frame_df = pd.read_sql_query("select mz from frames where frame_id={};".format(args.frame), source_conn)
    mz_min = frame_df.mz.min()
    mz_max = frame_df.mz.max()
    frame_df = pd.read_sql_query("select distinct frame_id from frames;", source_conn)
    frame_min = frame_df.frame_id.min()
    frame_max = frame_df.frame_id.max()
    print("frames {}-{}, scans {}-{}, m/z {}-{}".format(frame_min, frame_max, scan_min, scan_max, mz_min, mz_max))

points_df = pd.read_sql_query("select mz from frames where frame_id={} AND scan={};".format(args.frame, args.scan), source_conn)
print("{} points on scan {}".format(len(points_df.index), args.scan))

distance = np.zeros((len(points_df.index), len(points_df.index)))
for i,from_row in points_df.iterrows():
    for j,to_row in points_df.iterrows():
        if j != i:
            distance[i, j] = np.abs(from_row.mz - to_row.mz)

min_distance = np.min(distance[np.where(distance > 0.0)])
d = distance.reshape(-1)
d_zoom = d[np.where((d > 0) & (d < 0.036))]

hist, bins = np.histogram(d_zoom, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

plt.bar(center, hist, align='center', width=width)
plt.suptitle("Distances between points along a scan")
plt.title("database {}, frame {}, scan {}, points={}, minimum distance={}".format(args.database_name, args.frame, args.scan, len(points_df.index), min_distance))
plt.xlabel('m/z distance')
plt.ylabel('count')
plt.xticks(bins, rotation=70)
plt.margins(0.02)
plt.show()
