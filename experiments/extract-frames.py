import sys
import numpy as np
import pandas as pd
import sqlite3
import csv

FRAME_START = 137000
FRAME_END = 137004
SCAN = 50
SQLITE_FILE = "\\temp\\frames-20ms-th-0-137000-138000-V4.sqlite"
CSV_FILE = "\\temp\\frames-single-scan.csv"

conn = sqlite3.connect(SQLITE_FILE)
frame_df = pd.read_sql_query("select mz,intensity,scan from frames where frame_id>={} AND frame_id<={} ORDER BY MZ ASC;".format(FRAME_START, FRAME_END), conn)
conn.close()

scan_min = frame_df.scan.min()
scan_max = frame_df.scan.max()
print("scans from {} to {}".format(scan_min, scan_max))
points_df = frame_df[frame_df.scan == SCAN].sort('mz', ascending=True)

with open(CSV_FILE, 'wb') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(["mz", "intensity"])
	for index, point in frame_df.iterrows():
		writer.writerow([point.mz, point.intensity])
