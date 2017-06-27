import sys
import numpy as np
import pandas as pd
import sqlite3
import time


FRAME_START = 137000
FRAME_END = 137020
FRAMES_TO_SUM = 5
INSTRUMENT_RESOLUTION = 40000.0
MZ_INCREMENT = 0.0001
SOURCE_SQLITE_FILE = "\\temp\\frames-20ms-th-0-137000-138000-V4.sqlite"

# Formula from https://en.wikipedia.org/wiki/Gaussian_function
def gaussian(x, amplitude, peak, stddev):
    num = np.power((x-peak), 2.)
    den = 2. * np.power(stddev, 2.)
    return amplitude * np.exp(-num/den)

def getnearpos(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

source_conn = sqlite3.connect(SOURCE_SQLITE_FILE)

frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id>={} AND frame_id<{} ORDER BY FRAME_ID, MZ, SCAN ASC;".format(FRAME_START, FRAME_END), source_conn)
mz_min = frame_df.mz.min()
mz_max = frame_df.mz.max()
scan_min = frame_df.scan.min()
scan_max = frame_df.scan.max()

vector_length = int((mz_max-mz_min)/MZ_INCREMENT)
sum_vector = np.zeros(vector_length, dtype=np.int16)
eval_vector = np.zeros(vector_length, dtype=np.int16)
mz_vector = np.linspace(mz_min, mz_max, num=vector_length)

for base_frame in range(FRAME_START, FRAME_END, FRAMES_TO_SUM):
	print("base frame: {}".format(base_frame))
	frame_set_start = time.time()
	for scan in range(scan_min, scan_max+1):
		print("  scan: {} of {}".format(scan, scan_max))
		scan_start = time.time()
		for frame in range(base_frame, base_frame+FRAMES_TO_SUM):
			print("    frame: {}".format(frame))
			points_df = frame_df[(frame_df.scan == scan) & (frame_df.frame_id == frame)].sort_values('mz', ascending=True)
			for point_index in range(0,len(points_df)):
				point = points_df.iloc[point_index]
				stddev = (point.mz / INSTRUMENT_RESOLUTION) / 2.35482
				lower_index = getnearpos(mz_vector, point.mz-(3*stddev))
				upper_index = getnearpos(mz_vector, point.mz+(3*stddev))
				for eval_index in range(lower_index, upper_index):
				    eval_vector[eval_index] = int(gaussian(mz_vector[eval_index], point.intensity, point.mz, stddev))
				sum_vector += eval_vector
		scan_end = time.time()
		print("  summed scan in {} sec".format(scan_end-scan_start))
	frame_set_end = time.time()
	print("summed {} frames in {} sec".format(FRAMES_TO_SUM, frame_set_end-frame_set_start))
