import sys
import numpy as np
import pandas as pd
import sqlite3
import time
import peakutils
import matplotlib.pyplot as plt


FRAME_START = 700
FRAME_END = 720
FRAMES_TO_SUM = 5
INSTRUMENT_RESOLUTION = 40000.0
MZ_INCREMENT = 0.00001
SOURCE_SQLITE_FILE = "\\temp\\161117-TIMS_TuneMix_MS_AutoMSMS-frames-th-0-1-1565-V4.sqlite"
DEST_SQLITE_FILE = "\\temp\\summed-161117-TIMS_TuneMix_MS_AutoMSMS-frames-th-0-1-1565-V4.sqlite"

# Formula from https://en.wikipedia.org/wiki/Gaussian_function
def gaussian(x, amplitude, peak, stddev):
    num = np.power((x-peak), 2.)
    den = 2. * np.power(stddev, 2.)
    return amplitude * np.exp(-num/den)

def find_mz_index(min_mz, mz):
	idx = int((mz-min_mz)/MZ_INCREMENT)
	return idx

source_conn = sqlite3.connect(SOURCE_SQLITE_FILE)

dest_conn = sqlite3.connect(DEST_SQLITE_FILE)
dest_c = dest_conn.cursor()

dest_c.execute('''DROP TABLE IF EXISTS frames''')
dest_c.execute('''CREATE TABLE frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)''')
dest_c.execute('''DROP INDEX IF EXISTS idx_frames''')
dest_c.execute('''CREATE INDEX idx_frames ON frames (frame_id)''')

frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id>={} AND frame_id<{} ORDER BY FRAME_ID, MZ, SCAN ASC;".format(FRAME_START, FRAME_END), source_conn)
mz_min = frame_df.mz.min()
mz_max = frame_df.mz.max()
scan_min = frame_df.scan.min()
scan_max = frame_df.scan.max()
print("{} to {} m/z, {} to {} scan".format(mz_min, mz_max, scan_min, scan_max))

vector_length = int((mz_max-mz_min)/MZ_INCREMENT)
sum_vector = np.zeros(vector_length, dtype=np.float)
eval_vector = np.zeros(vector_length, dtype=np.float)
mz_vector = np.linspace(mz_min, mz_max, num=vector_length)

summedFrameId = 1

for base_frame in range(FRAME_START, FRAME_END, FRAMES_TO_SUM):
    print("base frame: {}".format(base_frame))
    # for scan in range(scan_min, scan_max+1):
    pointId = 0
    for scan in range(scan_min, scan_max+1):
        points = []
        print("  scan: {} of {}".format(scan, scan_max))
        for frame in range(base_frame, base_frame+FRAMES_TO_SUM):
            points_df = frame_df[(frame_df.scan == scan) & (frame_df.frame_id == frame)].sort_values('mz', ascending=True)
            print("    frame: {} ({} points on this scan)".format(frame, len(points_df)))
            for point_index in range(0,len(points_df)):
                point = points_df.iloc[point_index]
                stddev = (point.mz / INSTRUMENT_RESOLUTION) / 2.35482
                lower_index = find_mz_index(mz_min, point.mz-(4*stddev))
                upper_index = find_mz_index(mz_min, point.mz+(4*stddev))
                for eval_index in range(lower_index, upper_index+1):
                    eval_vector[eval_index] = gaussian(mz_vector[eval_index], point.intensity, point.mz, stddev)
                sum_vector += eval_vector
                eval_vector.fill(0.0)

        # sum_vector now contains the summed gaussians for the set of 5 frames for this scan line

        # Find the maxima for this scan
        signal_indices = np.where(sum_vector > 1)[0]
        truncated_sum_vector = sum_vector[signal_indices]
        if len(truncated_sum_vector) > 0:
            print("    {} points in the truncated sum vector".format(len(truncated_sum_vector)))
            indexes = peakutils.indexes(truncated_sum_vector, thres=0.01, min_dist=1000)

            # Write out the maxima as points for this scan
            for i in range(0,len(indexes)):
                points.append((summedFrameId, pointId, mz_vector[signal_indices[indexes[i]]], scan, int(sum_vector[signal_indices[indexes[i]]]), 0))
                pointId += 1

            print("    Writing scan {} of frame {} ({} points) to the database.".format(scan, summedFrameId, len(points)))
            dest_c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", points)
            dest_conn.commit()

        # f = plt.figure()
        # ax1 = f.add_subplot(111)
        # plt.xlim(1050, 1150)
        # ax1.plot(mz_vector, sum_vector, 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)
        # ax1.plot(mz_vector[signal_indices[indexes]], sum_vector[signal_indices[indexes]], 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        # plt.show()

        # reset the sum vector for the next scan
        sum_vector.fill(0.0)

    summedFrameId += 1

# Close the connection
source_conn.close()
dest_conn.close()
