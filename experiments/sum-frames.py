import sys
import numpy as np
import pandas as pd
import sqlite3
import time
import peakutils
import matplotlib.pyplot as plt

FRAMES_TO_SUM = 5
FRAME_START = 137000
FRAME_END = 137005
INSTRUMENT_RESOLUTION = 40000.0
MZ_INCREMENT = 0.001
MIN_MZ_BETWEEN_POINTS = 0.01
MIN_NUMBER_OF_MZ_POINTS_BETWEEN_PEAKS = max(1, int(round(MIN_MZ_BETWEEN_POINTS / MZ_INCREMENT)))
SOURCE_SQLITE_FILE = "\\temp\\frames-20ms-th-0-137000-138000-V4.sqlite"
DEST_SQLITE_FILE = "\\temp\\t.sqlite"

def gaussian(p):
    x = p[0]
    ampl = p[1]
    center = p[2]
    dev = p[3]
    return ampl * np.exp(-(x - center) ** 2 / (2 * dev ** 2))

# Find the index for the specified m/z value
def find_mz_index(min_mz, mz, vector_length):
    idx = int((mz-min_mz)/MZ_INCREMENT)
    if idx < 0:
        idx = 0
    elif idx > vector_length-1:
        idx = vector_length-1
    return idx

# Find the m/z indexes either side of the peak where the value is 95% of the peak's intensity
def find_peak_region_indexes(mz_vector_length, intensity_vector, peak_mz_index):
    lower_mz_index = 0
    upper_mz_index = mz_vector_length
    region_intensity_value = intensity_vector[peak_mz_index] * 0.95
    for i in range(peak_mz_index, 0, -1):
        if intensity_vector[i] <= region_intensity_value:
            lower_mz_index = i
            break
    for i in range(peak_mz_index, len(mz_vector), 1):
        if intensity_vector[i] <= region_intensity_value:
            upper_mz_index = i
            break
    print("peak at m/z={}: intensity {}, lower {}, m/z range: {} to {}".format(mz_vector[peak_mz_index], intensity_vector[peak_mz_index], region_intensity_value, mz_vector[lower_mz_index], mz_vector[upper_mz_index]))
    return (lower_mz_index, upper_mz_index)

def peak_region(mz_vector, peak_mz_index):
    mz_offset = HWHM(mz_vector[peak_mz_index])
    number_of_points = int(round(mz_offset / MZ_INCREMENT))
    lower_mz_index = max(peak_mz_index-number_of_points, 0)
    upper_mz_index = min(peak_mz_index+number_of_points, len(mz_vector))
    return (lower_mz_index, upper_mz_index)

# Find the index of the array value closest to the specified value
def get_near_pos(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def standard_deviation(mz):
    return (mz / INSTRUMENT_RESOLUTION) / 2.35482

def HWHM(mz):
    return (mz / INSTRUMENT_RESOLUTION) / 2.0

def main():
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
    # scan_min = 134
    # scan_max = 134
    intensity_max = frame_df.intensity.max()
    frame_v = frame_df.values
    print("frames {} to {}, m/z {} to {}, scans {} to {}, intensity max: {}".format(FRAME_START, FRAME_END, mz_min, mz_max, scan_min, scan_max, intensity_max))

    vector_length = int((mz_max-mz_min)/MZ_INCREMENT)
    sum_vector = np.zeros(vector_length, dtype=np.float)
    # eval_vector = np.zeros(vector_length, dtype=np.float)
    mz_vector = np.linspace(mz_min, mz_max, num=vector_length)

    summedFrameId = 1

    for base_frame in range(FRAME_START, FRAME_END, FRAMES_TO_SUM):
        frame_start = time.time()
        print("base frame: {}".format(base_frame))
        # for scan in range(scan_min, scan_max+1):
        pointId = 1
        points = []
        for scan in range(scan_min, scan_max+1):
            centroid_indexes = []
            lower_indexes = []
            upper_indexes = []
            number_of_points = 0
            # print("  scan: {} of {}".format(scan, scan_max))
            scan_sum_start = time.time()
            scan_start = time.time()
            for frame in range(base_frame, base_frame+FRAMES_TO_SUM):
                # points_df = frame_df[(frame_df.scan == scan) & (frame_df.frame_id == frame)]
                points_v = frame_v[np.where((frame_v[:,2] == scan) & (frame_v[:,0] == frame))]
                for point_index in range(0,len(points_v)):
                    point = points_v[point_index]
                    point_mz = point[1]
                    point_intensity = point[3]
                    stddev = standard_deviation(point_mz)
                    lower_index = find_mz_index(mz_min, point_mz-(4*stddev), vector_length)
                    upper_index = find_mz_index(mz_min, point_mz+(4*stddev), vector_length)
                    region = ((mz_vector[eval_index], point_intensity, point_mz, stddev) for eval_index in range(lower_index, upper_index+1))
                    sum_vector[lower_index:upper_index+1] += map(gaussian, region)
                number_of_points += len(points_v)
            scan_sum_end = time.time()
            # print("    summed scan ({} points) in {} sec".format(number_of_points, scan_sum_end-scan_sum_start))
            # sum_vector now contains the summed gaussians for the set of 5 frames for this scan line

            # Find the peak centroids and write them out to the database
            scan_centroids_start = time.time()
            signal_indices = np.where(sum_vector > 1)[0]
            truncated_sum_vector = sum_vector[signal_indices]
            if len(truncated_sum_vector) > 0:
                # Find the maxima for this scan
                indexes = peakutils.indexes(truncated_sum_vector, thres=0.00001, min_dist=MIN_NUMBER_OF_MZ_POINTS_BETWEEN_PEAKS)
                for i in range(0,len(indexes)):
                    # lower_peak_index, upper_peak_index = find_peak_region_indexes(len(mz_vector), sum_vector, signal_indices[indexes[i]])
                    lower_peak_index, upper_peak_index = peak_region(mz_vector, signal_indices[indexes[i]])
                    centroid_mz = peakutils.centroid(mz_vector[lower_peak_index:upper_peak_index+1], sum_vector[lower_peak_index:upper_peak_index+1])
                    centroid_index = find_mz_index(mz_min, centroid_mz, vector_length)
                    # add this point to the list
                    points.append((summedFrameId, pointId, centroid_mz, scan, int(round(sum_vector[centroid_index])), 0))
                    centroid_indexes.append(centroid_index)
                    lower_indexes.append(lower_peak_index)
                    upper_indexes.append(upper_peak_index)
                    pointId += 1
                    # print("centroid m/z: {}, lower m/z: {}, upper m/z: {}".format(centroid_mz, mz_vector[lower_peak_index], mz_vector[upper_peak_index]))
                scan_centroids_end = time.time()
                # print("    centroided scan in {} sec".format(scan_centroids_end-scan_centroids_start))

            # Plot a scan
            # f = plt.figure()
            # ax1 = f.add_subplot(111)
            # plt.xlim(1200, 1300)
            # plt.ylim(0, 675)
            # ax1.plot(mz_vector, sum_vector, 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)
            # ax1.plot(mz_vector[signal_indices[indexes]], sum_vector[signal_indices[indexes]], '+', markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.0, markersize=10)
            # ax1.plot(mz_vector[centroid_indexes], sum_vector[centroid_indexes], 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
            # ax1.plot(mz_vector[lower_indexes], sum_vector[lower_indexes], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
            # ax1.plot(mz_vector[upper_indexes], sum_vector[upper_indexes], 'o', markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
            # plt.show()

            # reset the sum vector for the next scan
            sum_vector.fill(0.0)
            scan_stop = time.time()
            # print("    {} seconds for scan".format(scan_stop-scan_start))

        # print("  writing frame {} ({} points) to the database.".format(summedFrameId, len(points)))
        dest_c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", points)
        dest_conn.commit()
        summedFrameId += 1
        frame_stop = time.time()
        print("  {} seconds for frame".format(frame_stop-frame_start))

    # Close the connection
    source_conn.close()
    dest_conn.close()

if __name__ == "__main__":
    main()
1400