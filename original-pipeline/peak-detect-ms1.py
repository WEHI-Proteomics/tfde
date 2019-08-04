import sys
import sqlite3
import pandas as pd
import argparse
import numpy as np
from scipy import signal
import peakutils
import time
import math
import collections
import json
import os

MIN_POINTS_IN_PEAK_TO_CHECK_FOR_TROUGHS = 10
INSTRUMENT_RESOLUTION = 40000.0

# The FWHM is the m/z / instrument resolution. Std dev is FWHM / 2.35482. See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
def standard_deviation(mz):
    FWHM = mz / INSTRUMENT_RESOLUTION
    return FWHM / 2.35482

# Source: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy/2415343
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

def findNearestGreaterThan(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff<0] = sys.maxint
    idx = diff.argmin()
    return idx, inputData[idx]

def findNearestLessThan(searchVal, inputData):
    diff = inputData - searchVal
    diff[diff>0] = -sys.maxint
    idx = diff.argmax()
    return idx, inputData[idx]

parser = argparse.ArgumentParser(description='A tree descent method for peak detection.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower frame number.', required=True)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper frame number.', required=True)
parser.add_argument('-sl','--scan_lower', type=int, help='The lower scan number.', required=True)
parser.add_argument('-su','--scan_upper', type=int, help='The upper scan number.', required=True)
parser.add_argument('-es','--empty_scans', type=int, default=2, help='Maximum number of empty scans to tolerate.', required=False)
parser.add_argument('-sd','--standard_deviations', type=int, default=4, help='Number of standard deviations to look either side of a point.', required=False)
parser.add_argument('-bs','--batch_size', type=int, default=10000, help='The size of the frames to be written to the database.', required=False)
args = parser.parse_args()

source_conn = sqlite3.connect(args.database_name)
src_c = source_conn.cursor()

print("Setting up tables...")

src_c.execute("DROP TABLE IF EXISTS peaks")
src_c.execute("DROP TABLE IF EXISTS peak_detect_info")

src_c.execute("CREATE TABLE peaks (frame_id INTEGER, peak_id INTEGER, centroid_mz REAL, centroid_scan REAL, intensity_sum INTEGER, scan_upper INTEGER, scan_lower INTEGER, std_dev_mz REAL, std_dev_scan REAL, rationale TEXT, intensity_max INTEGER, peak_max_mz REAL, peak_max_scan INTEGER, retention_time_secs REAL, cluster_id INTEGER, PRIMARY KEY (frame_id, peak_id))")
src_c.execute("CREATE TABLE peak_detect_info (item TEXT, value TEXT)")

print("Setting up indexes...")

src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_frames ON summed_frames (frame_id,intensity)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_frames_2 ON summed_frames (frame_id,point_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_frames_3 ON summed_frames (frame_id,peak_id)")

src_c.execute("update summed_frames set peak_id=0 where frame_id>={} and frame_id<={} and peak_id!=0".format(args.frame_lower, args.frame_upper))

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

# Peak array indices
FRAME_MZ_IDX = 0
FRAME_SCAN_IDX = 1
FRAME_INTENSITY_IDX = 2
FRAME_POINT_ID_IDX = 3
FRAME_RT_IDX = 4

mono_peaks = []
point_updates = []
start_run = time.time()
frame_count = 0
for frame_id in range(args.frame_lower, args.frame_upper+1):
    peak_id = 1
    frame_df = pd.read_sql_query("select mz,scan,intensity,point_id,retention_time_secs from summed_frames where frame_id={} order by intensity desc;".format(frame_id), source_conn)
    if len(frame_df) > 0:
        print("Detecting peaks in ms1 frame {} ({}% complete)".format(frame_id, round(float(frame_id-args.frame_lower)/(args.frame_upper-args.frame_lower+1)*100,1)))
        start_frame = time.time()
        frame_v = frame_df.values
        # Find the intensity of the point at the bottom of the top tenth percentile of points
        number_of_points = len(frame_v)
        top_tenth_percentile_index = int(number_of_points / 10)
        minimum_intensity = int(frame_v[top_tenth_percentile_index][FRAME_INTENSITY_IDX])
        # print("frame occupies {} bytes".format(frame_v.nbytes))
        while len(frame_v) > 0:
            peak_indices = np.empty(0, dtype=int)

            rationale = collections.OrderedDict()
            max_intensity_index = frame_v.argmax(axis=0)[FRAME_INTENSITY_IDX]
            mz = frame_v[max_intensity_index][FRAME_MZ_IDX]
            scan = int(frame_v[max_intensity_index][FRAME_SCAN_IDX])
            intensity = int(frame_v[max_intensity_index][FRAME_INTENSITY_IDX])
            point_id = int(frame_v[max_intensity_index][FRAME_POINT_ID_IDX])
            peak_indices = np.append(peak_indices, max_intensity_index)
            rationale["highest intensity point id"] = point_id

            # Stop looking for peaks if we've reached the bottom of the top tenth percentile of points
            if intensity < minimum_intensity:
                break

            # Look for other points belonging to this peak
            std_dev_window = standard_deviation(mz) * args.standard_deviations
            # Look in the 'up' direction
            scan_offset = 1
            missed_scans = 0
            while (missed_scans < args.empty_scans) and (scan-scan_offset >= args.scan_lower):
                nearby_indices_up = np.where((frame_v[:,FRAME_SCAN_IDX] == scan-scan_offset) & (frame_v[:,FRAME_MZ_IDX] >= mz - std_dev_window) & (frame_v[:,FRAME_MZ_IDX] <= mz + std_dev_window))[0]
                nearby_points_up = frame_v[nearby_indices_up]
                if len(nearby_indices_up) == 0:
                    missed_scans += 1
                else:
                    if len(nearby_indices_up) > 1:
                        # take the most intense point if there's more than one point found on this scan
                        frame_v_index_to_use = nearby_indices_up[np.argmax(nearby_points_up[:,FRAME_INTENSITY_IDX])]
                    else:
                        frame_v_index_to_use = nearby_indices_up[0]
                    # Update the m/z window
                    mz = frame_v[frame_v_index_to_use][FRAME_MZ_IDX]
                    std_dev_window = standard_deviation(mz) * args.standard_deviations
                    missed_scans = 0
                    peak_indices = np.append(peak_indices, frame_v_index_to_use)
                scan_offset += 1
            # Look in the 'down' direction
            scan_offset = 1
            missed_scans = 0
            mz = frame_v[max_intensity_index][FRAME_MZ_IDX]
            std_dev_window = standard_deviation(mz) * args.standard_deviations
            while (missed_scans < args.empty_scans) and (scan+scan_offset <= args.scan_upper):
                nearby_indices_down = np.where((frame_v[:,FRAME_SCAN_IDX] == scan+scan_offset) & (frame_v[:,FRAME_MZ_IDX] >= mz - std_dev_window) & (frame_v[:,FRAME_MZ_IDX] <= mz + std_dev_window))[0]
                nearby_points_down = frame_v[nearby_indices_down]
                if len(nearby_indices_down) == 0:
                    missed_scans += 1
                else:
                    if len(nearby_indices_down) > 1:
                        # take the most intense point if there's more than one point found on this scan
                        frame_v_index_to_use = nearby_indices_down[np.argmax(nearby_points_down[:,FRAME_INTENSITY_IDX])]
                    else:
                        frame_v_index_to_use = nearby_indices_down[0]
                    
                    # Update the m/z window
                    mz = frame_v[frame_v_index_to_use][FRAME_MZ_IDX]
                    std_dev_window = standard_deviation(mz) * args.standard_deviations
                    missed_scans = 0
                    peak_indices = np.append(peak_indices, frame_v_index_to_use)
                scan_offset += 1

            if len(peak_indices) > 1:
                if len(peak_indices) > MIN_POINTS_IN_PEAK_TO_CHECK_FOR_TROUGHS:
                    # Check whether it has more than one peak
                    # filter the intensity with a Gaussian filter
                    sorted_peaks_indexes = np.argsort(frame_v[peak_indices][:,FRAME_SCAN_IDX])
                    peaks_sorted = frame_v[peak_indices[sorted_peaks_indexes]]
                    rationale["point ids"] = peaks_sorted[:,FRAME_POINT_ID_IDX].astype(int).tolist()
                    filtered = signal.savgol_filter(peaks_sorted[:,FRAME_INTENSITY_IDX], 9, 5)
                    max_index = np.argmax(peaks_sorted[:,FRAME_INTENSITY_IDX])

                    peak_maxima_indexes = peakutils.indexes(filtered, thres=0.05, min_dist=2)
                    peak_minima_indexes = []
                    if len(peak_maxima_indexes) > 1:
                        for idx,peak_maxima_index in enumerate(peak_maxima_indexes):
                            if idx>0:
                                intensities_between_maxima = filtered[peak_maxima_indexes[idx-1]:peak_maxima_indexes[idx]+1]
                                minimum_intensity_index = np.argmin(intensities_between_maxima)+peak_maxima_indexes[idx-1]
                                peak_minima_indexes.append(minimum_intensity_index)

                    indices_to_delete = np.empty(0)
                    if len(peak_minima_indexes) > 0:
                        idx,lower_snip = findNearestLessThan(max_index, peak_minima_indexes)
                        idx,upper_snip = findNearestGreaterThan(max_index, peak_minima_indexes)
                        if lower_snip < max_index:
                            indices_to_delete = np.concatenate((indices_to_delete,np.arange(lower_snip)))
                        if upper_snip > max_index:
                            indices_to_delete = np.concatenate((indices_to_delete,np.arange(upper_snip+1,len(peaks_sorted))))
                        sorted_peaks_indexes = np.delete(sorted_peaks_indexes, indices_to_delete, 0)
                        peak_indices = peak_indices[sorted_peaks_indexes]
                        peaks_sorted = frame_v[peak_indices]
                        rationale["point ids after trimming"] = peaks_sorted[:,FRAME_POINT_ID_IDX].astype(int).tolist()

                # Add the peak to the collection
                peak_mz = []
                peak_scan = []
                peak_intensity = []

                # Update database
                for p in peak_indices:
                    # Collect all the points in the peak
                    peak_mz.append(frame_v[p][FRAME_MZ_IDX])
                    peak_scan.append(frame_v[p][FRAME_SCAN_IDX])
                    peak_intensity.append(int(frame_v[p][FRAME_INTENSITY_IDX]))

                    # Assign this peak ID to all the points in the peak
                    point_updates.append((int(peak_id), int(frame_id), int(frame_v[int(p)][FRAME_POINT_ID_IDX])))

                # Add the peak's details to the collection
                peak_intensity_sum = np.sum(peak_intensity)
                peak_intensity_max = np.max(peak_intensity)
                peak_scan_upper = np.max(peak_scan)
                peak_scan_lower = np.min(peak_scan)
                peak_mz_centroid, peak_std_dev_mz = weighted_avg_and_std(values=peak_mz, weights=peak_intensity)
                peak_scan_centroid, peak_std_dev_scan = weighted_avg_and_std(values=peak_scan, weights=peak_intensity)
                peak_points = frame_v[peak_indices]
                peak_max_index = peak_points[:,FRAME_INTENSITY_IDX].argmax()
                peak_max_mz = peak_points[peak_max_index][FRAME_MZ_IDX]
                peak_max_scan = peak_points[peak_max_index][FRAME_SCAN_IDX].astype(int)
                peak_centroid_rt = peakutils.centroid(peak_points[:,FRAME_RT_IDX], peak_points[:,FRAME_INTENSITY_IDX])
                cluster_id = 0
                mono_peaks.append((int(frame_id), int(peak_id), float(peak_mz_centroid), float(peak_scan_centroid), int(peak_intensity_sum), int(peak_scan_upper), int(peak_scan_lower), float(peak_std_dev_mz), 
                    float(peak_std_dev_scan), json.dumps(rationale), int(peak_intensity_max), float(peak_max_mz), int(peak_max_scan), peak_centroid_rt, int(cluster_id)))

                peak_id += 1

            # remove the points we've processed from the frame
            frame_v = np.delete(frame_v, peak_indices, 0)
    else:
        print("No points were found for frame {}".format(frame_id))

    stop_frame = time.time()
    print("{} seconds to detect peaks in ms1 frame {} - found {} peaks".format(stop_frame-start_frame, frame_id, peak_id-1))
    frame_count += 1
    # check if we've processed a batch number of frames - store in database if so
    if (frame_count % args.batch_size == 0):
        print("frame count {} - writing peaks to the database...".format(frame_count))

        # Write out the peaks
        src_c.executemany("INSERT INTO peaks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", mono_peaks)
        del mono_peaks[:]

        # Update the points in the frame table
        src_c.executemany("UPDATE summed_frames SET peak_id=? WHERE frame_id=? AND point_id=?", point_updates)
        del point_updates[:]

        source_conn.commit()

if len(mono_peaks) > 0:
    # Write out the remaining peaks
    src_c.executemany("INSERT INTO peaks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", mono_peaks)
    del mono_peaks[:]

if len(point_updates) > 0:
    # Update the points in the frame table
    src_c.executemany("UPDATE summed_frames SET peak_id=? WHERE frame_id=? AND point_id=?", point_updates)
    del point_updates[:]

stop_run = time.time()

# write out the processing info
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))

print("{} info: {}".format(parser.prog, info))

info_entry = []
info_entry.append(("summed frames {}-{}".format(args.frame_lower, args.frame_upper), json.dumps(info)))

src_c.executemany("INSERT INTO peak_detect_info VALUES (?, ?)", info_entry)
source_conn.commit()
source_conn.close()