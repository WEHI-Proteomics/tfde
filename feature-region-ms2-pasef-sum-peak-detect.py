from __future__ import print_function
import sys
import argparse
import numpy as np
import time
import pandas as pd
import peakutils
from operator import itemgetter
import sqlite3
import random
import json
from sys import getsizeof
import os
import traceback

#
# python -u ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect.py -cdb './UPS2_allion/UPS2_allion.sqlite' -ddb './UPS2_allion/UPS2_allion-features-1-455.sqlite' -ms1ce 10 -fl 1 -fu 455 -ml 100.0 -mu 2200.0 -bs 20 -fts 30 -fso 5 -mzsf 1000.0
#

# feature array indices
FEATURE_ID_IDX = 0
FEATURE_START_RT_IDX = 1
FEATURE_END_RT_IDX = 2
FEATURE_SCAN_LOWER_IDX = 3
FEATURE_SCAN_UPPER_IDX = 4

# frame array indices
FRAME_ID_IDX = 0
FRAME_MZ_IDX = 1
FRAME_SCAN_IDX = 2
FRAME_INTENSITY_IDX = 3
FRAME_RT_IDX = 4

# to avoid warnings about assigning to a dataframe view not being reflected in the original
pd.options.mode.chained_assignment = None

# so we can use profiling without removing @profile
import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return int((mz / instrument_resolution) / 2.35482)

@profile
def main():
    global ms2_frame_ids_v
    feature_count = 0

    parser = argparse.ArgumentParser(description='Sum MS2 frames in the region of the MS1 feature\'s drift and retention time.')
    parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
    parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
    parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
    parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
    parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
    parser.add_argument('-ms1ce','--ms1_collision_energy', type=float, help='Collision energy used for MS1.', required=True)
    parser.add_argument('-fts','--frames_to_sum', type=int, help='The number of MS2 source frames to sum.', required=True)
    parser.add_argument('-fso','--frame_summing_offset', type=int, help='The number of MS2 source frames to shift for each summation.', required=True)
    parser.add_argument('-mzsf','--mz_scaling_factor', type=float, default=1000.0, help='Scaling factor to convert m/z range to integers.', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=5000, help='The number of features to be written to the database.', required=False)
    parser.add_argument('-frso','--feature_region_scan_offset', type=int, default=3, help='Cater to the drift offset in ms2 by expanding the feature region scan range.', required=False)
    parser.add_argument('-mspp','--minimum_summed_points_per_peak', type=int, default=4, help='Minimum number of summed points to form a peak.', required=False)
    args = parser.parse_args()

    try:
        conv_conn = sqlite3.connect(args.converted_database_name)
        conv_c = conv_conn.cursor()

        dest_conn = sqlite3.connect(args.destination_database_name)
        dest_c = dest_conn.cursor()

        # from https://stackoverflow.com/questions/43741185/sqlite3-disk-io-error
        dest_c.execute("PRAGMA journal_mode = TRUNCATE")

        # Set up the tables if they don't exist already
        print("Setting up tables")
        dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions")
        dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions_info")
        dest_c.execute("DROP TABLE IF EXISTS feature_isolation_matches")
        dest_c.execute("CREATE TABLE summed_ms2_regions (feature_id INTEGER, peak_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, PRIMARY KEY (feature_id, peak_id, point_id))")
        dest_c.execute("CREATE TABLE summed_ms2_regions_info (item TEXT, value TEXT)")
        dest_c.execute("CREATE TABLE feature_isolation_matches (feature_id INTEGER, frame_id INTEGER, precursor_id INTEGER)")

        dest_c.execute("DROP TABLE IF EXISTS ms2_peaks")
        dest_c.execute("CREATE TABLE ms2_peaks (feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, composite_mzs_min INTEGER, composite_mzs_max INTEGER, centroid_scan INTEGER, intensity INTEGER, cofi_scan REAL, cofi_rt REAL, precursor INTEGER, PRIMARY KEY (feature_id, peak_id))")

        dest_c.execute("DROP TABLE IF EXISTS ms2_feature_region_points")

        # Store the arguments as metadata in the database for later reference
        info = []
        for arg in vars(args):
            info.append((arg, getattr(args, arg)))

        start_run = time.time()

        # load the features
        feature_list_df = pd.read_sql_query("select * from feature_list where feature_id >= {} and feature_id <= {} and charge_state >= {} order by feature_id ASC".format(args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state), dest_conn)

        # load the isolation windows from the instrument database
        instrument_db_name = '/home/ubuntu/HeLa_20KInt_2KIT_Slot1-46_01_1179.d/analysis.tdf'
        db_conn = sqlite3.connect(instrument_db_name)
        isolation_window_df = pd.read_sql_query("select * from PasefFrameMsMsInfo", db_conn)
        db_conn.close()

        # load the frame properties to get the retention time
        converted_db_name = '/home/ubuntu/HeLa_20KInt/HeLa_20KInt.sqlite'
        db_conn = sqlite3.connect(converted_db_name)
        frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties", db_conn)
        db_conn.close()

        # augment the isolation windows with their retention time
        isolation_window_df = pd.merge(isolation_window_df, frame_properties_df, how='left', left_on=['Frame'], right_on=['frame_id'])
        isolation_window_df.drop(['frame_id', 'CollisionEnergy'], axis=1, inplace=True)

        # and their mz window
        isolation_window_df['mz_lower'] = isolation_window_df.IsolationMz - (isolation_window_df.IsolationWidth / 2)
        isolation_window_df['mz_upper'] = isolation_window_df.IsolationMz + (isolation_window_df.IsolationWidth / 2)

        # for each feature, find all the matching isolation windows
        isolation_window_df['feature_id'] = 0
        features_with_isolation_matches = []
        points = []
        peaks = []

        for idx in range(len(feature_list_df)):
            feature_start_time = time.time()

            feature_df = feature_list_df.iloc[idx]
            feature_id = int(feature_df.feature_id)
            mono_peak_centroid_mz = feature_df.mono_peak_centroid_mz
            mono_peak_scan_lower = feature_df.mono_peak_scan_lower
            mono_peak_scan_upper = feature_df.mono_peak_scan_upper
            mono_peak_centroid_scan = feature_df.mono_peak_centroid_scan
            rt_lower = feature_df.feature_start_rt
            rt_upper = feature_df.feature_end_rt
            matches_df = isolation_window_df[
                                (
                                    (isolation_window_df.mz_lower <= mono_peak_centroid_mz) &
                                    (isolation_window_df.mz_upper >= mono_peak_centroid_mz)
                                ) &
                                (
                                    (
                                        (isolation_window_df.ScanNumBegin <= mono_peak_scan_lower) &
                                        (isolation_window_df.ScanNumEnd >= mono_peak_scan_lower)
                                    ) |
                                    (
                                        (isolation_window_df.ScanNumBegin <= mono_peak_scan_upper) &
                                        (isolation_window_df.ScanNumEnd >= mono_peak_scan_upper)
                                    ) |
                                    (
                                        (isolation_window_df.ScanNumBegin <= mono_peak_centroid_scan) &
                                        (isolation_window_df.ScanNumEnd >= mono_peak_centroid_scan)
                                    )
                                ) &
                                (
                                    (isolation_window_df.retention_time_secs >= rt_lower) &
                                    (isolation_window_df.retention_time_secs <= rt_upper)
                                )
            ]

            # keep the feature matches
            for match_idx in range(len(matches_df)):
                frame = matches_df.iloc[match_idx].Frame
                precursor = matches_df.iloc[match_idx].Precursor
                features_with_isolation_matches.append((feature_id, int(frame), int(precursor)))

            print("feature {} matched with {} isolation windows".format(feature_id, len(matches_df)))

            point_id = 1
            peak_id = 1
            feature_peak_count = 0

            # for each matching precursor group, sum the raw points from the ms2 frames
            precursor_groups = matches_df.groupby('Precursor')
            for precursor, match_group_df in precursor_groups:
                precursor_peak_count = 0
                # build the 'frame' of raw points for a precursor by loading the ms2 frame points for the isolation window's region
                for match_idx in range(len(match_group_df)):
                    match_df = match_group_df.iloc[match_idx]
                    ms2_frame_id = match_df.Frame
                    isolation_scan_lower = match_df.ScanNumBegin
                    isolation_scan_upper = match_df.ScanNumEnd
                    df = pd.read_sql_query("select frame_id,mz,scan,intensity,point_id,retention_time_secs from frames where frame_id == {} and scan <= {} and scan >= {} order by scan,mz;".format(ms2_frame_id, isolation_scan_upper, isolation_scan_lower), conv_conn)
                    if match_idx == 0:
                        frame_df = df.copy()
                    else:
                        frame_df = frame_df.append(df)
                    print("feature {}: added {} rows to the frame for precursor {}, total {} rows".format(feature_id, len(df), precursor, len(frame_df)))

                print("processing the raw ms2 points for feature {}".format(feature_id))
                if len(frame_df) > 0:
                    # scale the m/z values and make them integers
                    frame_df['scaled_mz'] = frame_df.mz * args.mz_scaling_factor
                    frame_df = frame_df.astype(np.int32)
                    frame_df['feature_id'] = feature_id
                    # take a copy for determining quality of candidate peaks
                    ms2_feature_region_points_df = frame_df.copy()
                    ms2_feature_region_points_df.sort_values('scaled_mz', inplace=True, ascending = True)
                    ms2_feature_region_points_df.set_index('scaled_mz', inplace=True)
                    # sum the intensity for duplicate rows (scan, mz) - from https://stackoverflow.com/questions/29583312/pandas-sum-of-duplicate-attributes
                    frame_df['intensity_combined'] = frame_df.groupby(['scan', 'scaled_mz'])['intensity'].transform('sum')
                    # drop the duplicate rows
                    frame_df.drop_duplicates(subset=('scan','scaled_mz'), inplace=True)
                    # create the frame array
                    frame_a = np.zeros(shape=(int(frame_df.scan.max()+1),int(frame_df.scaled_mz.max()+1)), dtype=np.int32)

                    # use scaled_mz as an index
                    frame_a[frame_df.scan, frame_df.scaled_mz] = frame_df.intensity_combined
                    min_scan = frame_df.scan.min()
                    max_scan = frame_df.scan.max()
                    min_mz = frame_df.scaled_mz.min()
                    max_mz = frame_df.scaled_mz.max()
                    subset_frame_a = frame_a[min_scan:max_scan+1,min_mz:max_mz+1]

                    # sum the feature region intensities by mz (i.e. column)
                    summed_intensities_by_mz = subset_frame_a.sum(axis=0)
                    # write out the array for visualisation purposes
                    summed_intensities_by_mz_df = pd.DataFrame(summed_intensities_by_mz, columns=['intensity'])
                    summed_intensities_by_mz_df['scaled_mz'] = summed_intensities_by_mz_df.index + min_mz
                    summed_intensities_by_mz_df['mz'] = summed_intensities_by_mz_df.scaled_mz / args.mz_scaling_factor
                    summed_intensities_by_mz_df.to_csv('/home/ubuntu/ms2-region-feature-{}.csv'.format(feature_id), mode='w', sep=',', index=True, header=True)
                    # sort by decreasing intensity
                    sorted_mzs = np.argsort(summed_intensities_by_mz)[::-1]  # need to add min_mz to get back to true mz
                    # check where we should stop
                    zero_indices = np.where(summed_intensities_by_mz[sorted_mzs] == 0)[0]
                    if len(zero_indices) > 0:
                        first_zero_index = zero_indices[0]
                    else:
                        first_zero_index = len(sorted_mzs)-1

                    for mz in sorted_mzs[:first_zero_index]:
                        if (summed_intensities_by_mz[mz] > 0):  # check if we've processed this mz already
                            # calculate the indices for this point's std dev window
                            four_std_dev = standard_deviation(min_mz + mz) * 4
                            lower_index = max(mz - four_std_dev, 0)
                            upper_index = min(mz + four_std_dev, len(summed_intensities_by_mz)-1)

                            # find the centroid m/z
                            mzs = np.arange(lower_index, upper_index+1)

                            # calculate the peak attributes
                            peak_composite_mzs_min = lower_index + min_mz
                            peak_composite_mzs_max = upper_index + min_mz
                            scans = range(0,subset_frame_a.shape[0])
                            peak_summed_intensities_by_mz = subset_frame_a[:,mzs].sum(axis=0)
                            peak_summed_intensities_by_scan = subset_frame_a[:,mzs].sum(axis=1)
                            total_peak_intensity = peak_summed_intensities_by_mz.sum()  # total intensity of the peak
                            centroid_mz = peakutils.centroid(mzs, peak_summed_intensities_by_mz)
                            centroid_scan = peakutils.centroid(scans, peak_summed_intensities_by_scan)
                            centroid_mz_descaled = float(min_mz + centroid_mz) / args.mz_scaling_factor

                            point_count_for_this_peak = np.count_nonzero(peak_summed_intensities_by_scan)
                            if point_count_for_this_peak >= args.minimum_summed_points_per_peak:
                                # for each summed point in the region, add an entry to the list
                                # write out the non-zero points for this peak
                                for scan in scans:
                                    point_intensity = peak_summed_intensities_by_scan[scan]
                                    if point_intensity > 0:
                                        points.append((feature_id, peak_id, point_id, centroid_mz_descaled, min_scan+scan, point_intensity))
                                        point_id += 1

                                # calculate the peak's centre of intensity
                                peak_points = ms2_feature_region_points_df.loc[peak_composite_mzs_min:peak_composite_mzs_max]
                                centre_of_intensity_scan = peakutils.centroid(peak_points.scan.astype(float), peak_points.intensity)
                                centre_of_intensity_rt = peakutils.centroid(peak_points.retention_time_secs.astype(float), peak_points.intensity)

                                # add the peak to the list
                                peaks.append((feature_id, peak_id, centroid_mz_descaled, peak_composite_mzs_min, peak_composite_mzs_max, min_scan+centroid_scan, total_peak_intensity, centre_of_intensity_scan, centre_of_intensity_rt, int(precursor)))
                                peak_id += 1
                                feature_peak_count += 1
                                precursor_peak_count += 1

                            # flag all the mz points we've processed in this peak
                            summed_intensities_by_mz[mzs] = 0
                else:
                    print("found no points in ms2 for feature {}".format(feature_id))
                print("feature {}: found {} peaks for precursor {}".format(feature_id, precursor_peak_count, precursor))

            feature_stop_time = time.time()
            feature_count += 1
            print("{} sec to find {} peaks for feature {} ({} features completed)".format(feature_stop_time-feature_start_time, feature_peak_count, feature_id, feature_count))
            print("")

            if (feature_count % args.batch_size) == 0:
                print("feature count {} - writing summed regions to the database...".format(feature_count))
                print("")
                # Store the points in the database
                dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, peak_id, point_id, mz, scan, intensity) VALUES (?, ?, ?, ?, ?, ?)", points)
                dest_conn.commit()
                del points[:]
                dest_c.executemany("INSERT INTO ms2_peaks (feature_id, peak_id, centroid_mz, composite_mzs_min, composite_mzs_max, centroid_scan, intensity, cofi_scan, cofi_rt, precursor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", peaks)
                dest_conn.commit()
                del peaks[:]

        # Store any remaining points in the database
        if len(points) > 0:
            dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, peak_id, point_id, mz, scan, intensity) VALUES (?, ?, ?, ?, ?, ?)", points)

        # Store any remaining peaks in the database
        if len(peaks) > 0:
            dest_c.executemany("INSERT INTO ms2_peaks (feature_id, peak_id, centroid_mz, composite_mzs_min, composite_mzs_max, centroid_scan, intensity, cofi_scan, cofi_rt, precursor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", peaks)

        # store the matches between features and isolation windows
        if len(features_with_isolation_matches) > 0:
            dest_c.executemany("INSERT INTO feature_isolation_matches (feature_id, frame_id, precursor_id) VALUES (?, ?, ?)", features_with_isolation_matches)

        stop_run = time.time()

        info.append(("run processing time (sec)", stop_run-start_run))
        info.append(("processed", time.ctime()))
        info.append(("processor", parser.prog))

        print("{} info: {}".format(parser.prog, info))

        info_entry = []
        info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), json.dumps(info)))

        dest_c.executemany("INSERT INTO summed_ms2_regions_info VALUES (?, ?)", info_entry)
        dest_conn.commit()

        dest_conn.close()
        conv_conn.close()
    except Exception as e:
        print("Exception {} caught in {} for {}".format(traceback.format_exc(), parser.prog, info))

if __name__ == "__main__":
    main()
