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
FEATURE_START_FRAME_IDX = 1
FEATURE_END_FRAME_IDX = 2
FEATURE_SCAN_LOWER_IDX = 3
FEATURE_SCAN_UPPER_IDX = 4
FEATURE_MZ_LOWER_IDX = 5
FEATURE_MZ_UPPER_IDX = 6

# frame array indices
FRAME_ID_IDX = 0
FRAME_MZ_IDX = 1
FRAME_SCAN_IDX = 2
FRAME_INTENSITY_IDX = 3

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

# Find the source MS2 frame IDs corresponding to the specified summed MS1 frame ID
def ms2_frame_ids_from_ms1_frame_id(ms1_frame_id, frames_to_sum, frame_summing_offset):

    # find the set of frames summed to make this MS1 frame
    lower_source_frame_index = (ms1_frame_id-1) * frame_summing_offset
    upper_source_frame_index = lower_source_frame_index + frames_to_sum
    return tuple(ms2_frame_ids_v[lower_source_frame_index:upper_source_frame_index,0])

@profile
def main():
    global ms2_frame_ids_v
    feature_count = 0

    parser = argparse.ArgumentParser(description='Sum MS2 frames in the region of the MS1 feature\'s drift and retention time.')
    parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
    parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
    parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
    parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
    parser.add_argument('-nrf','--number_of_random_features', type=int, help='Randomly select this many features from the specified feature range.', required=False)
    parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=True)
    parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=True)
    parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
    parser.add_argument('-ms1ce','--ms1_collision_energy', type=float, help='Collision energy used for MS1.', required=True)
    parser.add_argument('-fts','--frames_to_sum', type=int, help='The number of MS2 source frames to sum.', required=True)
    parser.add_argument('-fso','--frame_summing_offset', type=int, help='The number of MS2 source frames to shift for each summation.', required=True)
    parser.add_argument('-mzsf','--mz_scaling_factor', type=float, default=1000.0, help='Scaling factor to convert m/z range to integers.', required=False)
    parser.add_argument('-rff','--random_features_file', type=str, help='A text file containing the feature indexes to process.', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=5000, help='The number of features to be written to the database.', required=False)
    parser.add_argument('-frso','--feature_region_scan_offset', type=int, default=3, help='Cater to the drift offset in ms2 by expanding the feature region scan range.', required=False)
    parser.add_argument('-mspp','--minimum_summed_points_per_peak', type=int, default=4, help='Minimum number of summed points to form a peak.', required=False)
    args = parser.parse_args()

    if (args.random_features_file is not None) and (args.number_of_random_features is not None):
        print("Error: cannot specify -nrf and -rff at the same time.")
        exit

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
        dest_c.execute("CREATE TABLE summed_ms2_regions (feature_id INTEGER, peak_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, PRIMARY KEY (feature_id, peak_id, point_id))")
        dest_c.execute("CREATE TABLE summed_ms2_regions_info (item TEXT, value TEXT)")

        dest_c.execute("DROP TABLE IF EXISTS ms2_peaks")
        dest_c.execute("CREATE TABLE ms2_peaks (feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, composite_mzs_min INTEGER, composite_mzs_max INTEGER, centroid_scan INTEGER, intensity INTEGER, cofi_scan REAL, cofi_rt REAL, PRIMARY KEY (feature_id, peak_id))")

        dest_c.execute("DROP TABLE IF EXISTS ms2_feature_region_points")

        # Store the arguments as metadata in the database for later reference
        info = []
        for arg in vars(args):
            info.append((arg, getattr(args, arg)))

        start_run = time.time()

        print("Loading the MS2 frame IDs")
        ms2_frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy <> {} order by frame_id ASC;".format(args.ms1_collision_energy), conv_conn)
        ms2_frame_ids_v = ms2_frame_ids_df.values
        print("{} MS2 frames loaded".format(len(ms2_frame_ids_v)))

        # calculate the ms2 frame rate - assume they alternate 
        df = pd.read_sql_query("select value from convert_info where item=\'{}\'".format("raw_frame_period_in_msec"), conv_conn)
        raw_frame_period_in_msec = float(df.loc[0].value)
        raw_frame_ids_per_second = 1.0 / (raw_frame_period_in_msec * 10**-3)
        print("ms2 raw frames per second: {}".format(raw_frame_ids_per_second))

        if len(ms2_frame_ids_v) > 0:

            # Take the ms1 features within the m/z band of interest, and sum the ms2 frames over the whole m/z 
            # range (as we don't know where the fragments will be in ms2)

            print("Number of source frames that were summed {}, with offset {}".format(args.frames_to_sum, args.frame_summing_offset))
            print("Loading the MS1 features")
            features_df = pd.read_sql_query("""select feature_id,start_frame,end_frame,scan_lower,scan_upper,mz_lower,mz_upper 
                from features where feature_id >= {} and feature_id <= {} and 
                charge_state >= {} and mz_lower <= {} and mz_upper >= {} order by feature_id ASC;""".format(
                    args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state, args.mz_upper, args.mz_lower), 
                conv_conn)
            if args.number_of_random_features is not None:
                # Create a subset of features selected at random
                random_feature_indexes = random.sample(range(len(features_df)), args.number_of_random_features)
                random_feature_indexes_file = open('random_feature_indexes.txt', 'w')
                for item in random_feature_indexes:
                    random_feature_indexes_file.write("%s\n" % item)
                random_feature_indexes_file.close()
                features_df = features_df.iloc[random_feature_indexes]
            if args.random_features_file is not None:
                # read the file of feature indexes
                random_feature_indexes_file = open(args.random_features_file, 'r')
                random_feature_indexes = list(map(int, random_feature_indexes_file.read().splitlines()))
                random_feature_indexes_file.close()
                features_df = features_df.iloc[random_feature_indexes]
            features_v = features_df.values
            print("{} MS1 features loaded (feature IDs {})".format(len(features_v), features_df.feature_id.values))

            points = []
            peaks = []
            for feature in features_v:
                feature_start_time = time.time()

                feature_id = int(feature[FEATURE_ID_IDX])
                feature_start_frame = int(feature[FEATURE_START_FRAME_IDX])
                feature_end_frame = int(feature[FEATURE_END_FRAME_IDX])
                feature_scan_lower = int(feature[FEATURE_SCAN_LOWER_IDX]) - args.feature_region_scan_offset
                feature_scan_upper = int(feature[FEATURE_SCAN_UPPER_IDX]) + args.feature_region_scan_offset

                point_id = 1
                peak_id = 1

                # Load the MS2 frame points for the feature's region
                ms2_frame_ids = ()
                for frame_id in range(feature_start_frame, feature_end_frame+1):
                    ms2_frame_ids += ms2_frame_ids_from_ms1_frame_id(frame_id, args.frames_to_sum, args.frame_summing_offset)
                ms2_frame_ids = tuple(set(ms2_frame_ids))   # remove duplicates
                print("feature ID {} ({}% complete), MS1 frame IDs {}-{}, {} MS2 frames, scans {}-{}".format(feature_id, round(float(feature_id-args.feature_id_lower)/(args.feature_id_upper-args.feature_id_lower+1)*100,1), feature_start_frame, feature_end_frame, len(ms2_frame_ids), feature_scan_lower, feature_scan_upper))
                frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity,point_id from frames where frame_id in {} and scan <= {} and scan >= {} order by scan,mz;".format(ms2_frame_ids, feature_scan_upper, feature_scan_lower), conv_conn)
                # scale the m/z values and make them integers
                frame_df['scaled_mz'] = frame_df.mz * args.mz_scaling_factor
                frame_df = frame_df.astype(np.int32)
                # write out this dataframe for processing in a later step
                frame_df['feature_id'] = feature_id
                frame_df.to_sql(name='ms2_feature_region_points', con=dest_conn, if_exists='append', index=False, chunksize=None)
                # take a copy for determining quality of candidate peaks
                ms2_feature_region_points_df = frame_df.copy()
                ms2_feature_region_points_df.sort_values('scaled_mz', inplace=True, ascending = True)
                ms2_feature_region_points_df.set_index('scaled_mz', inplace=True)
                # sum the intensity for duplicate rows (scan, mz) - from https://stackoverflow.com/questions/29583312/pandas-sum-of-duplicate-attributes
                frame_df['intensity_combined'] = frame_df.groupby(['scan', 'scaled_mz'])['intensity'].transform('sum')
                # drop the duplicate rows
                frame_df.drop_duplicates(subset=('scan','scaled_mz'), inplace=True)
                # create the frame array
                print("frame_df length {}".len(frame_df))
                print("scan max {}, scaled_mz max {}".format(frame_df.scan.max(), frame_df.scaled_mz.max()))
                frame_a = np.zeros(shape=(int(frame_df.scan.max()+1),int(frame_df.scaled_mz.max()+1)), dtype=np.int32)
                print("allocated {} bytes for the frame array".format(getsizeof(frame_a)))

                # use scaled_mz as an index
                frame_a[frame_df.scan, frame_df.scaled_mz] = frame_df.intensity_combined
                min_scan = frame_df.scan.min()
                max_scan = frame_df.scan.max()
                min_mz = frame_df.scaled_mz.min()
                max_mz = frame_df.scaled_mz.max()
                subset_frame_a = frame_a[min_scan:max_scan+1,min_mz:max_mz+1]

                # sum the feature region intensities by mz (i.e. column)
                summed_intensities_by_mz = subset_frame_a.sum(axis=0)
                sorted_mzs = np.argsort(summed_intensities_by_mz)[::-1]  # need to add min_mz to get back to true mz
                # check where we should stop
                zero_indices = np.where(summed_intensities_by_mz[sorted_mzs] == 0)[0]
                if len(zero_indices) > 0:
                    first_zero_index = zero_indices[0]
                else:
                    first_zero_index = len(sorted_mzs)-1

                peak_count = 0
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
                            peak_points['retention_time_secs'] = peak_points.frame_id / raw_frame_ids_per_second
                            centre_of_intensity_scan = peakutils.centroid(peak_points.scan.astype(float), peak_points.intensity)
                            centre_of_intensity_rt = peakutils.centroid(peak_points.retention_time_secs.astype(float), peak_points.intensity)

                            # add the peak to the list
                            # feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, composite_mzs_min INTEGER, composite_mzs_max INTEGER, centroid_scan INTEGER, intensity INTEGER, centre_of_intensity_scan REAL, centre_of_intensity_rt REAL
                            peaks.append((feature_id, peak_id, centroid_mz_descaled, peak_composite_mzs_min, peak_composite_mzs_max, min_scan+centroid_scan, total_peak_intensity, centre_of_intensity_scan, centre_of_intensity_rt))
                            peak_id += 1
                            peak_count += 1

                        # flag all the mz points we've processed in this peak
                        summed_intensities_by_mz[mzs] = 0

                feature_stop_time = time.time()
                feature_count += 1
                print("{} sec to find {} peaks for feature {} ({} features completed)".format(feature_stop_time-feature_start_time, peak_count, feature_id, feature_count))
                print("")

                if (feature_count % args.batch_size) == 0:
                    print("feature count {} - writing summed regions to the database...".format(feature_count))
                    print("")
                    # Store the points in the database
                    dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, peak_id, point_id, mz, scan, intensity) VALUES (?, ?, ?, ?, ?, ?)", points)
                    dest_conn.commit()
                    del points[:]
                    #                                          feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, composite_mzs_min INTEGER, composite_mzs_max INTEGER, centroid_scan INTEGER, intensity INTEGER, centre_of_intensity_scan REAL, centre_of_intensity_rt REAL
                    dest_c.executemany("INSERT INTO ms2_peaks (feature_id, peak_id, centroid_mz, composite_mzs_min, composite_mzs_max, centroid_scan, intensity, cofi_scan, cofi_rt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", peaks)
                    dest_conn.commit()
                    del peaks[:]

            # Store any remaining points in the database
            if len(points) > 0:
                dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, peak_id, point_id, mz, scan, intensity) VALUES (?, ?, ?, ?, ?, ?)", points)

            # Store any remaining peaks in the database
            if len(peaks) > 0:
                #                                          feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, composite_mzs_min INTEGER, composite_mzs_max INTEGER, centroid_scan INTEGER, intensity INTEGER, centre_of_intensity_scan REAL, centre_of_intensity_rt REAL
                dest_c.executemany("INSERT INTO ms2_peaks (feature_id, peak_id, centroid_mz, composite_mzs_min, composite_mzs_max, centroid_scan, intensity, cofi_scan, cofi_rt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", peaks)

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
