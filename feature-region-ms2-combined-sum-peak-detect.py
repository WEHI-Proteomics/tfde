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

#
# For processing all the features in a range...
# python -u ./otf-peak-detect/feature-region-ms2-combined-peak-detect.py -cdb /media/data-drive/Hela_20A_20R_500.sqlite -sdb /media/data-drive/Hela_20A_20R_500-features.sqlite -ddb /media/data-drive/Hela_20A_20R_500-features-5-4546-5454.sqlite -fl 4546 -fu 5454 -ms2ce 27.0 -ml 440.0 -mu 555.0
#
# For picking and processing a random selection of feature indexes...
# python -u ./otf-peak-detect/feature-region-ms2-combined-peak-detect.py -cdb /media/data-drive/Hela_20A_20R_500.sqlite -sdb /media/data-drive/Hela_20A_20R_500-features.sqlite -ddb /media/data-drive/Hela_20A_20R_500-features-1-100000-random-1000-sf-1000.sqlite -fl 1 -fu 100000 -ms2ce 27.0 -ml 440.0 -mu 555.0 -nrf 1000
#
# For processing a previously-generated file of feature indexes...
# python -u ./otf-peak-detect/feature-region-ms2-combined-peak-detect.py -cdb /media/data-drive/Hela_20A_20R_500.sqlite -sdb /media/data-drive/Hela_20A_20R_500-features.sqlite -ddb /media/data-drive/Hela_20A_20R_500-features-1-100000-random-1000-sf-1000.sqlite -fl 1 -fu 100000 -ms2ce 27.0 -ml 440.0 -mu 555.0 -rff random_feature_indexes.txt
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

COMMIT_BATCH_SIZE = 100

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
    parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
    parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
    parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
    parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
    parser.add_argument('-nrf','--number_of_random_features', type=int, help='Randomly select this many features from the specified feature range.', required=False)
    parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=True)
    parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=True)
    parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
    parser.add_argument('-ms2ce','--ms2_collision_energy', type=float, help='Collision energy used for MS2.', required=True)
    parser.add_argument('-fts','--frames_to_sum', type=int, default=150, help='The number of MS2 source frames to sum.', required=False)
    parser.add_argument('-fso','--frame_summing_offset', type=int, default=25, help='The number of MS2 source frames to shift for each summation.', required=False)
    parser.add_argument('-mzsf','--mz_scaling_factor', type=float, default=1000.0, help='Scaling factor to convert m/z range to integers.', required=False)
    parser.add_argument('-rff','--random_features_file', type=str, help='A text file containing the feature indexes to process.', required=False)
    args = parser.parse_args()

    if (args.random_features_file is not None) and (args.number_of_random_features is not None):
        print("Error: cannot specify -nrf and -rff at the same time.")
        exit

    conv_conn = sqlite3.connect(args.converted_database_name)
    conv_c = conv_conn.cursor()

    src_conn = sqlite3.connect(args.source_database_name)
    src_c = src_conn.cursor()

    dest_conn = sqlite3.connect(args.destination_database_name)
    dest_c = dest_conn.cursor()

    # from https://stackoverflow.com/questions/43741185/sqlite3-disk-io-error
    dest_c.execute("PRAGMA journal_mode = TRUNCATE")

    # Set up the tables if they don't exist already
    print("Setting up tables")
    dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions")
    dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions_info")
    dest_c.execute("CREATE TABLE summed_ms2_regions (feature_id INTEGER, peak_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER)")
    dest_c.execute("CREATE TABLE summed_ms2_regions_info (item TEXT, value TEXT)")

    dest_c.execute("DROP TABLE IF EXISTS ms2_peaks")
    dest_c.execute("CREATE TABLE ms2_peaks (feature_id INTEGER, peak_id INTEGER, centroid_mz REAL, intensity INTEGER)")

    print("Setting up indexes")
    conv_c.execute("CREATE INDEX IF NOT EXISTS idx_frame_properties_2 ON frame_properties (collision_energy, frame_id)")
    conv_c.execute("CREATE INDEX IF NOT EXISTS idx_frames_2 ON frames (frame_id,scan)")

    src_c.execute("CREATE INDEX IF NOT EXISTS idx_features_1 ON features (feature_id,charge_state,mz_lower,mz_upper)")

    # Store the arguments as metadata in the database for later reference
    ms2_feature_info = []
    for arg in vars(args):
        ms2_feature_info.append((arg, getattr(args, arg)))

    start_run = time.time()

    print("Loading the MS2 frame IDs")
    ms2_frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.ms2_collision_energy), conv_conn)
    ms2_frame_ids_v = ms2_frame_ids_df.values
    print("{} MS2 frames loaded".format(len(ms2_frame_ids_v)))

    if len(ms2_frame_ids_v) > 0:

        # Take the ms1 features within the m/z band of interest, and sum the ms2 frames over the whole m/z 
        # range (as we don't know where the fragments will be in ms2)

        print("Number of source frames that were summed {}, with offset {}".format(args.frames_to_sum, args.frame_summing_offset))
        print("Loading the MS1 features")
        features_df = pd.read_sql_query("""select feature_id,start_frame,end_frame,scan_lower,scan_upper,mz_lower,mz_upper 
            from features where feature_id >= {} and feature_id <= {} and 
            charge_state >= {} and mz_lower <= {} and mz_upper >= {} order by feature_id ASC;""".format(
                args.feature_id_lower, args.feature_id_upper, args.minimum_charge_state, args.mz_upper, args.mz_lower), 
            src_conn)
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

        for feature in features_v:
            feature_start_time = time.time()

            feature_id = int(feature[FEATURE_ID_IDX])
            feature_start_frame = int(feature[FEATURE_START_FRAME_IDX])
            feature_end_frame = int(feature[FEATURE_END_FRAME_IDX])
            feature_scan_lower = int(feature[FEATURE_SCAN_LOWER_IDX])
            feature_scan_upper = int(feature[FEATURE_SCAN_UPPER_IDX])

            # Load the MS2 frame points for the feature's region
            ms2_frame_ids = ()
            for frame_id in range(feature_start_frame, feature_end_frame+1):
                ms2_frame_ids += ms2_frame_ids_from_ms1_frame_id(frame_id, args.frames_to_sum, args.frame_summing_offset)
            ms2_frame_ids = tuple(set(ms2_frame_ids))   # remove duplicates
            print("feature ID {}, MS1 frame IDs {}-{}, {} MS2 frames, scans {}-{}".format(feature_id, feature_start_frame, feature_end_frame, len(ms2_frame_ids), feature_scan_lower, feature_scan_upper))
            frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} and scan <= {} and scan >= {} order by scan,mz;".format(ms2_frame_ids, feature_scan_upper, feature_scan_lower), conv_conn)
            # scale the m/z values and make them integers
            frame_df['scaled_mz'] = frame_df.mz * args.mz_scaling_factor
            frame_df = frame_df.astype(np.int32)
            # sum the intensity for duplicate rows (scan, mz) - from https://stackoverflow.com/questions/29583312/pandas-sum-of-duplicate-attributes
            frame_df['intensity_combined'] = frame_df.groupby(['scan', 'scaled_mz'])['intensity'].transform('sum')
            # drop the duplicate rows
            frame_df.drop_duplicates(subset=('scan','scaled_mz'), inplace=True)
            # create the frame array
            frame_a = np.zeros(shape=(frame_df.scan.max()+1,frame_df.scaled_mz.max()+1), dtype=np.int32)
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

            points = []
            point_id = 1
            peaks = []
            peak_id = 1
            for mz in sorted_mzs[:first_zero_index]:
                if (summed_intensities_by_mz[mz] > 0):  # check if we've processed this mz already
                    # calculate the indices for this point's std dev window
                    four_std_dev = standard_deviation(min_mz + mz) * 4
                    lower_index = max(mz - four_std_dev, 0)
                    upper_index = min(mz + four_std_dev, len(summed_intensities_by_mz)-1)

                    # find the centroid m/z
                    mzs = np.arange(lower_index, upper_index+1)
                    peak_summed_intensities_by_mz = subset_frame_a[:,mzs].sum(axis=0)
                    peak_summed_intensities_by_scan = subset_frame_a[:,mzs].sum(axis=1)
                    total_peak_intensity = peak_summed_intensities_by_mz.sum()  # total intensity of the peak
                    centroid_mz = peakutils.centroid(mzs, peak_summed_intensities_by_mz)
                    centroid_mz_descaled = float(min_mz + centroid_mz) / args.mz_scaling_factor

                    # for each point in the region, add an entry to the list
                    feature_column = np.full(shape=subset_frame_a.shape[0], fill_value=feature_id, dtype=np.int32)
                    peak_column = np.full(shape=subset_frame_a.shape[0], fill_value=peak_id, dtype=np.int32)
                    point_column = np.arange(start=point_id, stop=point_id+subset_frame_a.shape[0], dtype=np.int32)
                    mz_column = np.full(shape=subset_frame_a.shape[0], fill_value=centroid_mz_descaled, dtype=np.float)
                    scan_column = np.arange(start=min_scan, stop=min_scan+subset_frame_a.shape[0], dtype=np.int32)
                    points_a = np.column_stack((feature_column, peak_column, point_column, mz_column, scan_column, peak_summed_intensities_by_scan))
                    points += points_a.tolist()
                    point_id += subset_frame_a.shape[0]

                    # add the peak to the list
                    peaks.append((feature_id, peak_id, centroid_mz_descaled, total_peak_intensity))
                    peak_id += 1

                    # flag all the mz points we've processed in this peak
                    summed_intensities_by_mz[mzs] = 0

            feature_stop_time = time.time()
            feature_count += 1
            print("{} sec for feature {}".format(feature_stop_time-feature_start_time, feature_id))
            print("")

            if feature_count % COMMIT_BATCH_SIZE == 0:
                print("feature count {} - writing summed regions to the database...".format(feature_count))
                print("")
                # Store the points in the database
                dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, peak_id, point_id, mz, scan, intensity) VALUES (?, ?, ?, ?, ?, ?)", points)
                dest_c.executemany("INSERT INTO ms2_peaks (feature_id, peak_id, centroid_mz, intensity) VALUES (?, ?, ?, ?)", peaks)
                dest_conn.commit()
                del points[:]
                del peaks[:]

        # Store any remaining points in the database
        if len(points) > 0:
            dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, peak_id, point_id, mz, scan, intensity) VALUES (?, ?, ?, ?, ?, ?)", points)

        # Store any remaining peaks in the database
        if len(peaks) > 0:
            dest_c.executemany("INSERT INTO ms2_peaks (feature_id, peak_id, centroid_mz, intensity) VALUES (?, ?, ?, ?)", peaks)

        stop_run = time.time()
        print("{} seconds to process run".format(stop_run-start_run))

        ms2_feature_info.append(("run processing time (sec)", stop_run-start_run))
        ms2_feature_info.append(("processed", time.ctime()))

        ms2_feature_info_entry = []
        ms2_feature_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in ms2_feature_info)))

        dest_c.executemany("INSERT INTO summed_ms2_regions_info VALUES (?, ?)", ms2_feature_info_entry)

        dest_conn.commit()

    dest_conn.close()
    src_conn.close()
    conv_conn.close()

if __name__ == "__main__":
    main()
