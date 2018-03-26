from __future__ import print_function
import sys
import argparse
import numpy as np
import time
import pandas as pd
import peakutils
from operator import itemgetter
import sqlite3
import scipy.sparse as sparse

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

MZ_SCALING_FACTOR = 10000.


def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482

# Find the source MS2 frame IDs corresponding to the specified summed MS1 frame ID
def ms2_frame_ids_from_ms1_frame_id(ms1_frame_id, frames_to_sum, frame_summing_offset):

    # find the set of frames summed to make this MS1 frame
    lower_source_frame_index = (ms1_frame_id-1) * frame_summing_offset
    upper_source_frame_index = lower_source_frame_index + frames_to_sum
    return tuple(ms2_frame_ids_v[lower_source_frame_index:upper_source_frame_index,0])

def main():
    global ms2_frame_ids_v

    parser = argparse.ArgumentParser(description='Sum MS2 frames in the region of the MS1 feature\'s drift and retention time.')
    parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
    parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
    parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
    parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=False)
    parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=False)
    parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=True)
    parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=True)
    parser.add_argument('-mf','--noise_threshold', type=int, default=2, help='Minimum number of MS2 frames a point must appear in to be processed.', required=False)
    parser.add_argument('-mcs','--minimum_charge_state', type=int, default=2, help='Minimum charge state to process.', required=False)
    parser.add_argument('-ms2ce','--ms2_collision_energy', type=float, help='Collision energy used for MS2.', required=True)
    parser.add_argument('-fts','--frames_to_sum', type=int, default=150, help='The number of MS2 source frames to sum.', required=False)
    parser.add_argument('-fso','--frame_summing_offset', type=int, default=25, help='The number of MS2 source frames to shift for each summation.', required=False)
    args = parser.parse_args()

    conv_conn = sqlite3.connect(args.converted_database_name)

    src_conn = sqlite3.connect(args.source_database_name)

    dest_conn = sqlite3.connect(args.destination_database_name)
    dest_c = dest_conn.cursor()

    # from https://stackoverflow.com/questions/43741185/sqlite3-disk-io-error
    dest_c.execute("PRAGMA journal_mode = TRUNCATE")

    # Set up the tables if they don't exist already
    print("Setting up tables")
    dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions")
    dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions_info")
    dest_c.execute("CREATE TABLE summed_ms2_regions (feature_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, number_frames INTEGER, peak_id INTEGER)")  # number_frames = number of source frames the point was found in
    dest_c.execute("CREATE TABLE summed_ms2_regions_info (item TEXT, value TEXT)")

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
        features_v = features_df.values
        print("{} MS1 features loaded".format(len(features_v)))

        points = []
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
            row = frame_df.scan.values
            col = (frame_df.mz.values*MZ_SCALING_FACTOR).astype(np.int64)
            data = frame_df.intensity.values
            frame_coo = sparse.coo_matrix((data, (row, col)), dtype=np.int64)
            frame_csr = frame_coo.tocsr()

            # Sum the points in the feature's region, just as we did for MS1 frames
            pointId = 1
            for scan in range(feature_scan_lower, feature_scan_upper+1):
                print("{} points on scan {}".format(len(frame_csr[scan,:].nonzero()[1]), scan))
                scan_v = frame_csr[scan,:].toarray()
                while scan_v.max() > 0:
                    point_mz = scan_v.argmax()
                    std_dev_point_mz_window = int(standard_deviation(point_mz) * 4.0)
                    window_low_mz = point_mz - std_dev_point_mz_window
                    window_high_mz = point_mz + std_dev_point_mz_window
                    # if there is more than one point, we need to work out the sum, centroid
                    mzs_in_window = frame_csr[scan,window_low_mz:window_high_mz+1].nonzero()[1] + window_low_mz
                    window_intensities = scan_v[0,mzs_in_window]
                    centroid_intensity = window_intensities.sum()
                    centroid_mz = peakutils.centroid(mzs_in_window, window_intensities)
                    # add the summed point to the list
                    points.append((feature_id, pointId, centroid_mz/MZ_SCALING_FACTOR, scan, int(round(centroid_intensity)), 0, 0))
                    pointId += 1
                    # flag the points we've processed
                    scan_v[0,mzs_in_window] = 0
            print("")
            feature_stop_time = time.time()
            print("{} sec for feature {}".format(feature_stop_time-feature_start_time, feature_id))

        # Store the points in the database
        dest_c.executemany("INSERT INTO summed_ms2_regions (feature_id, point_id, mz, scan, intensity, number_frames, peak_id) VALUES (?, ?, ?, ?, ?, ?, ?)", points)

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
