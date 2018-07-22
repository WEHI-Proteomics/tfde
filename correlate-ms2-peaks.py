import pandas as pd
import argparse
import numpy as np
import sqlite3
import time
import os
import peakutils
import json

# Number of points either side of the base peak's maximum intensity to check for correlation
BASE_PEAK_CORRELATION_SIDE_POINTS = 3

#
# nohup python -u ./otf-peak-detect/correlate-ms2-peaks.py -db /media/data-drive/Hela_20A_20R_500-features-1-100000-random-1000-sf-1000.sqlite -fl 1 -fu 100000 > correlate-ms2-peaks.log 2>&1 &
#

parser = argparse.ArgumentParser(description='Calculate correlation between MS1 and MS2 peaks for features.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=True)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=True)
parser.add_argument('-st','--scan_tolerance', type=float, help='Filter width in the scan dimension for ms2 peaks.', required=True)
parser.add_argument('-rtt','--rt_tolerance', type=float, help='Filter width (in seconds) in the RT dimension for ms2 peaks.', required=True)
args = parser.parse_args()

source_conn = sqlite3.connect(args.database_name)
src_c = source_conn.cursor()
src_c.execute("PRAGMA journal_mode = TRUNCATE")

conv_db_conn = sqlite3.connect(args.converted_database_name)

# Store the arguments as metadata in the database for later reference
peak_correlation_info = []
for arg in vars(args):
    peak_correlation_info.append((arg, getattr(args, arg)))

print("Setting up tables")
src_c.execute("DROP TABLE IF EXISTS peak_correlation")
src_c.execute("DROP TABLE IF EXISTS peak_correlation_info")
src_c.execute("CREATE TABLE peak_correlation (feature_id INTEGER, base_peak_id INTEGER, ms1_scan_centroid REAL, ms1_rt_centroid REAL, ms2_peak_id INTEGER, ms2_scan_centroid REAL, ms2_rt_centroid REAL, scan_distance REAL, rt_distance REAL, correlation REAL, PRIMARY KEY (feature_id, base_peak_id, ms2_peak_id))")
src_c.execute("CREATE TABLE peak_correlation_info (item TEXT, value TEXT)")

print("Setting up indexes")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_ms2_feature_region_points_1 ON ms2_feature_region_points (feature_id)")

start_run = time.time()

# calculate the ms2 frame rate - assume they alternate 
df = pd.read_sql_query("select value from convert_info where item=\'{}\'".format("raw_frame_period_in_msec"), conv_db_conn)
raw_frame_period_in_msec = float(df.loc[0].value)
raw_frame_ids_per_second = 1.0 / (raw_frame_period_in_msec * 10**-3)

print("Loading the MS1 base peaks for the feature range")
base_peak_ids_df = pd.read_sql_query("select feature_id,base_peak_id from feature_base_peaks where feature_id >= {} and feature_id <= {} order by feature_id ASC;".format(args.feature_id_lower, args.feature_id_upper), source_conn)

peak_correlation = []

print("Finding peak correlations for features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
for feature_ids_idx in range(0,len(base_peak_ids_df)):
    feature_id = base_peak_ids_df.loc[feature_ids_idx].feature_id.astype(int)
    base_peak_id = base_peak_ids_df.loc[feature_ids_idx].base_peak_id.astype(int)

    # get all the points for the feature's ms1 base peak
    ms1_base_peak_points_df = pd.read_sql_query("select * from summed_ms1_regions where feature_id={} and peak_id={}".format(feature_id, base_peak_id), source_conn)

    # create a composite key for feature_id and point_id to make the next step simpler
    ms1_base_peak_points_df['feature_point'] = ms1_base_peak_points_df['feature_id'].map(str) + '|' + ms1_base_peak_points_df['point_id'].map(str)

    # get the mapping from feature points to summed frame points
    ms1_feature_frame_join_df = pd.read_sql_query("select * from ms1_feature_frame_join where feature_id={}".format(feature_id), source_conn)

    # get the raw points
    ms1_feature_frame_join_df['feature_point'] = ms1_feature_frame_join_df['feature_id'].map(str) + '|' + ms1_feature_frame_join_df['feature_point_id'].map(str)
    ms1_feature_frame_join_df['frame_point'] = ms1_feature_frame_join_df['frame_id'].map(str) + '|' + ms1_feature_frame_join_df['frame_point_id'].map(str)
    frame_points = ms1_feature_frame_join_df.loc[ms1_feature_frame_join_df.feature_point.isin(ms1_base_peak_points_df.feature_point)]
    frames_list = tuple(frame_points.frame_id.astype(int))
    frame_point_list = tuple(frame_points.frame_point_id.astype(int))

    # get the summed to raw point mapping
    raw_point_ids_df = pd.read_sql_query("select * from raw_summed_join where summed_frame_id in {} and summed_point_id in {}".format(frames_list,frame_point_list), conv_db_conn)
    raw_point_ids_df['summed_frame_point'] = raw_point_ids_df['summed_frame_id'].map(str) + '|' + raw_point_ids_df['summed_point_id'].map(str)
    raw_point_ids = raw_point_ids_df.loc[raw_point_ids_df.summed_frame_point.isin(frame_points.frame_point)]

    raw_frame_list = tuple(raw_point_ids.raw_frame_id.astype(int))
    raw_point_list = tuple(raw_point_ids.raw_point_id.astype(int))
    raw_points_df = pd.read_sql_query("select frame_id,point_id,mz,scan,intensity from frames where frame_id in {} and point_id in {}".format(raw_frame_list,raw_point_list), conv_db_conn)
    raw_points_df['retention_time_secs'] = raw_points_df.frame_id / raw_frame_ids_per_second

    ms1_centroid_scan = peakutils.centroid(raw_points_df.scan, raw_points_df.intensity)
    ms1_centroid_rt = peakutils.centroid(raw_points_df.retention_time_secs, raw_points_df.intensity)

    ############################
    # Now process the ms2 points
    ############################

    # get the raw (unsummed) points the ms2 peaks mz belonging to this feature
    ms2_feature_region_points_df = pd.read_sql_query("select * from ms2_feature_region_points where feature_id={}".format(feature_id), source_conn)

    # get the ms2 peak summary information for this feature
    ms2_peaks_df = pd.read_sql_query("select * from ms2_peaks where feature_id={} order by peak_id ASC".format(feature_id), source_conn)
    print("correlating {} ms2 peaks for feature {} in range {}-{}".format(len(ms2_peaks_df), feature_id, args.feature_id_lower, args.feature_id_upper))

    # calculate the 2D centroid for each of the feature's ms2 peaks
    for ms2_peak_idx in range(len(ms2_peaks_df)):
        # get all the points for this ms2 peak
        ms2_peak_id = ms2_peaks_df.loc[ms2_peak_idx].peak_id.astype(int)

        # get all the mzs used to create this peak
        print("feature_id {}, ms2 peak id {}, comp mzs {}".format(feature_id, ms2_peak_id, ms2_peaks_df[ms2_peaks_df.peak_id == ms2_peak_id].composite_mzs.item()))
        peak_composite_mzs = json.loads(ms2_peaks_df[ms2_peaks_df.peak_id == ms2_peak_id].composite_mzs.item())
        peak_points = ms2_feature_region_points_df[ms2_feature_region_points_df.scaled_mz.isin(peak_composite_mzs)].sort_values(by=['scan'])
        peak_points.rename(columns = {'frame_id':'raw_frame_id'}, inplace = True)
        peak_points['retention_time_secs'] = peak_points.raw_frame_id / raw_frame_ids_per_second
        ms2_centroid_scan = int(peakutils.centroid(peak_points.scan, peak_points.intensity))
        ms2_centroid_rt = peakutils.centroid(peak_points.retention_time_secs, peak_points.intensity)

        scan_distance = ms1_centroid_scan - ms2_centroid_scan
        rt_distance = ms1_centroid_rt - ms2_centroid_rt
        correlation = 0.0

        peak_correlation.append((feature_id, base_peak_id, ms1_centroid_scan, ms1_centroid_rt, ms2_peak_id, ms2_centroid_scan, ms2_centroid_rt, scan_distance, rt_distance, correlation))

if len(peak_correlation) > 0:
    print("Writing out the peak correlations for features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
    # feature_id, base_peak_id, ms1_centroid_scan, ms1_centroid_rt, ms2_peak_id, ms2_centroid_scan, ms2_centroid_rt, scan_distance, rt_distance, correlation
    src_c.executemany("INSERT INTO peak_correlation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", peak_correlation)
else:
    print("Error: there are no peak correlations for feature range {}-{}".format(args.feature_id_lower, args.feature_id_upper))

stop_run = time.time()
print("{} seconds to process features {} to {}".format(stop_run-start_run, args.feature_id_lower, args.feature_id_upper))

# write out the processing info
peak_correlation_info.append(("run processing time (sec)", stop_run-start_run))
peak_correlation_info.append(("processed", time.ctime()))

peak_correlation_info_entry = []
peak_correlation_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in peak_correlation_info)))

src_c.executemany("INSERT INTO peak_correlation_info VALUES (?, ?)", peak_correlation_info_entry)

source_conn.commit()
source_conn.close()

conv_db_conn.close()
