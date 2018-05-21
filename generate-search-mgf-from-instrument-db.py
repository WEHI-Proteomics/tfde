from __future__ import print_function
import os
import multiprocessing as mp
from multiprocessing import Pool
import sqlite3
import pandas as pd
import argparse
import time
import numpy as np
import sys

def run_process(process):
    os.system(process)

def merge_summed_regions(source_db_name, destination_db_name):
    source_conn = sqlite3.connect(source_db_name)
    src_cur = source_conn.cursor()
    destination_conn = sqlite3.connect(destination_db_name)
    dst_cur = destination_conn.cursor()

    df = pd.read_sql_query("SELECT tbl_name,sql FROM sqlite_master WHERE type='table'", source_conn)
    for t_idx in range(0,len(df)):
        print("merging {}".format(df.loc[t_idx].tbl_name))
        table_df = pd.read_sql_query("SELECT * FROM {}".format(df.loc[t_idx].tbl_name), source_conn)
        table_df.to_sql(name=df.loc[t_idx].tbl_name, con=destination_conn, if_exists='append', index=False, chunksize=10000)

    source_conn.close()
    destination_conn.commit()
    destination_conn.close()

def merge_summed_regions_prep(source_db_name, destination_db_name):
    source_conn = sqlite3.connect(source_db_name)
    destination_conn = sqlite3.connect(destination_db_name)
    dst_cur = destination_conn.cursor()

    df = pd.read_sql_query("SELECT tbl_name,sql FROM sqlite_master WHERE type='table'", source_conn)
    for t_idx in range(0,len(df)):
        print("preparing {}".format(df.loc[t_idx].tbl_name))
        dst_cur.execute("drop table if exists {}".format(df.loc[t_idx].tbl_name))
        dst_cur.execute(df.loc[t_idx].sql)

    source_conn.close()
    destination_conn.commit()
    destination_conn.close()

#
# source activate py27
# nohup python -u ./otf-peak-detect/generate-search-mgf-from-instrument-db.py -dbd ./subset-2500-2740 -idb /media/data-drive/Hela_20A_20R_500_1_01_398.d/ -dbn Hela_20A_20R_500 -cems1 7 -cems2 27 -ml 440 -mu 550 -fl 2500 -fu 2740 -mpc 0.9 -op match_precursor_ms2_peaks > match.log
#

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the search MGF from the instrument database.')
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-idb','--instrument_database_name', type=str, help='The name of the instrument database.', required=True)
parser.add_argument('-dbn','--database_base_name', type=str, help='The base name of the destination databases.', required=True)
parser.add_argument('-fts','--frames_to_sum', type=int, default=150, help='The number of MS1 source frames to sum.', required=False)
parser.add_argument('-fso','--frame_summing_offset', type=int, default=25, help='The number of MS1 source frames to shift for each summation.', required=False)
parser.add_argument('-cems1','--ms1_collision_energy', type=int, help='Collision energy for ms1, in eV.', required=True)
parser.add_argument('-cems2','--ms2_collision_energy', type=int, help='Collision energy for ms2, in eV.', required=True)
parser.add_argument('-mpc','--minimum_peak_correlation', type=float, help='Minimum peak correlation', required=True)
parser.add_argument('-op','--operation', type=str, default='all', help='The operation to perform.', required=False)
parser.add_argument('-nf','--number_of_frames', type=int, help='The number of frames to convert.', required=False)
parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=True)
parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=True)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower summed frame number to process.', required=False)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper summed frame number to process.', required=False)
args = parser.parse_args()

processing_times = []
processing_start_time = time.time()

# make sure the processing directories exist
if not os.path.exists(args.data_directory):
    os.makedirs(args.data_directory)    

converted_database_name = "{}/{}.sqlite".format(args.data_directory, args.database_base_name)
frame_database_root = "{}/{}-frames".format(args.data_directory, args.database_base_name)  # used to split the data into frame-based sections
frame_database_name = converted_database_name  # combined the frame-based sections back into the converted database
feature_database_root = "{}/{}-features".format(args.data_directory, args.database_base_name)  # used to split the data into feature-based sections
feature_database_name = converted_database_name  # combined the feature-based sections back into the converted database

# find out about the compute environment
number_of_cores = mp.cpu_count()

# Set up the processing pool
pool = Pool()

convert_start_time = time.time()

# convert the instrument database
if (args.operation == 'all') or (args.operation == 'convert_instrument_db'):
    if args.number_of_frames is not None:
        run_process("python ./otf-peak-detect/convert-instrument-db.py -sdb {} -ddb {} -nf {}".format(args.instrument_database_name, converted_database_name, args.number_of_frames))
    else:
        run_process("python ./otf-peak-detect/convert-instrument-db.py -sdb {} -ddb {}".format(args.instrument_database_name, converted_database_name))

convert_stop_time = time.time()
processing_times.append(("database conversion", convert_stop_time-convert_start_time))

# find the total number of summed ms1 frames in the database
source_conn = sqlite3.connect(converted_database_name)
frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.ms1_collision_energy), source_conn)
frame_ids = tuple(frame_ids_df.values[:,0])
number_of_summed_frames = 1 + int(((len(frame_ids) - args.frames_to_sum) / args.frame_summing_offset))
source_conn.close()
print("number of summed ms1 frames in the converted database: {}".format(number_of_summed_frames))

if args.frame_lower is None:
    args.frame_lower = 1
    print("frame_lower set to {} from the data".format(args.frame_lower))

if args.frame_upper is None:
    args.frame_upper = number_of_summed_frames
    print("frame_upper set to {} from the data".format(args.frame_upper))

# split the summed frame range into batches
batch_splits = np.array_split(range(args.frame_lower,args.frame_upper+1), number_of_cores)
summed_frame_ranges = []
for s in batch_splits:
    if len(s) > 0:
        summed_frame_ranges.append((s[0],s[len(s)-1]))

# process the ms1 frames
sum_frame_ms1_processes = []
peak_detect_ms1_processes = []
cluster_detect_ms1_processes = []
for summed_frame_range in summed_frame_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(frame_database_root, summed_frame_range[0], summed_frame_range[1])
    sum_frame_ms1_processes.append("python ./otf-peak-detect/sum-frames-ms1.py -sdb {} -ddb {} -ce {} -fl {} -fu {}".format(converted_database_name, destination_db_name, args.ms1_collision_energy, summed_frame_range[0], summed_frame_range[1]))
    peak_detect_ms1_processes.append("python ./otf-peak-detect/peak-detect-ms1.py -db {} -fl {} -fu {}".format(destination_db_name, summed_frame_range[0], summed_frame_range[1]))
    cluster_detect_ms1_processes.append("python ./otf-peak-detect/cluster-detect-ms1.py -db {} -fl {} -fu {}".format(destination_db_name, summed_frame_range[0], summed_frame_range[1]))

# detect clusters in the ms1 frames
if (args.operation == 'all') or (args.operation == 'cluster_detect_ms1'):

    cluster_detect_start_time = time.time()

    run_process("python ./otf-peak-detect/sum-frames-ms1-prep.py -sdb {}".format(converted_database_name))
    pool.map(run_process, sum_frame_ms1_processes)
    pool.map(run_process, peak_detect_ms1_processes)
    pool.map(run_process, cluster_detect_ms1_processes)

    cluster_detect_stop_time = time.time()
    processing_times.append(("cluster detect", cluster_detect_stop_time-cluster_detect_start_time))

    recombine_frames_start_time = time.time()

    # recombine the frame range databases back into a combined database
    template_frame_range = summed_frame_ranges[0]
    template_db_name = "{}-{}-{}.sqlite".format(frame_database_root, template_frame_range[0], template_frame_range[1])
    merge_summed_regions_prep(template_db_name, frame_database_name)
    for summed_frame_range in summed_frame_ranges:
        source_db_name = "{}-{}-{}.sqlite".format(frame_database_root, summed_frame_range[0], summed_frame_range[1])
        print("merging {} into {}".format(source_db_name, frame_database_name))
        merge_summed_regions(source_db_name, frame_database_name)

    recombine_frames_stop_time = time.time()
    processing_times.append(("frame-based recombine", recombine_frames_stop_time-recombine_frames_start_time))

# detect features in the ms1 frames
if (args.operation == 'all') or (args.operation == 'feature_detect_ms1'):
    feature_detect_start_time = time.time()

    print("detecting features...")
    run_process("python ./otf-peak-detect/feature-detect-ms1.py -db {}".format(feature_database_name))

    feature_detect_stop_time = time.time()
    processing_times.append(("feature detect ms1", feature_detect_stop_time-feature_detect_start_time))

# find out how many features were detected
source_conn = sqlite3.connect(feature_database_name)
feature_info_df = pd.read_sql_query("select value from feature_info where item='features found'", source_conn)
number_of_features = int(feature_info_df.values[0][0])
source_conn.close()

# split the summed frame range into batches
batch_splits = np.array_split(range(1,number_of_features+1), number_of_cores)
feature_ranges = []
for s in batch_splits:
    if len(s) > 0:
        feature_ranges.append((s[0],s[len(s)-1]))

#
# from here, split the combined features database into feature range databases
#

# detect ms2 peaks in the feature's region
feature_region_ms2_sum_peak_processes = []
for feature_range in feature_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
    feature_region_ms2_sum_peak_processes.append("python ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect.py -cdb {} -ddb {} -ms2ce {} -fl {} -fu {} -ml {} -mu {} -bs 20".format(converted_database_name, destination_db_name, args.ms2_collision_energy, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))

if (args.operation == 'all') or (args.operation == 'feature_region_ms2_peak_detect'):
    ms2_peak_detect_start_time = time.time()
    run_process("python ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect-prep.py -cdb {}".format(converted_database_name))
    print("detecting ms2 peaks in the feature region...")
    pool.map(run_process, feature_region_ms2_sum_peak_processes)
    ms2_peak_detect_stop_time = time.time()
    processing_times.append(("feature region ms2 peak detect", ms2_peak_detect_stop_time-ms2_peak_detect_start_time))

# re-detect ms1 peaks in the feature's region, and calculate ms2 peak correlation
feature_region_ms1_sum_processes = []
feature_region_ms1_peak_processes = []
peak_correlation_processes = []
for feature_range in feature_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
    feature_region_ms1_sum_processes.append("python ./otf-peak-detect/feature-region-ms1-sum-frames.py -sdb {} -ddb {} -fl {} -fu {} -ml {} -mu {}".format(feature_database_name, destination_db_name, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))
    feature_region_ms1_peak_processes.append("python ./otf-peak-detect/feature-region-ms1-peak-detect.py -sdb {} -ddb {} -fl {} -fu {} -ml {} -mu {}".format(feature_database_name, destination_db_name, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))
    peak_correlation_processes.append("python ./otf-peak-detect/correlate-ms2-peaks.py -db {} -fl {} -fu {}".format(destination_db_name, feature_range[0], feature_range[1]))

if (args.operation == 'all') or (args.operation == 'feature_region_ms1_peak_detect'):
    ms1_peak_detect_start_time = time.time()
    print("summing ms1 frames, detecting peaks in the feature region...")
    run_process("python ./otf-peak-detect/feature-region-ms1-sum-frames-prep.py -sdb {}".format(feature_database_name))
    pool.map(run_process, feature_region_ms1_sum_processes)
    pool.map(run_process, feature_region_ms1_peak_processes)
    ms1_peak_detect_stop_time = time.time()
    processing_times.append(("feature region ms1 peak detect", ms1_peak_detect_stop_time-ms1_peak_detect_start_time))

# determine the drift offset between ms1 and ms2
match_precursor_ms2_peaks_processes = []
for feature_range in feature_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
    match_precursor_ms2_peaks_processes.append("python ./otf-peak-detect/match-precursor-ms2-peaks.py -db {} -fl {} -fu {}".format(destination_db_name, feature_range[0], feature_range[1]))

if (args.operation == 'all') or (args.operation == 'match_precursor_ms2_peaks'):
    match_precursor_ms2_peaks_start_time = time.time()
    print("matching precursor ms2 peaks...")
    pool.map(run_process, match_precursor_ms2_peaks_processes)
    match_precursor_ms2_peaks_stop_time = time.time()
    processing_times.append(("match precursor ms2 peaks", match_precursor_ms2_peaks_stop_time-match_precursor_ms2_peaks_start_time))

if (args.operation == 'all') or (args.operation == 'correlate_peaks'):
    peak_correlation_start_time = time.time()
    print("correlating peaks...")
    pool.map(run_process, peak_correlation_processes)
    peak_correlation_stop_time = time.time()
    processing_times.append(("peak correlation", peak_correlation_stop_time-peak_correlation_start_time))

if (args.operation == 'all') or (args.operation == 'recombine_feature_databases'):
    # recombine the feature range databases back into a combined database
    recombine_feature_databases_start_time = time.time()
    template_feature_range = feature_ranges[0]
    template_db_name = "{}-{}-{}.sqlite".format(feature_database_root, template_feature_range[0], template_feature_range[1])
    merge_summed_regions_prep(template_db_name, feature_database_name)
    for feature_range in feature_ranges:
        source_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        print("merging {} into {}".format(source_db_name, feature_database_name))
        merge_summed_regions(source_db_name, feature_database_name)
    recombine_feature_databases_stop_time = time.time()
    processing_times.append(("feature recombine", recombine_feature_databases_stop_time-recombine_feature_databases_start_time))

if (args.operation == 'all') or (args.operation == 'deconvolve_ms2_spectra'):
    # deconvolve the ms2 spectra with Hardklor
    deconvolve_ms2_spectra_start_time = time.time()
    print("deconvolving ms2 spectra...")
    run_process("python ./otf-peak-detect/deconvolve-ms2-spectra.py -fdb {} -bfn {} -dbd {} -mpc {}".format(feature_database_name, args.database_base_name, args.data_directory, args.minimum_peak_correlation))
    deconvolve_ms2_spectra_stop_time = time.time()
    processing_times.append(("deconvolve ms2 spectra", deconvolve_ms2_spectra_stop_time-deconvolve_ms2_spectra_start_time))

if (args.operation == 'all') or (args.operation == 'create_search_mgf'):
    # create search MGF
    create_search_mgf_start_time = time.time()
    print("creating the search MGF...")
    run_process("python ./otf-peak-detect/create-search-mgf.py -fdb {} -bfn {} -dbd {} -mpc {}".format(feature_database_name, args.database_base_name, args.data_directory, args.minimum_peak_correlation))
    create_search_mgf_stop_time = time.time()
    processing_times.append(("create search mgf", create_search_mgf_stop_time-create_search_mgf_stop_time))

processing_stop_time = time.time()
processing_times.append(("total processing", processing_stop_time-processing_start_time))

# gather statistics
statistics = []
source_conn = sqlite3.connect(converted_database_name)
frame_info_df = pd.read_sql_query("select max(frame_id) from frames", source_conn)
number_of_converted_frames = int(frame_info_df.values[0][0])
statistics.append(("number of converted frames", number_of_converted_frames))
deconvoluted_ions_df = pd.read_sql_query("select max(ion_id) from deconvoluted_ions", source_conn)
number_of_deconvoluted_ions = int(deconvoluted_ions_df.values[0][0])
statistics.append(("number of deconvoluted ions", number_of_deconvoluted_ions))
source_conn.close()

# print statistics
print("")
print("processing times")
for t in processing_times:
    print("{}\t\t{:.1f} seconds\t\t{:.1f}%".format(t[0], t[1], t[1]/(processing_stop_time-processing_start_time)*100.))
print("")
print("data")
for s in statistics:
    print("{}\t\t{:.1f}".format(s[0], s[1]))
