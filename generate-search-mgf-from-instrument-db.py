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
import json

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

def merge_summed_regions(source_db_name, destination_db_name, exceptions):
    print("merging {} with {}".format(source_db_name, destination_db_name))
    source_conn = sqlite3.connect(source_db_name)
    src_cur = source_conn.cursor()
    destination_conn = sqlite3.connect(destination_db_name)	
    dst_cur = destination_conn.cursor()

    df = pd.read_sql_query("SELECT tbl_name,sql FROM sqlite_master WHERE type='table'", source_conn)
    for t_idx in range(0,len(df)):
        table_name = df.loc[t_idx].tbl_name
        if table_name not in exceptions:
            print("merging {}".format(table_name))

            row_count = int(pd.read_sql('SELECT COUNT(*) FROM {table_name}'.format(table_name=table_name), source_conn).values)
            chunksize = 5000000
            number_of_chunks = int(row_count / chunksize)

            for i in range(number_of_chunks + 1):
                print("\tmerging chunk {} of {}".format(i, number_of_chunks))
                query = 'SELECT * FROM {table_name} LIMIT {offset}, {chunksize}'.format(table_name=table_name, offset=i * chunksize, chunksize=chunksize)
                table_df = pd.read_sql_query(query, con=source_conn)
                table_df.to_sql(name=table_name, con=destination_conn, if_exists='append', index=False, chunksize=None)

            # drop the table in the source database
            # src_cur.execute('DROP TABLE IF EXISTS {table_name}'.format(table_name=table_name))
        else:
            print("skipping merge of {}".format(table_name))

    # if we moved all the tables to the destination, delete the database file. Otherwise, vacuum it 
    # to minimise disk space.
    # if len(exceptions) == 0:
    #     source_conn.close()
    #     os.remove(source_db_name)
    # else:
    #     src_cur.execute('VACUUM')
    #     source_conn.close()

    source_conn.close()

def merge_summed_regions_prep(source_db_name, destination_db_name, exceptions):
    print("preparing to merge {} with {}".format(source_db_name, destination_db_name))
    source_conn = sqlite3.connect(source_db_name)
    destination_conn = sqlite3.connect(destination_db_name)
    dst_cur = destination_conn.cursor()

    source_df = pd.read_sql_query("SELECT tbl_name,sql FROM sqlite_master WHERE type='table'", source_conn)
    for t_idx in range(0,len(source_df)):
        table_name = source_df.loc[t_idx].tbl_name
        if table_name not in exceptions:
            print("preparing {}".format(table_name))
            dst_cur.execute("drop table if exists {}".format(table_name))
            print("executing {}".format(source_df.loc[t_idx].sql))
            dst_cur.execute(source_df.loc[t_idx].sql)
        else:
            print("skipping preparation of {}".format(table_name))

    source_conn.close()
    destination_conn.commit()
    destination_conn.close()

# return true if the specified step should be processed
def process_this_step(this_step, first_step):
    result = (processing_steps[this_step] >= processing_steps[first_step])
    return result

# return true if this isn't the last step
def continue_processing(this_step, final_step):
    result = (processing_steps[this_step] < processing_steps[final_step])
    if (result == False) and (args.shutdown_on_completion == True):
        run_process("sudo shutdown -P +5")
    return result
    
#
# source activate py27
# python -u ./otf-peak-detect/generate-search-mgf-from-instrument-db.py -idb /stornext/Sysbio/data/Projects/ProtemicsLab/Development/AllIon/BSA_All_Ion/BSA_All_Ion_Slot1-46_01_266.d -dbd ./BSA_All_Ion -dbn BSA_All_Ion -cems1 10 -mpc 0.9 -fts 30 -fso 5 -op cluster_detect_ms1 > BSA_All_Ion.log 2>&1
#

# check Python version
if not ((sys.version_info.major == 2) and (sys.version_info.minor == 7)):
    raise Exception("The pipeline is written for Python 2.7 but this is {}.{} Exiting.".format(sys.version_info.major, sys.version_info.minor))

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the search MGF from the instrument database.')
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-idb','--instrument_database_name', type=str, help='The name of the instrument database.', required=False)
parser.add_argument('-dbn','--database_base_name', type=str, help='The base name of the destination databases.', required=True)
parser.add_argument('-fts','--frames_to_sum', type=int, help='The number of MS1 source frames to sum.', required=True)
parser.add_argument('-fso','--frame_summing_offset', type=int, help='The number of MS1 source frames to shift for each summation.', required=True)
parser.add_argument('-cems1','--ms1_collision_energy', type=int, help='Collision energy for ms1, in eV.', required=True)
parser.add_argument('-op','--operation', type=str, default='all', help='The operation to perform.', required=False)
parser.add_argument('-fop','--final_operation', type=str, help='The final operation to perform.', required=False)
parser.add_argument('-sd','--shutdown_on_completion', action='store_true', help='Shut down the instance when complete.')
parser.add_argument('-nf','--number_of_frames', type=int, help='The number of frames to convert.', required=False)
parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=False)
parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=False)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower summed frame number to process.', required=False)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper summed frame number to process.', required=False)
parser.add_argument('-mnf','--minimum_number_of_frames', type=int, default=3, help='Minimum number of frames for a feature to be valid.', required=False)
parser.add_argument('-cst','--correlation_scan_tolerance', type=int, help='Scan distance either side of the feature base peak to include an ms2 peak for correlation.', required=True)
parser.add_argument('-crt','--correlation_rt_tolerance', type=int, help='RT distance either side of the feature base peak to include an ms2 peak for correlation.', required=True)
parser.add_argument('-mzsf','--ms2_mz_scaling_factor', type=float, default=1000.0, help='Scaling factor to convert m/z range to integers in ms2.', required=False)
parser.add_argument('-frts','--frame_tasks', type=int, default=1000, help='Number of worker tasks for frames.', required=False)
parser.add_argument('-fets','--feature_tasks', type=int, default=1000, help='Number of worker tasks for features.', required=False)
parser.add_argument('-mnp','--maximum_number_of_peaks_per_feature', type=int, default=500, help='The maximum number of peaks per feature.', required=False)
args = parser.parse_args()

processing_times = []
processing_start_time = time.time()

statistics = []

steps = []
steps.append('convert_instrument_db')
steps.append('cluster_detect_ms1')
steps.append('recombine_frame_databases')
steps.append('feature_detect_ms1')
steps.append('feature_region_ms2_peak_detect')
steps.append('feature_region_ms1_peak_detect')
steps.append('match_precursor_ms2_peaks')
steps.append('correlate_peaks')
steps.append('deconvolve_ms2_spectra')
steps.append('create_search_mgf')
steps.append('recombine_feature_databases')

processing_steps = {j:i for i,j in enumerate(steps)}

if (args.operation == 'all'):
    args.operation = steps[0]
    args.final_operation = steps[-1]

if (args.operation is None):
    args.operation = steps[0]

if (args.final_operation is None):
    args.final_operation = steps[-1]

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

##################################
# OPERATION: convert_instrument_db
##################################
if process_this_step(this_step='convert_instrument_db', first_step=args.operation):
    print("Starting the \'convert_instrument_db\' step")
    convert_start_time = time.time()

    # make sure the processing directories exist
    if not os.path.exists(args.instrument_database_name):
        print("Error - the instrument database directory does not exist. Exiting.")
        sys.exit(1)

    if args.number_of_frames is not None:
        run_process("python -u ./otf-peak-detect/convert-instrument-db.py -sdb '{}' -ddb '{}' -nf {}".format(args.instrument_database_name, converted_database_name, args.number_of_frames))
    else:
        run_process("python -u ./otf-peak-detect/convert-instrument-db.py -sdb '{}' -ddb '{}'".format(args.instrument_database_name, converted_database_name))

    # gather statistics
    source_conn = sqlite3.connect(converted_database_name)
    frame_info_df = pd.read_sql_query("select max(frame_id) from frames", source_conn)
    number_of_converted_frames = int(frame_info_df.values[0][0])
    statistics.append(("number of converted frames", number_of_converted_frames))
    source_conn.close()

    convert_stop_time = time.time()
    processing_times.append(("database conversion", convert_stop_time-convert_start_time))

    if not continue_processing(this_step='convert_instrument_db', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

if not os.path.exists(converted_database_name):
    print("Error - the converted database does not exist. Exiting.")
    sys.exit(1)

# Determine the mass range if it's not specified
if args.mz_lower is None:
    source_conn = sqlite3.connect(converted_database_name)
    df = pd.read_sql_query("select value from convert_info where item = \'mz_lower\'", source_conn)
    source_conn.close()
    if len(df) > 0:
        args.mz_lower = float(df.loc[0].value)
        print("mz_lower set to {} from the data".format(args.mz_lower))
    else:
        print("Error - could not find mz_lower from the convert_info table and it's needed in sebsequent steps. Exiting.")
        sys.exit(1)

if args.mz_upper is None:
    source_conn = sqlite3.connect(converted_database_name)
    df = pd.read_sql_query("select value from convert_info where item = \'mz_upper\'", source_conn)
    source_conn.close()
    if len(df) > 0:
        args.mz_upper = float(df.loc[0].value)
        print("mz_upper set to {} from the data".format(args.mz_upper))
    else:
        print("Error - could not find mz_upper from the convert_info table and it's needed in sebsequent steps. Exiting.")
        sys.exit(1)

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
batch_splits = np.array_split(range(args.frame_lower,args.frame_upper+1), args.frame_tasks)
summed_frame_ranges = []
for s in batch_splits:
    if len(s) > 0:
        summed_frame_ranges.append((s[0],s[len(s)-1]))

###############################
# OPERATION: cluster_detect_ms1
###############################
if process_this_step(this_step='cluster_detect_ms1', first_step=args.operation):
    print("Starting the \'cluster_detect_ms1\' step")
    cluster_detect_start_time = time.time()

    # build the process lists
    sum_frame_ms1_processes = []
    peak_detect_ms1_processes = []
    cluster_detect_ms1_processes = []
    for summed_frame_range in summed_frame_ranges:
        destination_db_name = "{}-{}-{}.sqlite".format(frame_database_root, summed_frame_range[0], summed_frame_range[1])
        sum_frame_ms1_processes.append("python -u ./otf-peak-detect/sum-frames-ms1.py -sdb '{}' -ddb '{}' -ce {} -fl {} -fu {} -fts {} -fso {}".format(converted_database_name, destination_db_name, args.ms1_collision_energy, summed_frame_range[0], summed_frame_range[1], args.frames_to_sum, args.frame_summing_offset))
        peak_detect_ms1_processes.append("python -u ./otf-peak-detect/peak-detect-ms1.py -db '{}' -fl {} -fu {}".format(destination_db_name, summed_frame_range[0], summed_frame_range[1]))
        cluster_detect_ms1_processes.append("python -u ./otf-peak-detect/cluster-detect-ms1.py -db '{}' -fl {} -fu {}".format(destination_db_name, summed_frame_range[0], summed_frame_range[1]))

    run_process("python -u ./otf-peak-detect/sum-frames-ms1-prep.py -sdb '{}'".format(converted_database_name))
    pool.map(run_process, sum_frame_ms1_processes)
    pool.map(run_process, peak_detect_ms1_processes)
    pool.map(run_process, cluster_detect_ms1_processes)

    cluster_detect_stop_time = time.time()
    processing_times.append(("cluster detect", cluster_detect_stop_time-cluster_detect_start_time))

    if not continue_processing(this_step='cluster_detect_ms1', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

######################################
# OPERATION: recombine_frame_databases
######################################
if process_this_step(this_step='recombine_frame_databases', first_step=args.operation):
    print("Starting the \'recombine_frame_databases\' step")
    recombine_frames_start_time = time.time()

    # recombine the frame range databases back into a combined database
    template_frame_range = summed_frame_ranges[0]
    template_db_name = "{}-{}-{}.sqlite".format(frame_database_root, template_frame_range[0], template_frame_range[1])
    table_exceptions = []
    merge_summed_regions_prep(template_db_name, frame_database_name, exceptions=table_exceptions)
    for summed_frame_range in summed_frame_ranges:
        source_db_name = "{}-{}-{}.sqlite".format(frame_database_root, summed_frame_range[0], summed_frame_range[1])
        merge_summed_regions(source_db_name, frame_database_name, exceptions=[])

    recombine_frames_stop_time = time.time()
    processing_times.append(("frame-based recombine", recombine_frames_stop_time-recombine_frames_start_time))

    if not continue_processing(this_step='recombine_frame_databases', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

# retrieve the summed frame rate
source_conn = sqlite3.connect(frame_database_name)
df = pd.read_sql_query("select value from summing_info", source_conn)
source_conn.close()
if len(df) > 0:
    entry = json.loads(df.loc[0].value)
    frames_per_second = float(entry["frames_per_second"])
    print("Frames per second is {}".format(frames_per_second))
else:
    print("Error - could not find the frame rate from the summing_info table and it's needed in sebsequent steps. Exiting.")
    sys.exit(1)

###############################
# OPERATION: feature_detect_ms1
###############################
if process_this_step(this_step='feature_detect_ms1', first_step=args.operation):
    print("Starting the \'feature_detect_ms1\' step")
    feature_detect_start_time = time.time()

    print("detecting features...")
    run_process("python -u ./otf-peak-detect/feature-detect-ms1.py -db '{}' -fps {} -mnf {}".format(feature_database_name, frames_per_second, args.minimum_number_of_frames))

    feature_detect_stop_time = time.time()
    processing_times.append(("feature detect ms1", feature_detect_stop_time-feature_detect_start_time))

    if not continue_processing(this_step='feature_detect_ms1', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

# find out how many features were detected
source_conn = sqlite3.connect(feature_database_name)
feature_info_df = pd.read_sql_query("select value from feature_info where item='features found'", source_conn)
number_of_features = int(feature_info_df.values[0][0])
source_conn.close()

# split the feature range into batches
batch_splits = np.array_split(range(1,number_of_features+1), args.feature_tasks)
feature_ranges = []
for s in batch_splits:
    if len(s) > 0:
        feature_ranges.append((s[0],s[len(s)-1]))

#
# from here, split the combined features database into feature range databases
#

###########################################
# OPERATION: feature_region_ms2_peak_detect
###########################################
if process_this_step(this_step='feature_region_ms2_peak_detect', first_step=args.operation):
    print("Starting the \'feature_region_ms2_peak_detect\' step")
    ms2_peak_detect_start_time = time.time()

    # build the process lists
    feature_region_ms2_sum_peak_processes = []
    for feature_range in feature_ranges:
        destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        feature_region_ms2_sum_peak_processes.append("python -u ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect.py -cdb '{}' -ddb '{}' -ms1ce {} -fl {} -fu {} -ml {} -mu {} -bs 20 -fts {} -fso {} -mzsf {}".format(converted_database_name, destination_db_name, args.ms1_collision_energy, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper, args.frames_to_sum, args.frame_summing_offset, args.ms2_mz_scaling_factor))

    run_process("python -u ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect-prep.py -cdb '{}'".format(converted_database_name))
    print("detecting ms2 peaks in the feature region...")
    pool.map(run_process, feature_region_ms2_sum_peak_processes)
    ms2_peak_detect_stop_time = time.time()
    processing_times.append(("feature region ms2 peak detect", ms2_peak_detect_stop_time-ms2_peak_detect_start_time))

    if not continue_processing(this_step='feature_region_ms2_peak_detect', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

###########################################
# OPERATION: feature_region_ms1_peak_detect
###########################################
if process_this_step(this_step='feature_region_ms1_peak_detect', first_step=args.operation):
    print("Starting the \'feature_region_ms1_peak_detect\' step")
    ms1_peak_detect_start_time = time.time()

    # build the process lists
    feature_region_ms1_sum_processes = []
    feature_region_ms1_peak_processes = []
    for feature_range in feature_ranges:
        destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        feature_region_ms1_sum_processes.append("python -u ./otf-peak-detect/feature-region-ms1-sum-frames.py -sdb '{}' -ddb '{}' -fl {} -fu {} -ml {} -mu {}".format(feature_database_name, destination_db_name, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))
        feature_region_ms1_peak_processes.append("python -u ./otf-peak-detect/feature-region-ms1-peak-detect.py -sdb '{}' -ddb '{}' -fl {} -fu {} -ml {} -mu {}".format(feature_database_name, destination_db_name, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))

    print("summing ms1 frames, detecting peaks in the feature region...")
    run_process("python -u ./otf-peak-detect/feature-region-ms1-sum-frames-prep.py -sdb '{}'".format(feature_database_name))
    pool.map(run_process, feature_region_ms1_sum_processes)
    pool.map(run_process, feature_region_ms1_peak_processes)
    ms1_peak_detect_stop_time = time.time()
    processing_times.append(("feature region ms1 peak detect", ms1_peak_detect_stop_time-ms1_peak_detect_start_time))

    if not continue_processing(this_step='feature_region_ms1_peak_detect', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

######################################
# OPERATION: match_precursor_ms2_peaks
######################################
if process_this_step(this_step='match_precursor_ms2_peaks', first_step=args.operation):
    print("Starting the \'match_precursor_ms2_peaks\' step")
    match_precursor_ms2_peaks_start_time = time.time()

    # determine the drift offset between ms1 and ms2
    match_precursor_ms2_peaks_processes = []
    for feature_range in feature_ranges:
        destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        match_precursor_ms2_peaks_processes.append("python -u ./otf-peak-detect/match-precursor-ms2-peaks.py -db '{}' -fdb '{}' -fl {} -fu {} -fps {}".format(destination_db_name, feature_database_name, feature_range[0], feature_range[1], frames_per_second))

    print("matching precursor ms2 peaks...")
    pool.map(run_process, match_precursor_ms2_peaks_processes)
    match_precursor_ms2_peaks_stop_time = time.time()
    processing_times.append(("match precursor ms2 peaks", match_precursor_ms2_peaks_stop_time-match_precursor_ms2_peaks_start_time))

    if not continue_processing(this_step='match_precursor_ms2_peaks', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

############################
# OPERATION: correlate_peaks
############################
if process_this_step(this_step='correlate_peaks', first_step=args.operation):
    print("Starting the \'correlate_peaks\' step")
    peak_correlation_start_time = time.time()

    peak_correlation_processes = []
    for feature_range in feature_ranges:
        destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        peak_correlation_processes.append("python -u ./otf-peak-detect/correlate-ms2-peaks.py -db '{}' -cdb '{}' -fl {} -fu {} -st {} -rtt {}".format(destination_db_name, converted_database_name, feature_range[0], feature_range[1], args.correlation_scan_tolerance, args.correlation_rt_tolerance))
    
    print("correlating peaks...")
    run_process("python -u ./otf-peak-detect/correlate-ms2-peaks-prep.py -cdb '{}'".format(converted_database_name))
    pool.map(run_process, peak_correlation_processes)
    peak_correlation_stop_time = time.time()
    processing_times.append(("peak correlation", peak_correlation_stop_time-peak_correlation_start_time))

    if not continue_processing(this_step='correlate_peaks', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

###################################
# OPERATION: deconvolve_ms2_spectra
###################################
if process_this_step(this_step='deconvolve_ms2_spectra', first_step=args.operation):
    print("Starting the \'deconvolve_ms2_spectra\' step")
    deconvolve_ms2_spectra_processes = []
    for feature_range in feature_ranges:
        destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        deconvolve_ms2_spectra_processes.append("python -u ./otf-peak-detect/deconvolve-ms2-spectra.py -fdb '{}' -frdb '{}' -bfn {} -dbd {} -fps {} -mnp {}".format(feature_database_name, destination_db_name, args.database_base_name, args.data_directory, frames_per_second, args.maximum_number_of_peaks_per_feature))

    # deconvolve the ms2 spectra with Hardklor
    deconvolve_ms2_spectra_start_time = time.time()
    print("deconvolving ms2 spectra...")
    pool.map(run_process, deconvolve_ms2_spectra_processes)
    deconvolve_ms2_spectra_stop_time = time.time()
    processing_times.append(("deconvolve ms2 spectra", deconvolve_ms2_spectra_stop_time-deconvolve_ms2_spectra_start_time))

    if not continue_processing(this_step='deconvolve_ms2_spectra', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

##############################
# OPERATION: create_search_mgf
##############################
if process_this_step(this_step='create_search_mgf', first_step=args.operation):
    print("Starting the \'create_search_mgf\' step")
    create_search_mgf_processes = []
    for feature_range in feature_ranges:
        destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        base_mgf_name = "features-{}-{}".format(feature_range[0], feature_range[1])
        create_search_mgf_processes.append("python -u ./otf-peak-detect/create-search-mgf.py -fdb '{}' -bfn {} -dbd {}".format(destination_db_name, base_mgf_name, args.data_directory))

    # create search MGF
    create_search_mgf_start_time = time.time()
    print("creating the search MGF...")
    pool.map(run_process, create_search_mgf_processes)
    # now join them all together
    mgf_directory = "{}/mgf".format(args.data_directory)
    hk_directory = "{}/hk".format(args.data_directory)
    search_headers_directory = "{}/search-headers".format(args.data_directory)
    output_directory = "{}/search".format(mgf_directory)
    combined_mgf_filename = "{}/search.mgf".format(output_directory)
    if os.path.exists(combined_mgf_filename):
        os.remove(combined_mgf_filename)
    for feature_range in feature_ranges:
        base_mgf_name = "features-{}-{}".format(feature_range[0], feature_range[1])
        mgf_filename = "{}/{}-search.mgf".format(output_directory, base_mgf_name)
        run_process("cat {} >> {}".format(mgf_filename, combined_mgf_filename))

    create_search_mgf_stop_time = time.time()
    processing_times.append(("create search mgf", create_search_mgf_stop_time-create_search_mgf_stop_time))

    # # gather statistics
    # source_conn = sqlite3.connect(converted_database_name)
    # deconvoluted_ions_df = pd.read_sql_query("select max(ion_id) from deconvoluted_ions", source_conn)
    # number_of_deconvoluted_ions = int(deconvoluted_ions_df.values[0][0])
    # statistics.append(("number of deconvoluted ions", number_of_deconvoluted_ions))
    # source_conn.close()

    if not continue_processing(this_step='create_search_mgf', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

########################################
# OPERATION: recombine_feature_databases
########################################
if process_this_step(this_step='recombine_feature_databases', first_step=args.operation):
    print("Starting the \'recombine_feature_databases\' step")
    # recombine the feature range databases back into a combined database
    recombine_feature_databases_start_time = time.time()
    table_exceptions = []
    table_exceptions.append('summed_ms1_regions')
    table_exceptions.append('summed_ms1_regions_info')
    table_exceptions.append('summed_ms2_regions')
    table_exceptions.append('summed_ms2_regions_info')
    table_exceptions.append('ms1_feature_region_peaks')
    table_exceptions.append('ms2_feature_region_points')
    table_exceptions.append('ms2_peaks')
    table_exceptions.append('peak_correlation')
    template_feature_range = feature_ranges[0]
    template_db_name = "{}-{}-{}.sqlite".format(feature_database_root, template_feature_range[0], template_feature_range[1])
    merge_summed_regions_prep(template_db_name, feature_database_name, exceptions=table_exceptions)
    for feature_range in feature_ranges:
        source_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
        merge_summed_regions(source_db_name, feature_database_name, exceptions=table_exceptions)
    recombine_feature_databases_stop_time = time.time()
    processing_times.append(("feature recombine", recombine_feature_databases_stop_time-recombine_feature_databases_start_time))

    if not continue_processing(this_step='recombine_feature_databases', final_step=args.final_operation):
        print("Not continuing to the next step - exiting")
        sys.exit(0)

processing_stop_time = time.time()
processing_times.append(("total processing", processing_stop_time-processing_start_time))

# print statistics
print("")
print("processing times")
for t in processing_times:
    print("{}\t\t{:.1f} seconds\t\t{:.1f}%".format(t[0], t[1], t[1]/(processing_stop_time-processing_start_time)*100.))
# print("")
# print("data")
# for s in statistics:
#     print("{}\t\t{:.1f}".format(s[0], s[1]))

if args.shutdown_on_completion == True:
    run_process("sudo shutdown -P +5")
