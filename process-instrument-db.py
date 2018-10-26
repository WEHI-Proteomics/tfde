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
import glob
import shutil

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
            print("merging {} from {} to {}".format(table_name, source_db_name, destination_db_name))

            row_count = int(pd.read_sql('SELECT COUNT(*) FROM {table_name}'.format(table_name=table_name), source_conn).values)
            chunksize = 5000000
            number_of_chunks = int(row_count / chunksize)

            for i in range(number_of_chunks + 1):
                print("\tmerging chunk {} of {}".format(i, number_of_chunks))
                query = 'SELECT * FROM {table_name} LIMIT {offset}, {chunksize}'.format(table_name=table_name, offset=i * chunksize, chunksize=chunksize)
                table_df = pd.read_sql_query(query, con=source_conn)
                table_df.to_sql(name=table_name, con=destination_conn, if_exists='append', index=False, chunksize=None)

            # drop the table in the source database
            print("dropping table {} from {}".format(table_name, source_db_name))
            src_cur.execute('DROP TABLE IF EXISTS {table_name}'.format(table_name=table_name))
        else:
            print("skipping merge of {}".format(table_name))

    # if we moved all the tables to the destination, delete the database file. Otherwise, vacuum it 
    # to minimise disk space.
    if len(exceptions) == 0:
        print("deleting {}".format(source_db_name))
        source_conn.close()
        os.remove(source_db_name)
    else:
        if args.vacuum_databases == True:
            print("vacuuming {}".format(source_db_name))
            src_cur.execute('VACUUM')
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

# returns true if there is an entry in each of the specified tables in each of the specified databases
def step_successful(step_name, databases, tables):
    expected_ok_count = len(databases) * len(tables)
    ok_count = 0
    for db in databases:
        db_conn = sqlite3.connect(db)
        tables_df = pd.read_sql_query("SELECT tbl_name FROM sqlite_master WHERE type='table'", db_conn)
        for tab in tables:
            if len(tables_df[tables_df.tbl_name == "{}".format(tab)]) == 1: # does the table exist
                df = pd.read_sql_query("select * from {}".format(tab), db_conn) # does it have an entry
                if len(df) > 0:
                    ok_count += 1
                else:
                    print("step {}: {} in {} does not have an entry".format(step_name, tab, db))
            else:
                print("step {}: {} does not have the table {}".format(step_name, db, tab))
    result = (ok_count == expected_ok_count)
    if result:
        print("step {} passed".format(step_name))
    else:
        print("step {} failed".format(step_name))
    return result

# return true if the specified step should be processed
def process_this_step(this_step, first_step):
    result = (processing_steps[this_step] >= processing_steps[first_step])
    return result

# return true if this isn't the last step
def continue_processing(this_step, final_step, databases=[], tables=[]):
    result = step_successful(this_step, databases, tables) and (processing_steps[this_step] < processing_steps[final_step])
    return result

def store_info(info, processing_times):
    processing_stop_time = time.time()
    info.append(("total processing", round(processing_stop_time-processing_start_time,1)))
    info.append(("processing times", json.dumps(processing_times)))
    print("step processing times: {}".format(json.dumps(processing_times)))
    # store it in the database
    info_entry_df = pd.DataFrame(info, columns=['item', 'value'])
    db_conn = sqlite3.connect(converted_database_name)
    info_entry_df.to_sql(name='processing_info', con=db_conn, if_exists='replace', index=False)
    db_conn.close()

def cleanup(info, processing_times, shutdown):
    store_info(info, processing_times)
    if shutdown == True:
        run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
    sys.exit(1)

# break up the list l into chunks of n items
# source: https://stackoverflow.com/a/1751478/1184799
def chunks(l, n):
    n = max(1, n)
    return list(l[i:i+n] for i in xrange(0, len(l), n))

#
# source activate py27
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
parser.add_argument('-cbs','--conversion_batch_size', type=int, help='The size of the frames to be written to the database during conversion.', required=False)
parser.add_argument('-cems1','--ms1_collision_energy', type=int, help='Collision energy for ms1, in eV.', required=True)
parser.add_argument('-op','--operation', type=str, default='all', help='The operation to perform.', required=False)
parser.add_argument('-fop','--final_operation', type=str, help='The final operation to perform.', required=False)
parser.add_argument('-sd','--shutdown_on_completion', action='store_true', help='Shut down the instance when complete.')
parser.add_argument('-nf','--number_of_frames', type=int, help='The number of frames to convert.', required=False)
parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=False)
parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=False)
parser.add_argument('-sl','--scan_lower', type=int, help='Lower scan to process.', required=False)
parser.add_argument('-su','--scan_upper', type=int, help='Upper scan to process.', required=False)
parser.add_argument('-fl','--frame_lower', type=int, help='The lower summed frame number to process.', required=False)
parser.add_argument('-fu','--frame_upper', type=int, help='The upper summed frame number to process.', required=False)
parser.add_argument('-mfl','--minimum_feature_length_secs', type=int, default=1, help='Minimum feature length in seconds for it to be valid.', required=False)
parser.add_argument('-ns','--number_of_seconds_each_side', type=int, default=5, help='Number of seconds to look for related clusters either side of the maximum cluster.', required=False)
parser.add_argument('-mzsf','--ms2_mz_scaling_factor', type=float, default=1000.0, help='Scaling factor to convert m/z range to integers in ms2.', required=False)
parser.add_argument('-frts','--frame_tasks_per_worker', type=int, default=5, help='Number of frame tasks assigned to each worker in the pool.', required=False)
parser.add_argument('-fets','--feature_tasks_per_worker', type=int, default=5, help='Number of feature tasks assigned to each worker in the pool.', required=False)
parser.add_argument('-mnp','--maximum_number_of_peaks_per_feature', type=int, default=500, help='The maximum number of peaks per feature.', required=False)
parser.add_argument('-es','--elution_start_sec', type=int, help='Only process frames from this time in sec.', required=False)
parser.add_argument('-ee','--elution_end_sec', type=int, help='Only process frames up to this time in sec.', required=False)
parser.add_argument('-vacdb','--vacuum_databases', action='store_true', help='Vacuum databases to reduce disk space.')
parser.add_argument('-nrtd','--negative_rt_delta_tolerance', type=float, default=-0.25, help='The negative RT delta tolerance.', required=False)
parser.add_argument('-prtd','--positive_rt_delta_tolerance', type=float, default=0.25, help='The positive RT delta tolerance.', required=False)
parser.add_argument('-nsd','--negative_scan_delta_tolerance', type=float, default=-4.0, help='The negative scan delta tolerance.', required=False)
parser.add_argument('-psd','--positive_scan_delta_tolerance', type=float, default=4.0, help='The positive scan delta tolerance.', required=False)
parser.add_argument('-naw','--noise_assessment_width_secs', type=float, default=1.0, help='Length of time in seconds to average the noise level for feature detection.', required=False)
parser.add_argument('-nao','--noise_assessment_offset_secs', type=float, default=1.0, help='Offset in seconds from the end of the feature frames for feature detection.', required=False)
parser.add_argument('-pasef','--pasef_mode', action='store_true', help='Analyse matches between features and isolation windows for a PASEF acquisition.')
args = parser.parse_args()

processing_times = []
processing_start_time = time.time()

info = []

steps = []
steps.append('convert_instrument_db')
steps.append('cluster_detect_ms1')
steps.append('recombine_frame_databases')
steps.append('feature_detect_ms1')
steps.append('feature_region_ms1_peak_detect')
steps.append('resolve_feature_list')
steps.append('feature_region_ms2_peak_detect')
steps.append('match_precursor_ms2_peaks')
steps.append('correlate_peaks')
steps.append('deconvolve_ms2_spectra')
steps.append('create_search_mgf')

processing_steps = {j:i for i,j in enumerate(steps)}

if (args.operation == 'all'):
    args.operation = steps[0]
    args.final_operation = steps[-1]

if (args.operation is None):
    args.operation = steps[0]

if (args.final_operation is None):
    args.final_operation = steps[-1]

converted_database_name = "{}/{}.sqlite".format(args.data_directory, args.database_base_name)
frame_database_root = "{}/{}-frames".format(args.data_directory, args.database_base_name)  # used to split the data into frame-based sections
frame_database_name = converted_database_name  # combined the frame-based sections back into the converted database
feature_database_root = "{}/{}-features".format(args.data_directory, args.database_base_name)  # used to split the data into feature-based sections
feature_database_name = converted_database_name  # combined the feature-based sections back into the converted database

# find out about the compute environment
number_of_cores = mp.cpu_count()

# store some metadata
info.append(("converted_database_name", converted_database_name))
info.append(("frame_database_root", frame_database_root))
info.append(("frame_database_name", frame_database_name))
info.append(("feature_database_root", feature_database_root))
info.append(("feature_database_name", feature_database_name))
info.append(("number_of_cores", number_of_cores))

# Set up the processing pool
pool = Pool() # the number of worker processes in the pool will be the number returned by cpu_count()

##################################
# OPERATION: convert_instrument_db
##################################
step_name = 'convert_instrument_db'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    step_start_time = time.time()

    # clean up the data directory from any previous runs
    if os.path.exists(args.data_directory):
        shutil.rmtree(args.data_directory)
    os.makedirs(args.data_directory)

    # make sure the instrument database exists
    if not os.path.exists(args.instrument_database_name):
        print("Error - the instrument database directory does not exist. Exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

    # check whether an elution period was specified
    if args.elution_start_sec is None:
        args.elution_start_sec = -1
    if args.elution_end_sec is None:
        args.elution_end_sec = -1

    run_process("python -u ./otf-peak-detect/convert-instrument-db.py -sdb '{}' -ddb '{}' -es {} -ee {} -bs {}".format(args.instrument_database_name, converted_database_name, args.elution_start_sec, args.elution_end_sec, args.conversion_batch_size))

    # gather statistics
    source_conn = sqlite3.connect(converted_database_name)
    frame_info_df = pd.read_sql_query("select max(frame_id) from frames", source_conn)
    number_of_converted_frames = int(frame_info_df.values[0][0])
    info.append(("number of converted frames", number_of_converted_frames))
    source_conn.close()

    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=[converted_database_name], tables=['convert_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

if not os.path.exists(converted_database_name):
    print("Error - the converted database does not exist. Exiting.")
    cleanup(info, processing_times, args.shutdown_on_completion)

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
        cleanup(info, processing_times, args.shutdown_on_completion)

if args.mz_upper is None:
    source_conn = sqlite3.connect(converted_database_name)
    df = pd.read_sql_query("select value from convert_info where item = \'mz_upper\'", source_conn)
    source_conn.close()
    if len(df) > 0:
        args.mz_upper = float(df.loc[0].value)
        print("mz_upper set to {} from the data".format(args.mz_upper))
    else:
        print("Error - could not find mz_upper from the convert_info table and it's needed in sebsequent steps. Exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

# Determine the scan range if it's not specified
if args.scan_lower is None:
    args.scan_lower = 1

if args.scan_upper is None:
    source_conn = sqlite3.connect(converted_database_name)
    df = pd.read_sql_query("select value from convert_info where item = \'num_scans\'", source_conn)
    source_conn.close()
    if len(df) > 0:
        args.scan_upper = int(df.loc[0].value)
        print("scan_upper set to {} from the data".format(args.scan_upper))
    else:
        print("Error - could not find scan_upper from the convert_info table and it's needed in sebsequent steps. Exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

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

# Store the arguments as metadata in the database for later reference
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

# split the summed frame range into batches of tasks for each worker in the pool
batch_splits = chunks(range(args.frame_lower,args.frame_upper+1), args.frame_tasks_per_worker)
summed_frame_ranges = []
for s in batch_splits:
    if len(s) > 0:
        summed_frame_ranges.append((s[0],s[len(s)-1]))
print("summed frame ranges: {}".format(summed_frame_ranges))

# define the frame database filenames
frame_databases = []
for summed_frame_range in summed_frame_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(frame_database_root, summed_frame_range[0], summed_frame_range[1])
    frame_databases.append(destination_db_name)

# create the dataframe to hold the frame batch info
frame_batch_df = pd.DataFrame()
frame_batch_df['lower'] = [item[0] for item in summed_frame_ranges]
frame_batch_df['upper'] = [item[1] for item in summed_frame_ranges]
frame_batch_df['db'] = frame_databases
info.append(("frame_batch_info", frame_batch_df.to_json(orient='records')))

###############################
# OPERATION: cluster_detect_ms1
###############################
step_name = 'cluster_detect_ms1'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    step_start_time = time.time()

    # clean up from previous runs
    for f in glob.glob("{}*".format(frame_database_root)):
        os.remove(f)
    for f in glob.glob("{}*".format(feature_database_root)):
        os.remove(f)

    # build the process lists
    sum_frame_ms1_processes = []
    peak_detect_ms1_processes = []
    cluster_detect_ms1_processes = []
    for idx in range(len(frame_batch_df)):
        destination_db_name = frame_batch_df.iloc[idx].db
        frame_lower = frame_batch_df.iloc[idx].lower
        frame_upper = frame_batch_df.iloc[idx].upper
        sum_frame_ms1_processes.append("python -u ./otf-peak-detect/sum-frames-ms1.py -sdb '{}' -ddb '{}' -ce {} -fl {} -fu {} -fts {} -fso {} -sl {} -su {}".format(converted_database_name, destination_db_name, args.ms1_collision_energy, frame_lower, frame_upper, args.frames_to_sum, args.frame_summing_offset, args.scan_lower, args.scan_upper))
        peak_detect_ms1_processes.append("python -u ./otf-peak-detect/peak-detect-ms1.py -db '{}' -fl {} -fu {} -sl {} -su {}".format(destination_db_name, frame_lower, frame_upper, args.scan_lower, args.scan_upper))
        cluster_detect_ms1_processes.append("python -u ./otf-peak-detect/cluster-detect-ms1.py -db '{}' -fl {} -fu {}".format(destination_db_name, frame_lower, frame_upper))

    run_process("python -u ./otf-peak-detect/sum-frames-ms1-prep.py -sdb '{}'".format(converted_database_name))
    pool.map(run_process, sum_frame_ms1_processes)
    pool.map(run_process, peak_detect_ms1_processes)
    pool.map(run_process, cluster_detect_ms1_processes)

    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=frame_batch_df.db.tolist(), tables=['summing_info','peak_detect_info','cluster_detect_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

######################################
# OPERATION: recombine_frame_databases
######################################
step_name = 'recombine_frame_databases'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    step_start_time = time.time()

    # recombine the frame range databases back into a combined database
    template_db_name = frame_batch_df.iloc[0].db
    table_exceptions = []
    merge_summed_regions_prep(template_db_name, frame_database_name, exceptions=table_exceptions)
    for idx in range(len(frame_batch_df)):
        source_db_name = frame_batch_df.iloc[idx].db
        merge_summed_regions(source_db_name, frame_database_name, exceptions=[])

    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

###############################
# OPERATION: feature_detect_ms1
###############################
step_name = 'feature_detect_ms1'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    step_start_time = time.time()

    print("detecting features...")
    run_process("python -u ./otf-peak-detect/feature-detect-ms1.py -db '{}' -mfl {} -ns {} -naw {} -nao {}".format(feature_database_name, args.minimum_feature_length_secs, args.number_of_seconds_each_side, args.noise_assessment_width_secs, args.noise_assessment_offset_secs))

    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=[feature_database_name], tables=['feature_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

# find out how many features were detected
source_conn = sqlite3.connect(feature_database_name)
feature_info_df = pd.read_sql_query("select value from feature_info where item='features found'", source_conn)
number_of_features = int(feature_info_df.values[0][0])
print("Number of features detected: {}".format(number_of_features))
source_conn.close()

# split the feature range into batches of tasks for each worker in the pool
batch_splits = chunks(range(1,number_of_features+1), args.feature_tasks_per_worker)
feature_ranges = []
for s in batch_splits:
    if len(s) > 0:
        feature_ranges.append((s[0],s[len(s)-1]))
print("Feature ranges: {}".format(feature_ranges))

# define the feature database filenames
feature_databases = []
for feature_range in feature_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(feature_database_root, feature_range[0], feature_range[1])
    feature_databases.append(destination_db_name)

# create the dataframe to hold the frame batch info
feature_batch_df = pd.DataFrame()
feature_batch_df['lower'] = [item[0] for item in feature_ranges]
feature_batch_df['upper'] = [item[1] for item in feature_ranges]
feature_batch_df['db'] = feature_databases
info.append(("feature_batch_info", feature_batch_df.to_json(orient='records')))

#
# from here, split the combined features database into feature range databases
#

###########################################
# OPERATION: feature_region_ms1_peak_detect
###########################################
step_name = 'feature_region_ms1_peak_detect'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    step_start_time = time.time()

    # clean up from previous runs
    for f in glob.glob("{}*".format(feature_database_root)):
        os.remove(f)

    # build the process lists
    feature_region_ms1_sum_processes = []
    feature_region_ms1_peak_processes = []
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        feature_lower = feature_batch_df.iloc[idx].lower
        feature_upper = feature_batch_df.iloc[idx].upper
        feature_region_ms1_sum_processes.append("python -u ./otf-peak-detect/feature-region-ms1-sum-frames.py -sdb '{}' -ddb '{}' -fl {} -fu {}".format(feature_database_name, destination_db_name, feature_lower, feature_upper))
        feature_region_ms1_peak_processes.append("python -u ./otf-peak-detect/feature-region-ms1-peak-detect.py -sdb '{}' -ddb '{}' -fl {} -fu {}".format(feature_database_name, destination_db_name, feature_lower, feature_upper))

    print("summing ms1 frames, detecting peaks in the feature region...")
    run_process("python -u ./otf-peak-detect/feature-region-ms1-sum-frames-prep.py -sdb '{}' -fdbr '{}'".format(feature_database_name, feature_database_root))
    pool.map(run_process, feature_region_ms1_sum_processes)
    pool.map(run_process, feature_region_ms1_peak_processes)
    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=feature_batch_df.db.tolist(), tables=['summed_ms1_regions_info','ms1_feature_region_peak_detect_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

#################################
# OPERATION: resolve_feature_list
#################################
step_name = 'resolve_feature_list'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    step_start_time = time.time()

    # build the process lists
    resolve_feature_list_processes = []
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        feature_lower = feature_batch_df.iloc[idx].lower
        feature_upper = feature_batch_df.iloc[idx].upper
        resolve_feature_list_processes.append("python -u ./otf-peak-detect/resolve-feature-list.py -fdb '{}' -frdb '{}' -fl {} -fu {}".format(feature_database_name, destination_db_name, feature_lower, feature_upper))

    print("resolving the feature list...")
    run_process("python -u ./otf-peak-detect/resolve-feature-list-prep.py -cdb '{}'".format(converted_database_name))
    pool.map(run_process, resolve_feature_list_processes)

    # write out the feature lists as a global CSV
    csv_file_name = "{}/{}-feature-list.csv".format(args.data_directory, args.database_base_name)
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        db_conn = sqlite3.connect(destination_db_name)
        print("writing feature list from {} to {}".format(destination_db_name, csv_file_name))
        df = pd.read_sql_query("select * from feature_list", db_conn)
        if idx == 0:
            df.to_csv(csv_file_name, mode='w', sep=',', index=False, header=True)
        else:
            df.to_csv(csv_file_name, mode='a', sep=',', index=False, header=False)
        db_conn.close()

    # write out the feature isotopes as a global CSV
    csv_file_name = "{}/{}-feature-isotopes.csv".format(args.data_directory, args.database_base_name)
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        db_conn = sqlite3.connect(destination_db_name)
        print("writing feature isotopes from {} to {}".format(destination_db_name, csv_file_name))
        df = pd.read_sql_query("select * from feature_isotopes", db_conn)
        if idx == 0:
            df.to_csv(csv_file_name, mode='w', sep=',', index=False, header=True)
        else:
            df.to_csv(csv_file_name, mode='a', sep=',', index=False, header=False)
        db_conn.close()

    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=feature_batch_df.db.tolist(), tables=['resolve_feature_list_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

###########################################
# OPERATION: feature_region_ms2_peak_detect
###########################################
step_name = 'feature_region_ms2_peak_detect'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    step_start_time = time.time()

    # build the process lists
    feature_region_ms2_sum_peak_processes = []
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        feature_lower = feature_batch_df.iloc[idx].lower
        feature_upper = feature_batch_df.iloc[idx].upper
        if args.pasef_mode:
            feature_region_ms2_sum_peak_processes.append("python -u ./otf-peak-detect/feature-region-ms2-pasef-sum-peak-detect.py -cdb '{}' -ddb '{}' -ms1ce {} -fl {} -fu {} -bs 20 -fts {} -fso {} -mzsf {}".format(converted_database_name, destination_db_name, args.ms1_collision_energy, feature_lower, feature_upper, args.frames_to_sum, args.frame_summing_offset, args.ms2_mz_scaling_factor))
        else:
            feature_region_ms2_sum_peak_processes.append("python -u ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect.py -cdb '{}' -ddb '{}' -ms1ce {} -fl {} -fu {} -bs 20 -fts {} -fso {} -mzsf {}".format(converted_database_name, destination_db_name, args.ms1_collision_energy, feature_lower, feature_upper, args.frames_to_sum, args.frame_summing_offset, args.ms2_mz_scaling_factor))

    run_process("python -u ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect-prep.py -cdb '{}'".format(converted_database_name))
    print("detecting ms2 peaks in the feature region...")
    pool.map(run_process, feature_region_ms2_sum_peak_processes)

    # write out the feature matches with isolation windows as a global CSV
    csv_file_name = "{}/{}-feature-isolation-matches.csv".format(args.data_directory, args.database_base_name)
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        db_conn = sqlite3.connect(destination_db_name)
        print("writing feature isolation matches from {} to {}".format(destination_db_name, csv_file_name))
        df = pd.read_sql_query("select * from feature_isolation_matches", db_conn)
        if idx == 0:
            df.to_csv(csv_file_name, mode='w', sep=',', index=False, header=True)
        else:
            df.to_csv(csv_file_name, mode='a', sep=',', index=False, header=False)
        db_conn.close()

    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=feature_batch_df.db.tolist(), tables=['summed_ms2_regions_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

######################################
# OPERATION: match_precursor_ms2_peaks
######################################
if not args.pasef_mode:  # not needed in PASEF mode
    step_name = 'match_precursor_ms2_peaks'
    if process_this_step(this_step=step_name, first_step=args.operation):
        print("Starting the \'{}\' step".format(step_name))
        step_start_time = time.time()

        # determine the drift offset between ms1 and ms2
        match_precursor_ms2_peaks_processes = []
        for idx in range(len(feature_batch_df)):
            destination_db_name = feature_batch_df.iloc[idx].db
            feature_lower = feature_batch_df.iloc[idx].lower
            feature_upper = feature_batch_df.iloc[idx].upper
            match_precursor_ms2_peaks_processes.append("python -u ./otf-peak-detect/match-precursor-ms2-peaks.py -ddb '{}' -sdb '{}' -fl {} -fu {}".format(destination_db_name, feature_database_name, feature_lower, feature_upper))

        print("matching precursor ms2 peaks...")
        pool.map(run_process, match_precursor_ms2_peaks_processes)
        step_stop_time = time.time()
        processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

        if continue_processing(this_step=step_name, final_step=args.final_operation, databases=feature_batch_df.db.tolist(), tables=['precursor_ms2_peak_matches_info']):
            print("Completed {}. Continuing to the next step.".format(step_name))
        else:
            print("Not continuing to the next step - exiting.")
            cleanup(info, processing_times, args.shutdown_on_completion)

############################
# OPERATION: correlate_peaks
############################
if not args.pasef_mode:  # not needed in PASEF mode
    step_name = 'correlate_peaks'
    if process_this_step(this_step=step_name, first_step=args.operation):
        print("Starting the \'{}\' step".format(step_name))
        step_start_time = time.time()

        peak_correlation_processes = []
        for idx in range(len(feature_batch_df)):
            destination_db_name = feature_batch_df.iloc[idx].db
            feature_lower = feature_batch_df.iloc[idx].lower
            feature_upper = feature_batch_df.iloc[idx].upper
            peak_correlation_processes.append("python -u ./otf-peak-detect/correlate-ms2-peaks.py -db '{}' -cdb '{}' -fl {} -fu {}".format(destination_db_name, converted_database_name, feature_lower, feature_upper))
        
        print("correlating peaks...")
        pool.map(run_process, peak_correlation_processes)
        step_stop_time = time.time()
        processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

        if continue_processing(this_step=step_name, final_step=args.final_operation, databases=feature_batch_df.db.tolist(), tables=['peak_correlation_info']):
            print("Completed {}. Continuing to the next step.".format(step_name))
        else:
            print("Not continuing to the next step - exiting.")
            cleanup(info, processing_times, args.shutdown_on_completion)

###################################
# OPERATION: deconvolve_ms2_spectra
###################################
step_name = 'deconvolve_ms2_spectra'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    deconvolve_ms2_spectra_processes = []
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        if args.pasef_mode:
            deconvolve_ms2_spectra_processes.append("python -u ./otf-peak-detect/deconvolve-ms2-spectra-pasef.py -fdb '{}' -frdb '{}' -dbd {}".format(feature_database_name, destination_db_name, args.data_directory))
        else:
            deconvolve_ms2_spectra_processes.append("python -u ./otf-peak-detect/deconvolve-ms2-spectra.py -fdb '{}' -frdb '{}' -dbd {} -mnp {} -nrtd {} -prtd {} -nsd {} -psd {}".format(feature_database_name, destination_db_name, args.data_directory, args.maximum_number_of_peaks_per_feature, args.negative_rt_delta_tolerance, args.positive_rt_delta_tolerance, args.negative_scan_delta_tolerance, args.positive_scan_delta_tolerance))

    # deconvolve the ms2 spectra with Hardklor
    step_start_time = time.time()
    print("deconvolving ms2 spectra...")
    run_process("python -u ./otf-peak-detect/deconvolve-ms2-spectra-prep.py -dbd '{}'".format(args.data_directory))
    pool.map(run_process, deconvolve_ms2_spectra_processes)
    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=feature_batch_df.db.tolist(), tables=['deconvolve_ms2_spectra_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)

##############################
# OPERATION: create_search_mgf
##############################
step_name = 'create_search_mgf'
if process_this_step(this_step=step_name, first_step=args.operation):
    print("Starting the \'{}\' step".format(step_name))
    create_search_mgf_processes = []
    output_directory = "{}/mgf/search".format(args.data_directory)
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        feature_lower = feature_batch_df.iloc[idx].lower
        feature_upper = feature_batch_df.iloc[idx].upper
        base_mgf_name = "features-{}-{}".format(feature_lower, feature_upper)
        if args.pasef_mode:
            create_search_mgf_processes.append("python -u ./otf-peak-detect/create-search-mgf-pasef.py -fdb '{}' -bfn {} -dbd {} -od {}".format(destination_db_name, base_mgf_name, args.data_directory, output_directory))
        else:
            create_search_mgf_processes.append("python -u ./otf-peak-detect/create-search-mgf.py -fdb '{}' -bfn {} -dbd {} -od {}".format(destination_db_name, base_mgf_name, args.data_directory, output_directory))

    # create search MGF
    step_start_time = time.time()

    # make sure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)    

    print("creating the search MGF...")
    pool.map(run_process, create_search_mgf_processes)
    # now join them all together
    mgf_directory = "{}/mgf".format(args.data_directory)
    hk_directory = "{}/hk".format(args.data_directory)
    search_headers_directory = "{}/search-headers".format(args.data_directory)
    output_directory = "{}/search".format(mgf_directory)
    combined_mgf_filename = "{}/{}-search.mgf".format(output_directory, args.database_base_name)
    # delete the search MGF if it already exists
    if os.path.exists(combined_mgf_filename):
        os.remove(combined_mgf_filename)
    for idx in range(len(feature_batch_df)):
        feature_lower = feature_batch_df.iloc[idx].lower
        feature_upper = feature_batch_df.iloc[idx].upper
        base_mgf_name = "features-{}-{}".format(feature_lower, feature_upper)
        mgf_filename = "{}/{}-search.mgf".format(output_directory, base_mgf_name)
        run_process("cat {} >> {}".format(mgf_filename, combined_mgf_filename))

    # write out the deconvoluted_ions table as a global CSV
    csv_file_name = "{}/{}-deconvoluted-ions.csv".format(args.data_directory, args.database_base_name)
    for idx in range(len(feature_batch_df)):
        destination_db_name = feature_batch_df.iloc[idx].db
        db_conn = sqlite3.connect(destination_db_name)
        df = pd.read_sql_query("SELECT * FROM sqlite_master WHERE type='table' and tbl_name='deconvoluted_ions'", db_conn)
        if len(df) > 0:
            print("writing deconvoluted ions from {} to {}".format(destination_db_name, csv_file_name))
            df = pd.read_sql_query("select * from deconvoluted_ions", db_conn)
            if idx == 0:
                df.to_csv(csv_file_name, mode='w', sep=',', index=False, header=True)
            else:
                df.to_csv(csv_file_name, mode='a', sep=',', index=False, header=False)
        db_conn.close()

    step_stop_time = time.time()
    processing_times.append((step_name, round(step_stop_time-step_start_time,1)))

    if continue_processing(this_step=step_name, final_step=args.final_operation, databases=feature_batch_df.db.tolist(), tables=['search_mgf_info']):
        print("Completed {}. Continuing to the next step.".format(step_name))
    else:
        print("Not continuing to the next step - exiting.")
        cleanup(info, processing_times, args.shutdown_on_completion)
