from __future__ import print_function
import os
import multiprocessing as mp
from multiprocessing import Pool
import sqlite3
import pandas as pd
import argparse

def run_process(process):
    os.system(process)

#
# python ./otf-peak-detect/generate-search-mgf-from-instrument-db.py -dbd /Volumes/Samsung_T5/databases/ -idb /Volumes/Samsung_T5/instrument/Hela_20A_20R_500_1_01_398.d/ -dbn Hela_20A_20R_500 -smgf /Volumes/Samsung_T5/mgf/Hela_20A_20R_500-search.mgf -cems1 7 -cems2 27
#

# Process the command line arguments
parser = argparse.ArgumentParser(description='Generates the search MGF from the instrument database.')
parser.add_argument('-dbd','--database_directory_name', type=str, help='The directory for the databases.', required=True)
parser.add_argument('-idb','--instrument_database_name', type=str, help='The name of the instrument database.', required=True)
parser.add_argument('-dbn','--database_base_name', type=str, help='The base name of the destination databases.', required=True)
parser.add_argument('-smgf','--search_mgf_name', type=str, help='File name of the generated search MGF.', required=True)
parser.add_argument('-fts','--frames_to_sum', type=int, default=150, help='The number of MS1 source frames to sum.', required=False)
parser.add_argument('-fso','--frame_summing_offset', type=int, default=25, help='The number of MS1 source frames to shift for each summation.', required=False)
parser.add_argument('-cems1','--ms1_collision_energy', type=int, help='Collision energy for ms1, in eV.', required=True)
parser.add_argument('-cems2','--ms2_collision_energy', type=int, help='Collision energy for ms2, in eV.', required=True)
parser.add_argument('-mpc','--minimum_peak_correlation', type=float, help='Minimum peak correlation', required=True)
parser.add_argument('-op','--operation', type=str, default='all', help='The operation to perform.', required=False)
parser.add_argument('-nf','--number_of_frames', type=int, help='The number of frames to convert.', required=False)
parser.add_argument('-ml','--mz_lower', type=float, help='Lower feature m/z to process.', required=True)
parser.add_argument('-mu','--mz_upper', type=float, help='Upper feature m/z to process.', required=True)
args = parser.parse_args()

converted_db_name = "{}/{}.sqlite".format(args.database_directory_name, args.database_base_name)
frame_database_name = "{}/{}-frames".format(args.database_directory_name, args.database_base_name)
feature_database_name = "{}/{}-features".format(args.database_directory_name, args.database_base_name)

# find out about the compute environment
number_of_batches = number_of_cores = mp.cpu_count()

# convert the database
convert_db_processes = []
convert_db_processes.append("python ./otf-peak-detect/convert-db.py -sdb {} -ddb {} -nf {}".format(args.instrument_database_name, converted_db_name, args.number_of_frames))

# Set up the processing pool
pool = Pool()
if (args.operation == 'all') or (args.operation == 'convert_db'):
    pool.map(run_process, convert_db_processes)

# Find the complete set of ms1 frame ids to be processed
source_conn = sqlite3.connect(converted_db_name)
frame_ids_df = pd.read_sql_query("select frame_id from frame_properties where collision_energy={} order by frame_id ASC;".format(args.ms1_collision_energy), source_conn)
frame_ids = tuple(frame_ids_df.values[:,0])
number_of_frames = 1 + int(((len(frame_ids) - args.frames_to_sum) / args.frame_summing_offset))
source_conn.close()

# Work out how many batches the available cores will support
batch_size = int(number_of_frames / number_of_batches)
if (batch_size * number_of_cores) < number_of_frames:
    number_of_batches += 1

print("number of frames {}, batch size {}, number of batches {}".format(number_of_frames, batch_size, number_of_batches))

frame_ranges = []
for batch_number in range(number_of_batches):
    first_frame_id = (batch_number * batch_size) + 1
    last_frame_id = first_frame_id + batch_size - 1
    if last_frame_id > number_of_frames:
        last_frame_id = number_of_frames
    frame_ranges.append((first_frame_id, last_frame_id))

# prepare to process the ms1 frames
prep_sum_frame_ms1_processes = []
prep_sum_frame_ms1_processes.append("python ./otf-peak-detect/sum-frames-ms1-prep.py -sdb {}".format(converted_db_name))

# process the ms1 frames
sum_frame_ms1_processes = []
peak_detect_ms1_processes = []
cluster_detect_ms1_processes = []
for frame_range in frame_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(frame_database_name, frame_range[0], frame_range[1])
    sum_frame_ms1_processes.append("python ./otf-peak-detect/sum-frames-ms1.py -sdb {} -ddb {} -ce {} -fl {} -fu {}".format(converted_db_name, destination_db_name, args.ms1_collision_energy, frame_range[0], frame_range[1]))
    peak_detect_ms1_processes.append("python ./otf-peak-detect/peak-detect-ms1.py -db {} -fl {} -fu {}".format(destination_db_name, frame_range[0], frame_range[1]))
    cluster_detect_ms1_processes.append("python ./otf-peak-detect/cluster-detect-ms1.py -db {} -fl {} -fu {}".format(destination_db_name, frame_range[0], frame_range[1]))

# execute the pipeline
if (args.operation == 'all') or (args.operation == 'sum_frame_ms1'):
    pool.map(run_process, prep_sum_frame_ms1_processes)
    pool.map(run_process, sum_frame_ms1_processes)
if (args.operation == 'all') or (args.operation == 'peak_detect_ms1'):
    pool.map(run_process, peak_detect_ms1_processes)
if (args.operation == 'all') or (args.operation == 'cluster_detect_ms1'):
    pool.map(run_process, cluster_detect_ms1_processes)

#
# generate a SQL command file to dump all the frame databases into .sql files
#
dump_sql_command_file_name = "{}/{}-dump.sql".format(args.database_directory_name, args.database_base_name)
sqlFile = open(dump_sql_command_file_name, 'w+')
for frame_range in frame_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(frame_database_name, frame_range[0], frame_range[1])
    db_sql_name = "{}-{}-{}-dump.sql".format(frame_database_name, frame_range[0], frame_range[1])
    print(".open {}".format(destination_db_name), file=sqlFile)
    print(".mode insert", file=sqlFile)
    print(".output {}".format(db_sql_name), file=sqlFile)
    print(".dump", file=sqlFile)
    print(".output", file=sqlFile)
print(".quit", file=sqlFile)
sqlFile.close()

#
# generate a SQL command file to load the .sql files into a combined database
#
combine_sql_command_file_name = "{}/{}-combine.sql".format(args.database_directory_name, args.database_base_name)
sqlFile = open(combine_sql_command_file_name, 'w+')
feature_db_name = "{}.sqlite".format(feature_database_name)
print(".open --new {}".format(feature_db_name), file=sqlFile)
for frame_range in frame_ranges:
    db_sql_name = "{}-{}-{}-dump.sql".format(frame_database_name, frame_range[0], frame_range[1])
    print(".read {}".format(db_sql_name), file=sqlFile)
print(".quit", file=sqlFile)
sqlFile.close()

# combine the interim processing databases into a feature database
dump_databases_processes = []
dump_databases_processes.append("sqlite3 < {}".format(dump_sql_command_file_name))

combine_databases_processes = []
combine_databases_processes.append("sqlite3 < {}".format(combine_sql_command_file_name))

if (args.operation == 'all') or (args.operation == 'combine_frame_databases'):
    print("dumping the frame databases...")
    pool.map(run_process, dump_databases_processes)
    print("loading the frame databases into a combined database for feature-based processing...")
    pool.map(run_process, combine_databases_processes)

# detect features in the ms1 frames
if (args.operation == 'all') or (args.operation == 'feature_detect_ms1'):
    print("detecting features...")
    run_process("python ./otf-peak-detect/feature-detect-ms1.py -db {}".format(feature_db_name))

# find out how many features were detected
source_conn = sqlite3.connect(feature_db_name)
feature_info_df = pd.read_sql_query("select value from feature_info where item='features found'", source_conn)
number_of_features = int(feature_info_df.values[0][0])
source_conn.close()

# work out how many batches the available cores will support
number_of_batches = number_of_cores
batch_size = int(number_of_features / number_of_batches)
if (batch_size * number_of_cores) < number_of_features:
    number_of_batches += 1

print("number of features {}, batch size {}, number of batches {}".format(number_of_features, batch_size, number_of_batches))

# work out the feature ranges for each batch
feature_ranges = []
for batch_number in range(number_of_batches):
    first_feature_id = (batch_number * batch_size) + 1
    last_feature_id = first_feature_id + batch_size - 1
    if last_feature_id > number_of_features:
        last_feature_id = number_of_features
    feature_ranges.append((first_feature_id, last_feature_id))

#
# from here, split the combined features database into feature range databases
#

# detect ms2 peaks in the feature's region
feature_region_ms2_sum_peak_processes = []
for feature_range in feature_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(feature_database_name, feature_range[0], feature_range[1])
    feature_region_ms2_sum_peak_processes.append("python ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect.py -cdb {} -sdb {} -ddb {} -ms2ce {} -fl {} -fu {} -ml {} -mu {} -bs 20".format(converted_db_name, feature_db_name, destination_db_name, args.ms2_collision_energy, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))

if (args.operation == 'all') or (args.operation == 'feature_region_ms2_peak_detect'):
    run_process("python ./otf-peak-detect/feature-region-ms2-combined-sum-peak-detect-prep.py -cdb {} -sdb {}".format(converted_db_name, feature_db_name))
    print("detecting ms2 peaks in the feature region...")
    pool.map(run_process, feature_region_ms2_sum_peak_processes)

# re-detect ms1 peaks in the feature's region, and calculate ms2 peak correlation
feature_region_ms1_sum_processes = []
feature_region_ms1_peak_processes = []
peak_correlation_processes = []
for feature_range in feature_ranges:
    destination_db_name = "{}-{}-{}.sqlite".format(feature_database_name, feature_range[0], feature_range[1])
    feature_region_ms1_sum_processes.append("python ./otf-peak-detect/feature-region-ms1-sum-frames.py -sdb {} -ddb {} -fl {} -fu {} -ml {} -mu {}".format(feature_db_name, destination_db_name, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))
    feature_region_ms1_peak_processes.append("python ./otf-peak-detect/feature-region-ms1-peak-detect.py -sdb {} -ddb {} -fl {} -fu {} -ml {} -mu {}".format(feature_db_name, destination_db_name, feature_range[0], feature_range[1], args.mz_lower, args.mz_upper))
    peak_correlation_processes.append("python ./otf-peak-detect/correlate-ms2-peaks.py -db {} -fl {} -fu {}".format(destination_db_name, feature_range[0], feature_range[1]))

if (args.operation == 'all') or (args.operation == 'feature_region_ms1_peak_detect'):
    print("summing ms1 frames, detecting peaks in the feature region...")
    run_process("python ./otf-peak-detect/feature-region-ms1-sum-frames-prep.py -sdb {}".format(feature_db_name))
    pool.map(run_process, feature_region_ms1_sum_processes)
    pool.map(run_process, feature_region_ms1_peak_processes)

if (args.operation == 'all') or (args.operation == 'correlate_peaks'):
    print("correlating peaks...")
    pool.map(run_process, peak_correlation_processes)

def merge_summed_regions(source_db_name, destination_db_name):
    source_conn = sqlite3.connect(source_db_name)
    src_cur = source_conn.cursor()
    destination_conn = sqlite3.connect(destination_db_name)
    dst_cur = destination_conn.cursor()

    df = pd.read_sql_query("SELECT name,sql FROM sqlite_master WHERE type='table'", source_conn)
    for t_idx in range(0,len(df)):
        print("merging {}".format(df.loc[t_idx].name))
        table_df = pd.read_sql_query("SELECT * FROM {}".format(df.loc[t_idx].name), source_conn)
        table_df.to_sql(df.loc[t_idx].name, destination_conn, if_exists='append')

    source_conn.close()
    destination_conn.commit()
    destination_conn.close()

def merge_summed_regions_prep(source_db_name, destination_db_name):
    source_conn = sqlite3.connect(source_db_name)
    destination_conn = sqlite3.connect(destination_db_name)
    dst_cur = destination_conn.cursor()

    df = pd.read_sql_query("SELECT name,sql FROM sqlite_master WHERE type='table'", source_conn)
    for t_idx in range(0,len(df)):
        print("preparing {}".format(df.loc[t_idx].name))
        dst_cur.execute("drop table if exists {}".format(df.loc[t_idx].name))
        dst_cur.execute(df.loc[t_idx].sql)

    source_conn.close()
    destination_conn.commit()
    destination_conn.close()

if (args.operation == 'all') or (args.operation == 'create_search_mgf'):

    # recombine the feature range databases back into a combined database
    template_feature_range = feature_ranges[0]
    template_db_name = "{}-{}-{}.sqlite".format(feature_database_name, template_feature_range[0], template_feature_range[1])
    merge_summed_regions_prep(template_db_name, feature_db_name)
    for feature_range in feature_ranges:
        source_db_name = "{}-{}-{}.sqlite".format(feature_database_name, feature_range[0], feature_range[1])
        print("merging {} into {}".format(source_db_name, feature_db_name))
        merge_summed_regions(source_db_name, feature_db_name)

    # deconvolve the ms2 spectra with Hardklor
    print("deconvolving ms2 spectra...")
    run_process("python ./otf-peak-detect/deconvolve-ms2-spectra.py -fdb {} -srdb {} -bfn {} -mpc {}".format(feature_db_name, destination_db_name, args.database_base_name, args.minimum_peak_correlation))

    # create search MGF
    print("creating the search MGF...")
    run_process("python ./otf-peak-detect/create_search_mgf.py -fdb {} -srdb {} -bfn {} -mpc {}".format(feature_db_name, destination_db_name, args.database_base_name, args.minimum_peak_correlation))
