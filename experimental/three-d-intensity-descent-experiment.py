import glob
import os
import shutil
import time
import argparse
import sys
import multiprocessing as mp
from multiprocessing import Pool

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)


parser = argparse.ArgumentParser(description='Process .')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./open-path/pda/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-nfb','--number_of_frames_in_batch', type=int, default=1000, help='The number of frames in a batch.', required=False)
parser.add_argument('-nw','--number_of_workers', type=int, default=10, help='Number of workers available to the pool.', required=False)
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the raw database directory exists
RAW_DIR = '{}/raw-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(RAW_DIR):
    print("The raw database directory is required but doesn't exist: {}".format(RAW_DIR))
    sys.exit(1)

# set up the converted databases directory
CONVERTED_DB_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if os.path.exists(CONVERTED_DB_DIR):
    shutil.rmtree(CONVERTED_DB_DIR)
os.makedirs(CONVERTED_DB_DIR)

# check the log directory exists
LOG_DIR = "{}/logs".format(EXPERIMENT_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# process all the runs
raw_database_names_l = glob.glob("{}/*.d".format(RAW_DIR))

# set up the processing pool
pool = Pool(processes=args.number_of_workers)

# convert the raw databases
processes = []
current_directory = os.path.abspath(os.path.dirname(__file__))
for raw_database_name in raw_database_names_l:
    run_name = os.path.basename(raw_database_name).split('_Slot')[0]

    LOG_FILE_NAME = "{}/{}-convert-raw-database.log".format(LOG_DIR, run_name)
    if os.path.isfile(LOG_FILE_NAME):
        os.remove(LOG_FILE_NAME)

    cmd = "python -u {}/convert-raw-database.py -eb {} -en {} -rn {} -rdd {} -nfb {} -ini {} > {} 2>&1".format(current_directory, args.experiment_base_dir, args.experiment_name, run_name, raw_database_name, args.number_of_frames_in_batch, args.ini_file, LOG_FILE_NAME)
    processes.append(cmd)

# execute the processes and wait for them to finish
pool.map(run_process, processes)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
