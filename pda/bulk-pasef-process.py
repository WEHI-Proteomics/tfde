import glob
import os
import shutil
import time
import argparse
import ray

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

# This program sequentially executes pasef-process.py for each run in the experiment. A future optimisation would be to run them concurrently but I want it to run on a somewhat ordinary machine.

parser = argparse.ArgumentParser(description='Process all the runs in an experiment.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pda/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use the recalibrated ms1 features.')
parser.add_argument('-ssm','--small_set_mode', action='store_true', help='A small subset of the data for testing purposes.')
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the associations directory exists
ASSOCIATIONS_DIR = "{}/associations".format(EXPERIMENT_DIR)
if os.path.exists(ASSOCIATIONS_DIR):
    shutil.rmtree(ASSOCIATIONS_DIR)
os.makedirs(ASSOCIATIONS_DIR)
print("The associations directory was created: {}".format(ASSOCIATIONS_DIR))

# check the log directory exists
LOG_DIR = "{}/logs".format(EXPERIMENT_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if args.recalibration_mode:
    LOG_FILE_NAME = "{}/{}-recalibration-pasef-process.log".format(LOG_DIR, args.experiment_name)
else:
    LOG_FILE_NAME = "{}/{}-pasef-process.log".format(LOG_DIR, args.experiment_name)

# check the configuration file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# initialise Ray
print("Starting a Ray cluster for the subprocesses to join.")
r = ray.init(object_store_memory=20000000000,
             redis_max_memory=25000000000)
redis_address = r['redis_address']

# process all the runs - we'll use the raw data files as the source of run names
raw_files_l = glob.glob("{}/raw-databases/*.d".format(EXPERIMENT_DIR))
if args.small_set_mode:
    raw_files_l = [raw_files_l[0]]

for raw_file in raw_files_l:
    run_name = os.path.basename(raw_file).split('_Slot')[0]
    print("processing {}".format(run_name))

    if not args.recalibration_mode:
        cmd = "python -u ~/otf-peak-detect/pda/pasef-process.py -eb {} -en {} -rn {} -ini {} -os linux -rm join -ra {} > {} 2>&1".format(args.experiment_base_dir, args.experiment_name, run_name, args.ini_file, redis_address, LOG_FILE_NAME)
    else:
        cmd = "python -u ~/otf-peak-detect/pda/pasef-process.py -eb {} -en {} -rn {} -ini {} -os linux -rm join -ra {} -ao -recal > {} 2>&1".format(args.experiment_base_dir, args.experiment_name, run_name, args.ini_file, redis_address, LOG_FILE_NAME)
    run_process(cmd)

stop_run = time.time()
print("total running time (bulk-pasef-process): {} seconds".format(round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
