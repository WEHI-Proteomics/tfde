import glob
import os
import shutil
import time
import argparse

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

parser = argparse.ArgumentParser(description='Convert the raw databases to processing databases.')
parser.add_argument('-raw','--raw_database_dir', type=str, help='Path to the raw databases directory.', required=True)
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

# check the raw databases directory exists
RAW_DIR = args.raw_database_dir
if not os.path.exists(RAW_DIR):
    print("The raw databasese directory is required but doesn't exist: {}".format(RAW_DIR))
    sys.exit(1)

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_DIR)
    print("The experiment directory was created: {}".format(EXPERIMENT_DIR))

# check the converted databases directory exists
CONVERTED_DIR = "{}/converted-databases".format(EXPERIMENT_DIR)
if os.path.exists(CONVERTED_DIR):
    shutil.rmtree(CONVERTED_DIR)
os.makedirs(CONVERTED_DIR)
print("The converted databases directory was created: {}".format(CONVERTED_DIR))

# check the log directory exists
LOG_DIR = "{}/logs".format(EXPERIMENT_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print("The logs directory was created: {}".format(LOG_DIR))

start_run = time.time()

for file in glob.glob("{}/*.d".format(RAW_DIR)):
    db_name = os.path.basename(file)
    run_name = db_name.split('_Slot')[0]
    print("processing {}".format(run_name))
    cmd = "python -u ~/otf-peak-detect/original-pipeline/convert-instrument-db.py -sdb {} -ddb {}/{}-converted.sqlite -bs 2000 > {}/{}-convert.log 2>&1".format(file, CONVERTED_DIR, run_name, LOG_DIR, run_name)
    run_process(cmd)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
