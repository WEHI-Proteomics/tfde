import glob
import os
import shutil
import time
import argparse

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

parser = argparse.ArgumentParser(description='Manage the ms1 and ms2 processing, and generate the MGF.')
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use the recalibrated ms1 features.')
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

BASE_DIR = '/home/ubuntu/190719_Hela_Ecoli/190719_Hela_Ecoli_1to1'
RAW_DIR = '{}/raw'.format(BASE_DIR)
CONVERTED_DIR = '{}/converted'.format(BASE_DIR)
INI_FILE = '/home/ubuntu/otf-peak-detect/pda/pasef-process-short-gradient.ini'

start_run = time.time()

for raw_db_file in glob.glob("{}/*.d".format(RAW_DIR)):
    db_name = os.path.basename(raw_db_file).split('_Slot')[0]
    base_processing_dir = CONVERTED_DIR
    print("processing {}".format(db_name))
    if not args.recalibration_mode:
        cmd = "python -u ~/otf-peak-detect/pda/pasef-process.py -rdb {} -bpd {} -pn {} -ini {} -os linux > {}/{}-processing.log 2>&1".format(raw_db_file, base_processing_dir, db_name, INI_FILE, BASE_DIR, db_name)
    else:
        cmd = "python -u ~/otf-peak-detect/pda/pasef-process.py -rdb {} -bpd {} -pn {} -ini {} -os linux -recal > {}/{}-processing.log 2>&1".format(raw_db_file, base_processing_dir, db_name, INI_FILE, BASE_DIR, db_name)
    run_process(cmd)

stop_run = time.time()
print("total running time (bulk-pasef-process): {} seconds".format(round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
