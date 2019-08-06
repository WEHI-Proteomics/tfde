import glob
import os
import shutil
import time

BASE_DIR = '/home/ubuntu/190719_Hela_Ecoli/190719_Hela_Ecoli_1to1'
RAW_DIR = '{}/raw'.format(BASE_DIR)
CONVERTED_DIR = '{}/converted'.format(BASE_DIR)
INI_FILE = '/home/ubuntu/otf-peak-detect/pda/pasef-process-short-gradient.ini'

start_run = time.time()

for raw_db_file in glob.glob("{}/*.d".format(RAW_DIR)):
    db_name = os.path.basename(raw_db_file).split('_Slot')[0]
    base_processing_dir = CONVERTED_DIR
    print("processing {}".format(db_name))
    cmd = "python -u ~/otf-peak-detect/pda/pasef-process.py -rdb {} -bpd {} -pn {} -ini {} -os linux > {}/{}-processing.log 2>&1".format(raw_db_file, base_processing_dir, db_name, INI_FILE, BASE_DIR, db_name)
    # print(cmd)
    os.system(cmd)

stop_run = time.time()
print("total running time (bulk-pasef-process): {} seconds".format(round(stop_run-start_run,1)))
