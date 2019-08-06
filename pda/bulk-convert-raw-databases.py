import glob
import os
import shutil
import time

BASE_DIR = '/home/ubuntu/190719_Hela_Ecoli/190719_Hela_Ecoli_1to1'
RAW_DIR = '{}/raw'.format(BASE_DIR)
CONVERTED_DIR = '{}/converted'.format(BASE_DIR)

start_run = time.time()

if os.path.exists(CONVERTED_DIR):
    shutil.rmtree(CONVERTED_DIR)
os.makedirs(CONVERTED_DIR)

# make the converted directories
for file in glob.glob("{}/*.d".format(RAW_DIR)):
    db_name = os.path.basename(file)
    convert_dir_name = db_name.split('_Slot')[0]
    print("processing {}".format(convert_dir_name))
    os.makedirs('{}/{}'.format(CONVERTED_DIR, convert_dir_name))
    cmd = "python -u ~/otf-peak-detect/original-pipeline/convert-instrument-db.py -sdb {}/{} -ddb {}/{}/{}-converted.sqlite -bs 2000 > {}/{}-convert.log 2>&1".format(RAW_DIR, db_name, CONVERTED_DIR, convert_dir_name, convert_dir_name, BASE_DIR, convert_dir_name)
    os.system(cmd)

stop_run = time.time()
print("total running time (bulk-convert-raw-databases): {} seconds".format(round(stop_run-start_run,1)))
