import glob
import os
import shutil

BASE_DIR = '/home/ubuntu/190719_Hela_Ecoli/190719_Hela_Ecoli_1to1'
RAW_DIR = '{}/raw'.format(BASE_DIR)
CONVERTED_DIR = '{}/converted'.format(BASE_DIR)

if os.path.exists(CONVERTED_DIR):
    shutil.rmtree(CONVERTED_DIR)
os.makedirs(CONVERTED_DIR)

# make the converted directories
for file in glob.glob("{}/*.d".format(RAW_DIR)):
    db_name = os.path.basename(file)
    convert_dir_name = db_name.split('_Slot')[0]
    os.makedirs('{}/{}'.format(CONVERTED_DIR, convert_dir_name))
    cmd = "python -u ~/otf-peak-detect/original-pipeline/convert-instrument-db.py -sdb {}/{} -ddb {}/{}/{}-converted.sqlite -bs 2000 > {}/{}.log 2>&1".format(RAW_DIR, db_name, CONVERTED_DIR, convert_dir_name, convert_dir_name, BASE_DIR, convert_dir_name)
    os.system(cmd)
