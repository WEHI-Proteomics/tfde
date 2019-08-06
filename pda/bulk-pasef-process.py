import glob
import os
import shutil

BASE_DIR = '/home/ubuntu/190719_Hela_Ecoli/190719_Hela_Ecoli_1to1'
RAW_DIR = '{}/raw'.format(BASE_DIR)
CONVERTED_DIR = '{}/converted'.format(BASE_DIR)
INI_NAME = '/home/ubuntu/otf-peak-detect/pda/pasef-process-short-gradient.ini'

# process the converted directories
for file in glob.glob("{}/*.sqlite".format(CONVERTED_DIR)):
    db_name = os.path.basename(file)
    convert_dir_name = db_name.split('.sqlite')[0]
    print("processing {}".format(convert_dir_name))
    os.makedirs('{}/{}'.format(CONVERTED_DIR, convert_dir_name))
    cmd = "python -u ~/otf-peak-detect/pda/pasef-process.py -rdb {}/{} -bpd {} -pn {} -ini {} -os linux > {}/{}.log 2>&1".format(RAW_DIR, db_name, CONVERTED_DIR, convert_dir_name, INI_NAME, db_name, convert_dir_name)
    print(cmd)
    # os.system(cmd)
