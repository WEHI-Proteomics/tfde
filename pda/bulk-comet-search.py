import glob
import os
import shutil
import time
import argparse

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

parser = argparse.ArgumentParser(description='Convert the raw databases to processing databases.')
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

RUN_NAME = "190719_Hela_Ecoli"
BASE_DIR = '/home/ubuntu/{}'.format(RUN_NAME)
COMET_OUTPUT_DIR = "{}/comet-output".format(BASE_DIR)

start_run = time.time()

# process all the MGFs in the base directory
for file in glob.glob('{}/**/*.mgf'.format(BASE_DIR), recursive=True):
    mgf_name = os.path.basename(file)
    db_name = mgf_name.split('-search')[0]
    print("processing {}".format(db_name))
    cmd = "./crux-3.2.Linux.x86_64/bin/crux comet --parameter-file ./otf-peak-detect/comet/TimsTOF.params --output-dir {} --fileroot \"{}\" {} ./otf-peak-detect/fasta/uniprot-proteome-human-Ecoli.fasta".format(COMET_OUTPUT_DIR, db_name, file)
    run_process(cmd)

stop_run = time.time()
print("total running time (bulk-comet-search): {} seconds".format(round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
