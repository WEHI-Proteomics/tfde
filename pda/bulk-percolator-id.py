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
BASE_COMET_OUTPUT_DIR = '/home/ubuntu/{}/comet-output'.format(RUN_NAME)
PERCOLATOR_OUTPUT_DIR = '/home/ubuntu/{}/percolator-output'.format(RUN_NAME)

start_run = time.time()

# process all the Comet output files in the base directory
comet_output_file_list = glob.glob('{}/*.comet.target.pin'.format(BASE_COMET_OUTPUT_DIR))
comet_output_file_list_as_string = ' '.join(map(str, comet_output_file_list))

cmd = "./crux-3.2.Linux.x86_64/bin/crux percolator --overwrite T --subset-max-train 1000000 --klammer F --maxiter 10 --output-dir {} --picked-protein ./otf-peak-detect/fasta/uniprot-proteome-human-Ecoli.fasta --protein T --protein-enzyme trypsin --search-input auto --verbosity 30 --fileroot {} {}".format(PERCOLATOR_OUTPUT_DIR, RUN_NAME, comet_output_file_list_as_string)
run_process(cmd)

stop_run = time.time()
print("total running time (bulk-percolator-id): {} seconds".format(round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
