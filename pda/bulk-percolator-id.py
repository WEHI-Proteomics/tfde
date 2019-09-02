import glob
import os
import shutil
import time
import argparse

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

parser = argparse.ArgumentParser(description='Run Percolator on the Comet files.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use the recalibration comet output files.')
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

if args.recalibration_mode:
    COMET_OUTPUT_DIR = "{}/comet-output-recalibrated".format(EXPERIMENT_DIR)
else:
    COMET_OUTPUT_DIR = "{}/comet-output".format(EXPERIMENT_DIR)
if not os.path.exists(COMET_OUTPUT_DIR):
    print("The comet output directory is required but does not exist: {}".format(COMET_OUTPUT_DIR))
    sys.exit(1)

if args.recalibration_mode:
    PERCOLATOR_OUTPUT_DIR = "{}/percolator-output-recalibrated".format(EXPERIMENT_DIR)
else:
    PERCOLATOR_OUTPUT_DIR = "{}/percolator-output".format(EXPERIMENT_DIR)
if os.path.exists(PERCOLATOR_OUTPUT_DIR):
    shutil.rmtree(PERCOLATOR_OUTPUT_DIR)
os.makedirs(PERCOLATOR_OUTPUT_DIR)
print("The percolator output directory was created: {}".format(PERCOLATOR_OUTPUT_DIR))

PERCOLATOR_STDOUT_FILE_NAME = "{}/percolator-stdout.log".format(PERCOLATOR_OUTPUT_DIR)

start_run = time.time()

# process all the Comet output files in the base directory
comet_output_file_list = glob.glob('{}/*.comet.target.pin'.format(COMET_OUTPUT_DIR))
comet_output_file_list_as_string = ' '.join(map(str, comet_output_file_list))

cmd = "./crux-3.2.Linux.x86_64/bin/crux percolator --overwrite T --subset-max-train 1000000 --klammer F --maxiter 10 --output-dir {} --picked-protein ./otf-peak-detect/fasta/uniprot-proteome-human-Ecoli.fasta --protein T --protein-enzyme trypsin --search-input auto --verbosity 30 --fileroot {} {} > {}".format(PERCOLATOR_OUTPUT_DIR, RUN_NAME, comet_output_file_list_as_string, PERCOLATOR_STDOUT_FILE_NAME)
run_process(cmd)

stop_run = time.time()
print("total running time (bulk-percolator-id): {} seconds".format(round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
