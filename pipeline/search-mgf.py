import glob
import os
import shutil
import time
import argparse
import sys


# run the command in a shell
def run_process(process):
    print("Executing: {}".format(process))
    exit_status = os.WEXITSTATUS(os.system(process))
    if exit_status != 0:
        print('command had an exit status of {}'.format(exit_status))


###########################
parser = argparse.ArgumentParser(description='Search the MGF against a sequence database and produce a collection of PSMs.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-fdm','--feature_detection_method', type=str, choices=['pasef','3did'], help='Which feature detection method.', required=True)
parser.add_argument('-ff','--fasta_file_name', type=str, default='./otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta', help='File name of the FASTA file.', required=False)
parser.add_argument('-cp','--comet_parameter_file_name', type=str, default='./otf-peak-detect/comet/TimsTOF.params', help='File name of the Comet parameter file.', required=False)
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use the recalibrated MGF.')
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the MGF directory
MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, args.feature_detection_method)
if not os.path.exists(MGF_DIR):
    print("The MGF directory is required but doesn't exist: {}".format(MGF_DIR))
    sys.exit(1)

# check the MGF file
if not args.recalibration_mode:
    MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, args.experiment_name, args.run_name, args.feature_detection_method)
else:
    MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, args.experiment_name, args.run_name, args.feature_detection_method)

if not os.path.isfile(MGF_FILE):
    print("The MGF file is required but doesn't exist: {}".format(MGF_FILE))
    sys.exit(1)

# set up the Comet output directory
COMET_OUTPUT_DIR = "{}/comet-output-{}".format(EXPERIMENT_DIR, args.feature_detection_method)
if not os.path.exists(COMET_OUTPUT_DIR):
    os.makedirs(COMET_OUTPUT_DIR)

# run comet on it
cmd = "{}/crux-3.2.Linux.x86_64/bin/crux comet --parameter-file {} --output-dir {} --fileroot \"{}\" {} {}".format(os.getcwd(), args.comet_parameter_file_name, COMET_OUTPUT_DIR, args.run_name, MGF_FILE, args.fasta_file_name)
run_process(cmd)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
