import glob
import os
import shutil
import time
import argparse
import sys
import configparser
from configparser import ExtendedInterpolation
from os.path import expanduser


# run the command in a shell
def run_process(process):
    print("Executing: {}".format(process))
    exit_status = os.WEXITSTATUS(os.system(process))
    if exit_status != 0:
        print('command had an exit status of {}'.format(exit_status))
    return exit_status


###########################
parser = argparse.ArgumentParser(description='Search the MGF against a sequence database and produce a collection of PSMs.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-pdm','--precursor_definition_method', type=str, choices=['pasef','3did'], help='The method used to define the precursor cuboids.', required=True)
parser.add_argument('-ff','--fasta_file_name', type=str, default='./otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta', help='File name of the FASTA file.', required=False)
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use the recalibrated MGF.')
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
args = parser.parse_args()

# Print the arguments for the log
info = {}
for arg in vars(args):
    info[arg] = getattr(args, arg)
print(info)

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
INITIAL_SEARCH_PARAMETERS = cfg.get('comet','INITIAL_SEARCH_PARAMETERS')
RECALIBRATED_SEARCH_PARAMETERS = cfg.get('comet','RECALIBRATED_SEARCH_PARAMETERS')

# check the MGF directory
MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
if not os.path.exists(MGF_DIR):
    print("The MGF directory is required but doesn't exist: {}".format(MGF_DIR))
    sys.exit(1)

# check the MGF file
if not args.recalibration_mode:
    MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, args.experiment_name, args.run_name, args.precursor_definition_method)
    COMET_OUTPUT_DIR = "{}/comet-output-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
    COMET_PARAMS = INITIAL_SEARCH_PARAMETERS
else:
    MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, args.experiment_name, args.run_name, args.precursor_definition_method)
    COMET_OUTPUT_DIR = "{}/comet-output-{}-recalibrated".format(EXPERIMENT_DIR, args.precursor_definition_method)
    COMET_PARAMS = RECALIBRATED_SEARCH_PARAMETERS

file_directory = os.path.abspath(os.path.dirname(__file__))
COMET_PARAM_FILE = '{}/../comet/{}'.format(file_directory, COMET_PARAMS)

# check the MGF file
if not os.path.isfile(MGF_FILE):
    print("The MGF file is required but doesn't exist: {}".format(MGF_FILE))
    sys.exit(1)

# set up the Comet output directory
if not os.path.exists(COMET_OUTPUT_DIR):
    os.makedirs(COMET_OUTPUT_DIR)

# run comet on it
cmd = "{}/crux-4.0.Linux.x86_64/bin/crux comet --parameter-file {} --output-dir {} --fileroot \"{}\" {} {}".format(expanduser("~"), COMET_PARAM_FILE, COMET_OUTPUT_DIR, args.run_name, MGF_FILE, args.fasta_file_name)
exit_status = run_process(cmd)
if exit_status != 0:
    sys.exit(1)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
