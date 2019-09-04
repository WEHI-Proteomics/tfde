import glob
import os
import shutil
import time
import argparse

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

parser = argparse.ArgumentParser(description='Find all the MGFs and run a Comet search on them.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use a tighter tolerance on the search.')
parser.add_argument('-shutdown','--shutdown', action='store_true', help='Shut down the machine when complete.')
args = parser.parse_args()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

if args.recalibration_mode:
    COMET_OUTPUT_DIR = "{}/recalibrated-comet-output".format(EXPERIMENT_DIR)
else:
    COMET_OUTPUT_DIR = "{}/comet-output".format(EXPERIMENT_DIR)
if os.path.exists(COMET_OUTPUT_DIR):
    shutil.rmtree(COMET_OUTPUT_DIR)
os.makedirs(COMET_OUTPUT_DIR)
print("The comet output directory was created: {}".format(COMET_OUTPUT_DIR))

if args.recalibration_mode:
    MGF_DIR = "{}/recalibrated-mgfs".format(EXPERIMENT_DIR)
else:
    MGF_DIR = "{}/mgfs".format(EXPERIMENT_DIR)
if not os.path.exists(MGF_DIR):
    print("The MGF directory is required but does not exist: {}".format(MGF_DIR))
    sys.exit(1)

if args.recalibration_mode:
    COMET_PARAMETER_FILE = './otf-peak-detect/comet/TimsTOF-recalibration.params'
else:
    COMET_PARAMETER_FILE = './otf-peak-detect/comet/TimsTOF.params'
if not os.path.exists(COMET_PARAMETER_FILE):
    print("The comet parameter file is required but does not exist: {}".format(COMET_PARAMETER_FILE))
    sys.exit(1)

start_run = time.time()

# process all the MGFs in the base directory
for file in glob.glob('{}/*.mgf'.format(MGF_DIR)):
    mgf_name = os.path.basename(file)
    run_name = mgf_name.split('-search')[0]
    print("processing {}".format(run_name))
    cmd = "./crux-3.2.Linux.x86_64/bin/crux comet --parameter-file {} --output-dir {} --fileroot \"{}\" {} ./otf-peak-detect/fasta/uniprot-proteome-human-Ecoli.fasta".format(COMET_PARAMETER_FILE, COMET_OUTPUT_DIR, run_name, file)
    run_process(cmd)

stop_run = time.time()
print("total running time (bulk-comet-search): {} seconds".format(round(stop_run-start_run,1)))

if args.shutdown:
    run_process("sudo shutdown -P +5") # shutdown the instance in 5 minutes from now
