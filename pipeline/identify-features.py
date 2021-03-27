import glob
import os
import shutil
import time
import argparse
import sys
import pandas as pd

# run the command in a shell
def run_process(process):
    print("Executing: {}".format(process))
    exit_status = os.WEXITSTATUS(os.system(process))
    if exit_status != 0:
        print('command had an exit status of {}'.format(exit_status))


################################
parser = argparse.ArgumentParser(description='Re-rank the collection of PSMs from Comet using the Percolator algorithm.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-ff','--fasta_file_name', type=str, default='./otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta', help='File name of the FASTA file.', required=False)
parser.add_argument('-pe','--protein_enzyme', type=str, default='trypsin', choices=['no_enzyme','elastase','pepsin','proteinasek','thermolysin','trypsinp','chymotrypsin','lys-n','lys-c','arg-c','asp-n','glu-c','trypsin'], help='Enzyme used for digestion. Passed to percolator.', required=False)
parser.add_argument('-fdm','--feature_detection_method', type=str, choices=['pasef','3did'], help='Which feature detection method.', required=True)
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

# check the comet directory
COMET_OUTPUT_DIR = "{}/comet-output-{}".format(EXPERIMENT_DIR, args.feature_detection_method)
if not os.path.exists(COMET_OUTPUT_DIR):
    print("The comet output directory is required but does not exist: {}".format(COMET_OUTPUT_DIR))
    sys.exit(1)

# set up the output directory
PERCOLATOR_OUTPUT_DIR = "{}/percolator-output-{}".format(EXPERIMENT_DIR, args.feature_detection_method)
if os.path.exists(PERCOLATOR_OUTPUT_DIR):
    shutil.rmtree(PERCOLATOR_OUTPUT_DIR)
os.makedirs(PERCOLATOR_OUTPUT_DIR)

# process all the Comet output files in the base directory
PERCOLATOR_STDOUT_FILE_NAME = "{}/percolator-stdout.log".format(PERCOLATOR_OUTPUT_DIR)
comet_output_file_list = glob.glob('{}/*.comet.target.pin'.format(COMET_OUTPUT_DIR))
comet_output_file_list_as_string = ' '.join(map(str, comet_output_file_list))
cmd = "{}/crux-3.2.Linux.i686/bin/crux percolator --overwrite T --subset-max-train 1000000 --klammer F --maxiter 10 --output-dir {} --picked-protein {} --protein T --protein-enzyme {} --search-input auto --verbosity 30 --fileroot {} {} > {} 2>&1".format(os.getcwd(), PERCOLATOR_OUTPUT_DIR, args.fasta_file_name, args.protein_enzyme, args.experiment_name, comet_output_file_list_as_string, PERCOLATOR_STDOUT_FILE_NAME)
run_process(cmd)

# determine the mapping between the percolator index and the run file name - this is only available by parsing percolator's stdout redirected to a text file.
print("Determining the mapping between percolator index and each run")
mapping = []
with open(PERCOLATOR_STDOUT_FILE_NAME) as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('INFO: Assigning index'):
            splits = line.split(' ')
            percolator_index = int(splits[3])
            comet_filename = splits[5]
            run_name = comet_filename.split('/')[-1].split('.')[0]  # e.g. 190719_Hela_Ecoli_1to3_06
            mapping.append((percolator_index, run_name))
mapping_df = pd.DataFrame(mapping, columns=['percolator_idx','run_name'])

# # load the percolator output
# PERCOLATOR_OUTPUT_FILE_NAME = "{}/{}.percolator.target.psms.txt".format(PERCOLATOR_OUTPUT_DIR, args.experiment_name)
# print("Loading the percolator output from {}".format(PERCOLATOR_OUTPUT_FILE_NAME))
# psms_df = pd.read_csv(PERCOLATOR_OUTPUT_FILE_NAME, sep='\t')

# # merge the run names with the percolator output
# percolator_df = pd.merge(psms_df, mapping_df, how='left', left_on=['file_idx'], right_on=['percolator_idx'])

# # merge the features with the percolator identifications
# identifications_df = pd.merge(features_df, percolator_df, how='left', left_on=['run_name','feature_id'], right_on=['run_name','scans'])

# # write out the identifications
# print("writing {} identifications to {}".format(len(dedup_df), FEATURES_DEDUP_FILE))
# info.append(('dedup_running_time',round(time.time()-dedup_start_run,1)))
# content_d = {'features_df':dedup_df, 'metadata':info}
# with open(FEATURES_DEDUP_FILE, 'wb') as handle:
#     pickle.dump(content_d, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
