import glob
import os
import shutil
import time
import argparse
import sys
import pandas as pd
import pickle
import configparser
from configparser import ExtendedInterpolation

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
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
parser.add_argument('-ns','--use_unsaturated_points_for_mz', action='store_true', help='Use the mono m/z calculated with only non-saturated points.')
parser.add_argument('-recal','--recalibration_mode', action='store_true', help='Use the recalibrated Comet output.')
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

if not args.recalibration_mode:
    COMET_OUTPUT_DIR = "{}/comet-output-{}".format(EXPERIMENT_DIR, args.feature_detection_method)
    PERCOLATOR_OUTPUT_DIR = "{}/percolator-output-{}".format(EXPERIMENT_DIR, args.feature_detection_method)
    if args.use_unsaturated_points_for_mz:
        monoisotopic_mz_column_name = 'mono_mz_without_saturated_points'
    else:
        monoisotopic_mz_column_name = 'monoisotopic_mz'
else:
    COMET_OUTPUT_DIR = "{}/comet-output-{}-recalibrated".format(EXPERIMENT_DIR, args.feature_detection_method)
    PERCOLATOR_OUTPUT_DIR = "{}/percolator-output-{}-recalibrated".format(EXPERIMENT_DIR, args.feature_detection_method)
    monoisotopic_mz_column_name = 'recalibrated_monoisotopic_mz'

# check the comet directory
if not os.path.exists(COMET_OUTPUT_DIR):
    print("The comet output directory is required but does not exist: {}".format(COMET_OUTPUT_DIR))
    sys.exit(1)

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
PROTON_MASS = cfg.getfloat('common','PROTON_MASS')
ADD_C_CYSTEINE_DA = cfg.getfloat('common','ADD_C_CYSTEINE_DA')
MAXIMUM_Q_VALUE = cfg.getfloat('common','MAXIMUM_Q_VALUE')

# set up the output directory
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
mapping_df = pd.DataFrame(mapping, columns=['file_idx','run_name'])

# load the percolator output
PERCOLATOR_OUTPUT_FILE_NAME = "{}/{}.percolator.target.psms.txt".format(PERCOLATOR_OUTPUT_DIR, args.experiment_name)
psms_df = pd.read_csv(PERCOLATOR_OUTPUT_FILE_NAME, sep='\t')
psms_df.rename(columns={'scan': 'feature_id'}, inplace=True)
psms_df.drop(['charge'], axis=1, inplace=True)

# remove the poor quality identifications
psms_df = psms_df[psms_df['percolator q-value'] <= MAXIMUM_Q_VALUE]
psms_df = psms_df[psms_df['peptide mass'] > 0]

# add the run names
percolator_df = pd.merge(psms_df, mapping_df, how='left', left_on=['file_idx'], right_on=['file_idx'])

# load the features
FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, args.feature_detection_method)
df_l = []
if not args.recalibration_mode:
    files_l = glob.glob('{}/exp-{}-run-*-features-*-dedup.pkl'.format(FEATURES_DIR, args.experiment_name))
else:
    files_l = glob.glob('{}/exp-{}-run-*-features-*-recalibrated.pkl'.format(FEATURES_DIR, args.experiment_name))

for f in files_l:
    with open(f, 'rb') as handle:
        d = pickle.load(handle)
    df_l.append(d['features_df'])
features_df = pd.concat(df_l, axis=0, sort=False)

# merge the identifications with the features
identifications_df = pd.merge(features_df, percolator_df, how='left', left_on=['run_name','feature_id'], right_on=['run_name','feature_id'])

# remove any features that were not identified
identifications_df.dropna(subset=['sequence'], inplace=True)

# add the mass of cysteine carbamidomethylation to the theoretical peptide mass from percolator, for the fixed modification of carbamidomethyl
identifications_df['observed_monoisotopic_mass'] = (identifications_df[monoisotopic_mz_column_name] * identifications_df.charge) - (PROTON_MASS * identifications_df.charge)
identifications_df['theoretical_peptide_mass'] = identifications_df['peptide mass'] + (identifications_df.sequence.str.count('C') * ADD_C_CYSTEINE_DA)

# now we can calculate the difference between the feature's monoisotopic mass and the theoretical peptide mass that is calculated from the 
# sequence's molecular formula and its modifications
identifications_df['mass_accuracy_ppm'] = (identifications_df['observed_monoisotopic_mass'] - identifications_df['theoretical_peptide_mass']) / identifications_df['theoretical_peptide_mass'] * 10**6
identifications_df['mass_error'] = identifications_df['observed_monoisotopic_mass'] - identifications_df['theoretical_peptide_mass']

# count how many unique peptides were identified
sequences_l = []
for group_name,group_df in identifications_df.groupby(['sequence','charge'], as_index=False):
    sequences_l.append({'sequence_charge':group_name, 'feature_ids':group_df.feature_id.tolist()})
sequences_df = pd.DataFrame(sequences_l)
print('there were {} unique peptides identified'.format(len(sequences_df)))

# set up the output directory
IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, args.feature_detection_method)
if not os.path.exists(IDENTIFICATIONS_DIR):
    os.makedirs(IDENTIFICATIONS_DIR)

# write out the identifications
if not args.recalibration_mode:
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.pkl'.format(IDENTIFICATIONS_DIR, args.experiment_name, args.feature_detection_method)
else:
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.pkl'.format(IDENTIFICATIONS_DIR, args.experiment_name, args.feature_detection_method)
print("writing {} identifications to {}".format(len(identifications_df), IDENTIFICATIONS_FILE))
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
content_d = {'identifications_df':identifications_df, 'metadata':info}
with open(IDENTIFICATIONS_FILE, 'wb') as handle:
    pickle.dump(content_d, handle)

# finish up
stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
