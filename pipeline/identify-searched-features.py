import glob
import os
import shutil
import time
import argparse
import sys
import pandas as pd
import configparser
from configparser import ExtendedInterpolation
from os.path import expanduser
import json

# run the command in a shell
def run_process(process):
    print("Executing: {}".format(process))
    exit_status = os.WEXITSTATUS(os.system(process))
    if exit_status != 0:
        print('command had an exit status of {}'.format(exit_status))
    return exit_status

# calculate the monoisotopic mass    
def calculate_monoisotopic_mass_from_mz(monoisotopic_mz, charge):
    monoisotopic_mass = (monoisotopic_mz * charge) - (PROTON_MASS * charge)
    return monoisotopic_mass


################################
parser = argparse.ArgumentParser(description='Re-rank the collection of PSMs from Comet using the Percolator algorithm.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-ff','--fasta_file_name', type=str, default='./otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta', help='File name of the FASTA file.', required=False)
parser.add_argument('-pe','--protein_enzyme', type=str, default='trypsin', choices=['no_enzyme','elastase','pepsin','proteinasek','thermolysin','trypsinp','chymotrypsin','lys-n','lys-c','arg-c','asp-n','glu-c','trypsin'], help='Enzyme used for digestion. Passed to percolator.', required=False)
parser.add_argument('-pdm','--precursor_definition_method', type=str, choices=['pasef','3did'], help='The method used to define the precursor cuboids.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
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
    COMET_OUTPUT_DIR = "{}/comet-output-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
    PERCOLATOR_OUTPUT_DIR = "{}/percolator-output-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
    # the monoisotopic m/z to use
    monoisotopic_mz_column_name = 'monoisotopic_mz'
else:
    COMET_OUTPUT_DIR = "{}/comet-output-{}-recalibrated".format(EXPERIMENT_DIR, args.precursor_definition_method)
    PERCOLATOR_OUTPUT_DIR = "{}/percolator-output-{}-recalibrated".format(EXPERIMENT_DIR, args.precursor_definition_method)
    # the monoisotopic m/z to use
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
comet_output_file_list = glob.glob('{}/*.comet.pin'.format(COMET_OUTPUT_DIR))
if len(comet_output_file_list) == 0:
    print('found no comet input files in {}'.format(COMET_OUTPUT_DIR))
    sys.exit(1)
else:
    print('found {} comet input files in {}'.format(len(comet_output_file_list), COMET_OUTPUT_DIR))

comet_output_file_list_as_string = ' '.join(map(str, comet_output_file_list))
cmd = "{}/crux-4.0.Linux.x86_64/bin/crux percolator --overwrite T --subset-max-train 1000000 --klammer F --maxiter 10 --output-dir {} --picked-protein {} --protein T --protein-enzyme {} --search-input auto --verbosity 30 --fileroot {} {} > {} 2>&1".format(expanduser("~"), PERCOLATOR_OUTPUT_DIR, args.fasta_file_name, args.protein_enzyme, args.experiment_name, comet_output_file_list_as_string, PERCOLATOR_STDOUT_FILE_NAME)
exit_status = run_process(cmd)
if exit_status != 0:
    sys.exit(1)

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
psms_df = psms_df[(psms_df['peptide mass'] > 0) & (psms_df['percolator q-value'] <= MAXIMUM_Q_VALUE)]

# add the run names
percolator_df = pd.merge(psms_df, mapping_df, how='left', left_on=['file_idx'], right_on=['file_idx'])
del psms_df
del mapping_df

# load the detected features
FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, args.precursor_definition_method)
df_l = []
if not args.recalibration_mode:
    files_l = glob.glob('{}/exp-{}-run-*-features-*-dedup.feather'.format(FEATURES_DIR, args.experiment_name))
else:
    files_l = glob.glob('{}/exp-{}-run-*-features-*-recalibrated.feather'.format(FEATURES_DIR, args.experiment_name))

print('loading the detected features and merging them with the identifications')
for f in files_l:
    df = pd.merge(pd.read_feather(f), percolator_df, how='inner', left_on=['run_name','feature_id'], right_on=['run_name','feature_id'])
    df_l.append(df)
identifications_df = pd.concat(df_l, axis=0, sort=False, ignore_index=True)
del df_l[:]

# add the mass of cysteine carbamidomethylation to the theoretical peptide mass from percolator, for the fixed modification of carbamidomethyl
print('calculating mass error for identifications')
identifications_df['observed_monoisotopic_mass'] = calculate_monoisotopic_mass_from_mz(identifications_df[monoisotopic_mz_column_name], identifications_df.charge)
identifications_df['theoretical_peptide_mass'] = identifications_df['peptide mass'] + (identifications_df.sequence.str.count('C') * ADD_C_CYSTEINE_DA)

# now we can calculate the difference between the feature's monoisotopic mass and the theoretical peptide mass that is calculated from the 
# sequence's molecular formula and its modifications
identifications_df['mass_accuracy_ppm'] = (identifications_df['observed_monoisotopic_mass'] - identifications_df['theoretical_peptide_mass']) / identifications_df['theoretical_peptide_mass'] * 10**6
identifications_df['mass_error'] = identifications_df['observed_monoisotopic_mass'] - identifications_df['theoretical_peptide_mass']

# count how many unique peptides were identified
print('counting unique peptides')
sequences_l = []
for group_name,group_df in identifications_df.groupby(['sequence','charge'], as_index=False):
    if group_df['percolator q-value'].min() <= MAXIMUM_Q_VALUE:
        sequences_l.append({'sequence_charge':group_name, 'feature_ids':group_df.feature_id.tolist()})
sequences_df = pd.DataFrame(sequences_l)
print('there were {} unique peptides identified with q-value less than {}'.format(len(sequences_df), MAXIMUM_Q_VALUE))

# set up the output directory
IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, args.precursor_definition_method)
if not os.path.exists(IDENTIFICATIONS_DIR):
    os.makedirs(IDENTIFICATIONS_DIR)

# write out the identifications
if not args.recalibration_mode:
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.feather'.format(IDENTIFICATIONS_DIR, args.experiment_name, args.precursor_definition_method)
else:
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.feather'.format(IDENTIFICATIONS_DIR, args.experiment_name, args.precursor_definition_method)
print("writing {} identifications to {}".format(len(identifications_df), IDENTIFICATIONS_FILE))
identifications_df.reset_index(drop=True, inplace=True)
identifications_df.to_feather(IDENTIFICATIONS_FILE, compression_level=None, chunksize=500)

# write the metadata
info.append(('total_running_time',round(time.time()-start_run,1)))
info.append(('processor',parser.prog))
info.append(('processed', time.ctime()))
with open(IDENTIFICATIONS_FILE.replace('.feather','-metadata.json'), 'w') as handle:
    json.dump(info, handle)

# finish up
stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
