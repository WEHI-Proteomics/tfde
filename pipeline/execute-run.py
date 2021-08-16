from doit import get_var
from doit.action import CmdAction
from os.path import expanduser
import datetime
import time
import shutil
import os
import sys

# This is the set of tasks to take a raw instrument database and create a list of peptides

ini_file = '{}/pasef-process-short-gradient.ini'.format(os.path.dirname(os.path.realpath(__file__)))
fasta_file_name = '{}/../fasta/Human_Yeast_Ecoli.fasta'.format(os.path.dirname(os.path.realpath(__file__)))


# the function get_var() gets the named argument from the command line as a string; if it's not present it uses the specified default
config = {
    'experiment_base_dir': get_var('eb', '/media/big-ssd/experiments'),
    'experiment_name': get_var('en', None),
    'run_name': get_var('rn', None),
    'fasta_file_name': get_var('ff', fasta_file_name),
    'ini_file': get_var('ini', ini_file),
    'precursor_definition_method': get_var('pdm', 'pasef'),
    'rt_lower': get_var('rl', 1650),
    'rt_upper': get_var('ru', 2200),
    'correct_for_saturation': get_var('cs', 'true'),
    'filter_by_mass_defect': get_var('fmdw', 'true'),
    'proportion_of_cores_to_use': get_var('pc', 0.8)
    }

print('execution arguments: {}'.format(config))

# correct for saturation
if config['correct_for_saturation'] == 'true':
    config['cs_flag'] = '-cs'
else:
    config['cs_flag'] = ''

# filter by mass defect windows
if config['filter_by_mass_defect'] == 'true':
    config['fmdw_flag'] = '-fmdw'
else:
    config['fmdw_flag'] = ''

EXPERIMENT_DIR = "{}/{}".format(config['experiment_base_dir'], config['experiment_name'])

start_run = time.time()


####################
# feature extraction
####################
def task_define_precursor_cuboids():
    # input
    RAW_DATABASE_NAME = "{}/raw-databases/{}.d/analysis.tdf".format(EXPERIMENT_DIR, config['run_name'])
    # command
    cmd = 'python -u define-precursor-cuboids-pasef.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rl {rl} -ru {ru} -rm cluster'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], INI_FILE=config['ini_file'], rl=int(config['rt_lower']), ru=int(config['rt_upper']))
    # output
    CUBOIDS_DIR = '{}/precursor-cuboids-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-{}.pkl'.format(CUBOIDS_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])

    return {
        'file_dep': [RAW_DATABASE_NAME],
        'actions': [cmd],
        'targets': [CUBOIDS_FILE],
        'clean': ['rm -rf {}'.format(CUBOIDS_DIR)],
        'verbosity': 2
    }

def task_detect_features():
    # input
    CUBOIDS_DIR = "{}/precursor-cuboids-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-{}.hdf'.format(CUBOIDS_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])
    # command
    cmd = 'python -u detect-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rm cluster -pc {proportion_of_cores_to_use} -rl {rl} -ru {ru} {cs} {fmdw}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], INI_FILE=config['ini_file'], proportion_of_cores_to_use=config['proportion_of_cores_to_use'], rl=int(config['rt_lower']), ru=int(config['rt_upper']), cs=config['cs_flag'], fmdw=config['fmdw_flag'])
    # output
    FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    FEATURES_FILE = '{}/exp-{}-run-{}-features-{}.hdf'.format(FEATURES_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])

    return {
        'file_dep': [CUBOIDS_FILE],
        'actions': [cmd],
        'targets': [FEATURES_FILE],
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

def task_remove_duplicate_features():
    # input
    FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    FEATURES_FILE = '{}/exp-{}-run-{}-features-{}.hdf'.format(FEATURES_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])
    # command
    cmd = 'python -u remove-duplicate-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'])
    # output
    FEATURES_DEDUP_FILE = '{}/exp-{}-run-{}-features-{}-dedup.hdf'.format(FEATURES_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])
    # pass-through command (no de-dup)
    # cmd = 'cp {} {}'.format(FEATURES_FILE, FEATURES_DEDUP_FILE)

    return {
        'file_dep': [FEATURES_FILE],
        'actions': [cmd],
        'targets': [FEATURES_DEDUP_FILE],
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

####################
# initial search
####################

def task_render_mgf():
    # input
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-dedup.hdf'.format(FEATURES_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])
    # command
    cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])

    return {
        'file_dep': [FEATURES_FILE],
        'actions': [cmd],
        'targets': [MGF_FILE],
        'clean': ['rm -rf {}'.format(MGF_DIR)],
        'verbosity': 2
    }

def task_search_mgf():
    # input
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])
    # cmd
    cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=config['run_name'])

    return {
        'file_dep': [MGF_FILE],
        'actions': [cmd],
        'targets': [comet_output],
        'clean': ['rm -rf {experiment_base}/comet-output-pasef'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_identify_searched_features():
    # input
    comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=config['run_name'])
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.hdf'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])

    return {
        'file_dep': [comet_output],
        'actions': [cmd],
        'targets': [IDENTIFICATIONS_FILE],
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-pasef'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

####################
# mass recalibration
####################

def task_mass_recalibration():
    # input
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.hdf'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    # command
    cmd = 'python -u recalibrate-feature-mass.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -pdm {precursor_definition_method} -rm cluster'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'])
    # output
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    RECAL_FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.hdf'.format(FEATURES_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])

    return {
        'file_dep': [IDENTIFICATIONS_FILE],
        'actions': [cmd],
        'targets': [RECAL_FEATURES_FILE],
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

####################
# second search
####################

def task_render_mgf_recalibrated():
    # input
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.hdf'.format(FEATURES_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])
    # command
    cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])

    return {
        'file_dep': [FEATURES_FILE],
        'actions': [cmd],
        'targets': [MGF_FILE],
        'clean': ['rm -rf {}'.format(MGF_DIR)],
        'verbosity': 2
    }

def task_search_mgf_recalibrated():
    # input
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, config['experiment_name'], config['run_name'], config['precursor_definition_method'])
    # cmd
    cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=config['run_name'])

    return {
        'file_dep': [MGF_FILE],
        'actions': [cmd],
        'targets': [comet_output],
        'clean': ['rm -rf {experiment_base}/comet-output-pasef-recalibrated'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_identify_searched_features_recalibrated():
    # input
    comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=config['run_name'])
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.hdf'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])

    return {
        'file_dep': [comet_output],
        'actions': [cmd],
        'targets': [IDENTIFICATIONS_FILE],
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-pasef-recalibrated'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_make_copies():
    target_directory_name = ''

    def set_up_target_dir():
        nonlocal target_directory_name
        # set up copy directory
        d = datetime.datetime.now()
        target_directory_name = '/media/big-ssd/results-{}/{}-results-cs-{}-fmdw-{}-{}'.format(config['experiment_name'], config['experiment_name'], config['correct_for_saturation'], config['filter_by_mass_defect'], d.strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(target_directory_name):
            os.makedirs(target_directory_name)
        print('copying results to {}'.format(target_directory_name))

    def finish_up():
        stop_run = time.time()
        print("total running time ({}): {} seconds".format(config, round(stop_run-start_run,1)))

    def create_features_cmd_string():
        # copy features
        source_features_dir = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
        cmd = 'cp -r {}/ {}/'.format(source_features_dir, target_directory_name)
        return cmd

    def create_mgfs_cmd_string():
        # copy MGFs
        source_features_dir = '{}/mgf-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
        cmd = 'cp -r {}/ {}/'.format(source_features_dir, target_directory_name)
        return cmd

    def create_idents_cmd_string():
        # copy identifications
        source_identifications_dir = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
        cmd = 'cp -r {}/ {}/'.format(source_identifications_dir, target_directory_name)
        return cmd

    # input
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_RECAL_FILE = '{}/exp-{}-identifications-{}-recalibrated.pkl'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])

    return {
        'file_dep': [IDENTIFICATIONS_RECAL_FILE],
        'actions': [set_up_target_dir, CmdAction(create_features_cmd_string), CmdAction(create_mgfs_cmd_string), CmdAction(create_idents_cmd_string), finish_up],
        'verbosity': 2
    }
