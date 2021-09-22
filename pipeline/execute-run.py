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
    'run_names': get_var('runs', None),
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

# the names of the runs to process
run_names_l = config['run_names'].split(',')

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
    depend_l = []
    cmd_l = []
    target_l = []
    CUBOIDS_DIR = '{}/precursor-cuboids-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        # input
        RAW_DATABASE_NAME = "{}/raw-databases/{}.d/analysis.tdf".format(EXPERIMENT_DIR, run_name)
        depend_l.append(RAW_DATABASE_NAME)
        # command
        cmd = 'python -u define-precursor-cuboids-pasef.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rl {rl} -ru {ru} -rm cluster'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], rl=int(config['rt_lower']), ru=int(config['rt_upper']))
        cmd_l.append(cmd)
        # output
        CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-{}.feather'.format(CUBOIDS_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(CUBOIDS_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(CUBOIDS_DIR)],
        'verbosity': 2
    }

def task_detect_features():
    depend_l = []
    cmd_l = []
    target_l = []
    CUBOIDS_DIR = "{}/precursor-cuboids-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        # input
        CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-{}.feather'.format(CUBOIDS_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        depend_l.append(CUBOIDS_FILE)
        # command
        cmd = 'python -u detect-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rm cluster -pc {proportion_of_cores_to_use} -rl {rl} -ru {ru} {cs} {fmdw}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], proportion_of_cores_to_use=config['proportion_of_cores_to_use'], rl=int(config['rt_lower']), ru=int(config['rt_upper']), cs=config['cs_flag'], fmdw=config['fmdw_flag'])
        cmd_l.append(cmd)
        # output
        FEATURES_FILE = '{}/chunks/exp-{}-run-{}-features-{}-000.feather'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(FEATURES_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

def task_remove_duplicate_features():
    depend_l = []
    cmd_l = []
    target_l = []
    FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        # input
        FEATURES_FILE = '{}/chunks/exp-{}-run-{}-features-{}-000.feather'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        depend_l.append(FEATURES_FILE)
        # command
        cmd = 'python -u remove-duplicate-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        FEATURES_DEDUP_FILE = '{}/exp-{}-run-{}-features-{}-dedup.feather'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(FEATURES_DEDUP_FILE)
        # pass-through command (no de-dup)
        # cmd = 'cp {} {}'.format(FEATURES_FILE, FEATURES_DEDUP_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

####################
# initial search
####################

def task_render_mgf():
    depend_l = []
    cmd_l = []
    target_l = []
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        # input
        FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-dedup.feather'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        depend_l.append(FEATURES_FILE)
        # command
        cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(MGF_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(MGF_DIR)],
        'verbosity': 2
    }

def task_search_mgf():
    depend_l = []
    cmd_l = []
    target_l = []
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        # input
        MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        depend_l.append(MGF_FILE)
        # cmd
        cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        target_l.append(comet_output)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {experiment_base}/comet-output-pasef'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_identify_searched_features():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    for run_name in run_names_l:
        comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        depend_l.append(comet_output)
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    cmd_l.append(cmd)
    # output
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.feather'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    target_l.append(IDENTIFICATIONS_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-pasef'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

####################
# mass recalibration
####################

def task_mass_recalibration():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.feather'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    depend_l.append(IDENTIFICATIONS_FILE)
    # command
    cmd = 'python -u recalibrate-feature-mass.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -pdm {precursor_definition_method} -rm cluster'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'])
    cmd_l.append(cmd)
    # output
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        RECAL_FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.feather'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(RECAL_FEATURES_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

####################
# second search
####################

def task_render_mgf_recalibrated():
    depend_l = []
    cmd_l = []
    target_l = []
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        # input
        FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.feather'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        depend_l.append(FEATURES_FILE)
        # command
        cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(MGF_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(MGF_DIR)],
        'verbosity': 2
    }

def task_search_mgf_recalibrated():
    depend_l = []
    cmd_l = []
    target_l = []
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    for run_name in run_names_l:
        # input
        MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        depend_l.append(MGF_FILE)
        # cmd
        cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        target_l.append(comet_output)

    return {
        'file_dep': [MGF_FILE],
        'actions': [cmd],
        'targets': [comet_output],
        'clean': ['rm -rf {experiment_base}/comet-output-pasef-recalibrated'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_identify_searched_features_recalibrated():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    for run_name in run_names_l:
        comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        depend_l.append(comet_output)
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    cmd_l.append(cmd)
    # output
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.feather'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    target_l.append(IDENTIFICATIONS_FILE)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-pasef-recalibrated'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_make_copies():
    target_directory_name = ''

    def set_up_target_dir():
        nonlocal target_directory_name
        # set up copy directory
        d = datetime.datetime.now()
        target_directory_name = '/media/big-ssd/results-{}/cs-{}-fmdw-{}-{}'.format(config['experiment_name'], config['correct_for_saturation'], config['filter_by_mass_defect'], d.strftime("%Y-%m-%d-%H-%M-%S"))
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
    IDENTIFICATIONS_RECAL_FILE = '{}/exp-{}-identifications-{}-recalibrated.feather'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])

    return {
        'file_dep': [IDENTIFICATIONS_RECAL_FILE],
        'actions': [set_up_target_dir, CmdAction(create_features_cmd_string), CmdAction(create_mgfs_cmd_string), CmdAction(create_idents_cmd_string), finish_up],
        'verbosity': 2
    }
