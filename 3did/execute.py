from doit import get_var
from doit.action import CmdAction
from os.path import expanduser
import datetime
import time
import shutil
import os
import sys

# This is the set of tasks to run 3DID on a raw instrument database

# default configuration file location
ini_file = '{}/../pipeline/pasef-process-short-gradient.ini'.format(os.path.dirname(os.path.realpath(__file__)))

# the function get_var() gets the named argument from the command line as a string; if the argument is not present it uses the specified default
config = {
    'experiment_base_dir': get_var('eb', '/media/big-ssd/experiments'),
    'experiment_name': get_var('en', None),
    'run_name': get_var('rn', None),
    'raw_database_name': get_var('rdn', None),
    'ini_file': get_var('ini', ini_file),
    'proportion_of_cores_to_use': get_var('pc', 0.8),
    'mz_width_per_segment': get_var('mw', 20),
    'minvi': get_var('minvi', 3000)
    }

print('execution arguments: {}'.format(config))

EXPERIMENT_DIR = "{}/{}".format(config['experiment_base_dir'], config['experiment_name'])

start_run = time.time()


####################
# feature extraction
####################
def task_detect_features():
    # input
    RAW_DATABASE_NAME = "{experiment_dir}/raw-databases/{run_name}.d".format(experiment_dir=EXPERIMENT_DIR, run_name=config['run_name'])
    # command
    cmd = 'python -u detect-features-with-3did.py -eb {experiment_base} -en {experiment_name} -rdn {raw_database_name} -rn {run_name} -mw {mz_width_per_segment} -pc {proportion_of_cores_to_use} -ini {INI_FILE} -rm cluster -minvi {minvi}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], raw_database_name=config['raw_database_name'], run_name=config['run_name'], mz_width_per_segment=config['mz_width_per_segment'], proportion_of_cores_to_use=config['proportion_of_cores_to_use'], INI_FILE=config['ini_file'], minvi=config['minvi'])
    # output
    FEATURES_DIR = '{experiment_dir}/features-3did'.format(experiment_dir=EXPERIMENT_DIR)
    FEATURES_FILE = '{features_dir}/exp-{experiment_name}-run-{run_name}-features-3did.pkl'.format(features_dir=FEATURES_DIR, experiment_name=config['experiment_name'], run_name=config['run_name'])

    return {
        'file_dep': [RAW_DATABASE_NAME],
        'actions': [cmd],
        'targets': [FEATURES_FILE],
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

def task_classify_features():
    # input
    FEATURES_DIR = '{experiment_dir}/features-3did'.format(experiment_dir=EXPERIMENT_DIR)
    FEATURES_FILE = '{features_dir}/exp-{experiment_name}-run-{run_name}-features-3did.pkl'.format(features_dir=FEATURES_DIR, experiment_name=config['experiment_name'], run_name=config['run_name'])
    # command
    cmd = 'python -u classify-detected-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'])
    # output
    FEATURES_IDENT_FILE = '{features_dir}/exp-{experiment_name}-run-{run_name}-features-3did-ident.pkl'.format(features_dir=FEATURES_DIR, experiment_name=config['experiment_name'], run_name=config['run_name'])

    return {
        'file_dep': [FEATURES_FILE],
        'actions': [cmd],
        'targets': [FEATURES_IDENT_FILE],
        'clean': ['rm {}'.format(FEATURES_IDENT_FILE)],
        'verbosity': 2
    }

def task_remove_duplicate_features():
    # input
    FEATURES_DIR = "{experiment_dir}/features-3did".format(experiment_dir=EXPERIMENT_DIR)
    FEATURES_IDENT_FILE = '{features_dir}/exp-{experiment_name}-run-{run_name}-features-3did-ident.pkl'.format(features_dir=FEATURES_DIR, experiment_name=config['experiment_name'], run_name=config['run_name'])
    # command
    cmd = 'python -u ../pipeline/remove-duplicate-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -pdm 3did'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=config['run_name'], INI_FILE=config['ini_file'])
    # output
    FEATURES_DEDUP_FILE = '{features_dir}/exp-{experiment_name}-run-{run_name}-features-3did-dedup.pkl'.format(features_dir=FEATURES_DIR, experiment_name=config['experiment_name'], run_name=config['run_name'])

    return {
        'file_dep': [FEATURES_IDENT_FILE],
        'actions': [cmd],
        'targets': [FEATURES_DEDUP_FILE],
        'clean': ['rm {}'.format(FEATURES_DEDUP_FILE)],
        'verbosity': 2
    }

def task_make_copies():
    target_directory_name = ''

    def set_up_target_dir():
        nonlocal target_directory_name
        # set up copy directory
        d = datetime.datetime.now()
        target_directory_name = '/media/big-ssd/results-{}-3did/minvi-{}-{}'.format(config['experiment_name'], config['minvi'], d.strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(target_directory_name):
            os.makedirs(target_directory_name)
        print('copying results to {}'.format(target_directory_name))

    def finish_up():
        stop_run = time.time()
        print("total running time ({}): {} seconds".format(config, round(stop_run-start_run,1)))

    def create_features_cmd_string():
        # copy features
        source_features_dir = '{}/features-3did'.format(EXPERIMENT_DIR)
        cmd = 'cp -r {}/ {}/'.format(source_features_dir, target_directory_name)
        return cmd

    # input
    FEATURES_DIR = "{}/features-3did".format(EXPERIMENT_DIR)
    FEATURES_DEDUP_FILE = '{}/exp-{}-run-{}-features-3did-dedup.pkl'.format(FEATURES_DIR, config['experiment_name'], config['run_name'])

    return {
        'file_dep': [FEATURES_DEDUP_FILE],
        'actions': [set_up_target_dir, CmdAction(create_features_cmd_string), finish_up],
        'verbosity': 2
    }
