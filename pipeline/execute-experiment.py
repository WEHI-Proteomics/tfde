from doit import get_var
from doit.action import CmdAction
import datetime
import time
import os
import glob

# This is the set of tasks to take a raw instrument database and create a list of peptides


def get_run_names(experiment_dir):
    # process all the runs
    database_names_l = glob.glob("{}/converted-databases/*.sqlite".format(experiment_dir))
    # convert the raw databases
    run_names = []
    for database_name in database_names_l:
        run_name = os.path.basename(database_name).split('.sqlite')[0].split('run-')[1].split('-converted')[0]
        run_names.append(run_name)
    return run_names


ini_file = '{}/pasef-process-short-gradient.ini'.format(os.path.dirname(os.path.realpath(__file__)))
fasta_file_name = '{}/../fasta/Human_Yeast_Ecoli.fasta'.format(os.path.dirname(os.path.realpath(__file__)))


# the function get_var() gets the named argument from the command line as a string; if it's not present it uses the specified default
config = {
    'experiment_base_dir': get_var('eb', '/media/big-ssd/experiments'),
    'experiment_name': get_var('en', None),
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
# raw conversion (TO BE ADDED)
####################





####################
# feature extraction
####################
def task_define_precursor_cuboids():
    # input
    CONVERTED_DB_DIR = "{}/converted-databases".format(EXPERIMENT_DIR)
    converted_database_file_list = glob.glob('{}/*-converted.sqlite'.format(CONVERTED_DB_DIR))
    # output directory
    CUBOIDS_DIR = '{}/precursor-cuboids-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])

    cmd_l = []
    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # command
        cmd = 'python -u define-precursor-cuboids-pasef.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rl {rl} -ru {ru} -rm cluster -pc {pc}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], rl=int(config['rt_lower']), ru=int(config['rt_upper'], pc=config['proportion_of_cores_to_use']))
        cmd_l.append(cmd)
        # outputs
        cuboids_file = '{}/exp-{}-run-{}-precursor-cuboids-{}.pkl'.format(CUBOIDS_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(cuboids_file)

    return {
        'file_dep': converted_database_file_list,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(CUBOIDS_DIR)],
        'verbosity': 2
    }

def task_detect_features():
    CUBOIDS_DIR = "{}/precursor-cuboids-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])

    dependency_l = []
    cmd_l = []
    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        cuboids_file = '{}/exp-{}-run-{}-precursor-cuboids-{}.pkl'.format(CUBOIDS_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        dependency_l.append(cuboids_file)
        # command
        cmd = 'python -u detect-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rm cluster -pc {proportion_of_cores_to_use} -pdm {precursor_definition_method} -rl {rl} -ru {ru} {cs} {fmdw}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], proportion_of_cores_to_use=config['proportion_of_cores_to_use'], precursor_definition_method=config['precursor_definition_method'], rl=int(config['rt_lower']), ru=int(config['rt_upper']), cs=config['cs_flag'], fmdw=config['fmdw_flag'])
        cmd_l.append(cmd)
        # output
        features_file = '{}/exp-{}-run-{}-features-{}.pkl'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(features_file)

    return {
        'file_dep': dependency_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

def task_remove_duplicate_features():
    FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])

    dependency_l = []
    cmd_l = []
    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        features_file = '{}/exp-{}-run-{}-features-{}.pkl'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        dependency_l.append(features_file)
        # command
        cmd = 'python -u remove-duplicate-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        features_dedup_file = '{}/exp-{}-run-{}-features-{}-dedup.pkl'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(features_dedup_file)
        # pass-through command (no de-dup)
        # cmd = 'cp {} {}'.format(features_file, features_dedup_file)

    return {
        'file_dep': dependency_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }



####################
# initial search
####################

def task_render_mgf():
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])

    dependency_l = []
    cmd_l = []
    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        features_file = '{}/exp-{}-run-{}-features-{}-dedup.pkl'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        dependency_l.append(features_file)
        # command
        cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        mgf_file = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(mgf_file)

    return {
        'file_dep': dependency_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(MGF_DIR)],
        'verbosity': 2
    }

def task_search_mgf():
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])

    dependency_l = []
    cmd_l = []
    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        mgf_file = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        dependency_l.append(mgf_file)
        # cmd
        cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        target_l.append(comet_output)

    return {
        'file_dep': dependency_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {experiment_base}/comet-output-pasef'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_identify_searched_features():
    dependency_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        dependency_l.append(comet_output)
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.pkl'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])

    return {
        'file_dep': dependency_l,
        'actions': [cmd],
        'targets': [IDENTIFICATIONS_FILE],
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-pasef'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

####################
# mass recalibration
####################

def task_mass_recalibration():
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])

    # input
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.pkl'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    # command
    cmd = 'python -u recalibrate-feature-mass.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -pdm {precursor_definition_method} -rm cluster -pc {pc}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'], pc=config['proportion_of_cores_to_use'])

    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # output
        recal_features_file = '{}/exp-{}-run-{}-features-{}-recalibrated.pkl'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(recal_features_file)

    return {
        'file_dep': [IDENTIFICATIONS_FILE],
        'actions': [cmd],
        'targets': target_l,
        'clean': ['rm -rf {}'.format(FEATURES_DIR)],
        'verbosity': 2
    }

####################
# second search
####################

def task_render_mgf_recalibrated():
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])

    dependency_l = []
    cmd_l = []
    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        features_file = '{}/exp-{}-run-{}-features-{}-recalibrated.pkl'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        dependency_l.append(features_file)
        # command
        cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        mgf_file = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        target_l.append(mgf_file)

    return {
        'file_dep': dependency_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(MGF_DIR)],
        'verbosity': 2
    }

def task_search_mgf_recalibrated():
    MGF_DIR = "{}/mgf-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])

    dependency_l = []
    cmd_l = []
    target_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        mgf_file = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        dependency_l.append(mgf_file)
        # cmd
        cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
        cmd_l.append(cmd)
        # output
        comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        target_l.append(comet_output)

    return {
        'file_dep': dependency_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {experiment_base}/comet-output-pasef-recalibrated'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }

def task_identify_searched_features_recalibrated():
    dependency_l = []
    for run_name in get_run_names(EXPERIMENT_DIR):
        # input
        comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
        dependency_l.append(comet_output)
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -pdm {precursor_definition_method} -recal'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], fasta_name=config['fasta_file_name'], precursor_definition_method=config['precursor_definition_method'])
    # output
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.pkl'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])

    return {
        'file_dep': dependency_l,
        'actions': [cmd],
        'targets': [IDENTIFICATIONS_FILE],
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-pasef-recalibrated'.format(experiment_base=EXPERIMENT_DIR)],
        'verbosity': 2
    }
