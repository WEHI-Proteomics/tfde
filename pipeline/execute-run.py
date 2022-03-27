from doit import get_var
from doit.action import CmdAction
import datetime
import time
import os
from os.path import expanduser

# This is the set of tasks to take a raw instrument database and create a list of peptides

# To run from a clean state:
# doit -f ./tfde/pipeline/execute-run.py clean pc=0.8 cs=true fmdw=true en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104,P3856_YHE211_2_Slot1-1_1_5105,P3856_YHE211_3_Slot1-1_1_5106,P3856_YHE211_4_Slot1-1_1_5107,P3856_YHE211_5_Slot1-1_1_5108,P3856_YHE211_6_Slot1-1_1_5109,P3856_YHE211_7_Slot1-1_1_5110,P3856_YHE211_8_Slot1-1_1_5111,P3856_YHE211_9_Slot1-1_1_5112,P3856_YHE211_10_Slot1-1_1_5113 rl=1650 ru=2200 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta"
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104,P3856_YHE211_2_Slot1-1_1_5105,P3856_YHE211_3_Slot1-1_1_5106,P3856_YHE211_4_Slot1-1_1_5107,P3856_YHE211_5_Slot1-1_1_5108,P3856_YHE211_6_Slot1-1_1_5109,P3856_YHE211_7_Slot1-1_1_5110,P3856_YHE211_8_Slot1-1_1_5111,P3856_YHE211_9_Slot1-1_1_5112,P3856_YHE211_10_Slot1-1_1_5113 rl=1650 ru=2200 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta"

# To run a single task, for example 'identify_searched_features':
# doit -f ./tfde/pipeline/execute-run.py clean identify_searched_features pc=0.8 cs=true fmdw=true en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104,P3856_YHE211_2_Slot1-1_1_5105,P3856_YHE211_3_Slot1-1_1_5106,P3856_YHE211_4_Slot1-1_1_5107,P3856_YHE211_5_Slot1-1_1_5108,P3856_YHE211_6_Slot1-1_1_5109,P3856_YHE211_7_Slot1-1_1_5110,P3856_YHE211_8_Slot1-1_1_5111,P3856_YHE211_9_Slot1-1_1_5112,P3856_YHE211_10_Slot1-1_1_5113 rl=1650 ru=2200 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta"
# doit -f ./tfde/pipeline/execute-run.py identify_searched_features pc=0.8 cs=true fmdw=true en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104,P3856_YHE211_2_Slot1-1_1_5105,P3856_YHE211_3_Slot1-1_1_5106,P3856_YHE211_4_Slot1-1_1_5107,P3856_YHE211_5_Slot1-1_1_5108,P3856_YHE211_6_Slot1-1_1_5109,P3856_YHE211_7_Slot1-1_1_5110,P3856_YHE211_8_Slot1-1_1_5111,P3856_YHE211_9_Slot1-1_1_5112,P3856_YHE211_10_Slot1-1_1_5113 rl=1650 ru=2200 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta"


ini_file = '{}/pasef-process-short-gradient.ini'.format(os.path.dirname(os.path.realpath(__file__)))
fasta_file_name = '{}/../fasta/Human_Yeast_Ecoli.fasta'.format(os.path.dirname(os.path.realpath(__file__)))


# the function get_var() gets the named argument from the command line as a string; if it's not present it uses the specified default
config = {
    'experiment_base_dir': get_var('eb', '/media/big-ssd/experiments'),
    'experiment_name': get_var('en', None),
    'run_names': get_var('rn', None),
    'fasta_file_name': get_var('ff', fasta_file_name),
    'ini_file': get_var('ini', ini_file),
    'precursor_definition_method': get_var('pdm', 'pasef'),
    'rt_lower': get_var('rl', 1650),
    'rt_upper': get_var('ru', 2200),
    'correct_for_saturation': get_var('cs', 'true'),
    'filter_by_mass_defect': get_var('fmdw', 'true'),
    'use_denoised_db': get_var('dd', 'false'),
    'proportion_of_cores_to_use': get_var('pc', 0.8)
}

print('execution arguments: {}'.format(config))

# the names of the runs to process
if config['run_names'] is None:
    run_names_l = []
else:
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

# use the denoised data
if config['use_denoised_db'] == 'true':
    config['use_denoised_db_flag'] = '-d'
else:
    config['use_denoised_db_flag'] = ''

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
        cmd = 'python -u define-precursor-cuboids-pasef.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rl {rl} -ru {ru} -rm cluster {denoised}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], rl=int(config['rt_lower']), ru=int(config['rt_upper']), denoised=config['use_denoised_db_flag'])
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
        CUBOIDS_FILE = '{}/exp-{}-run-{}-precursor-cuboids-{}.feather'.format(CUBOIDS_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        FEATURES_FILE = '{}/chunks/exp-{}-run-{}-features-{}-000.feather'.format(FEATURES_DIR, config['experiment_name'], run_name, config['precursor_definition_method'])
        if not os.path.isfile(FEATURES_FILE):
            # input
            depend_l.append(CUBOIDS_FILE)
            # command
            cmd = 'python -u detect-features.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -rm cluster -pc {proportion_of_cores_to_use} -rl {rl} -ru {ru} {cs} {fmdw} {denoised}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_name=run_name, INI_FILE=config['ini_file'], proportion_of_cores_to_use=config['proportion_of_cores_to_use'], rl=int(config['rt_lower']), ru=int(config['rt_upper']), cs=config['cs_flag'], fmdw=config['fmdw_flag'], denoised=config['use_denoised_db_flag'])
            cmd_l.append(cmd)
            # output
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
        comet_output = '{experiment_base}/comet-output-{pdm}/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'], run_name=run_name)
        target_l.append(comet_output)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {experiment_base}/comet-output-{pdm}'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'])],
        'verbosity': 2
    }

def task_identify_searched_features():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    for run_name in run_names_l:
        comet_output = '{experiment_base}/comet-output-{pdm}/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'], run_name=run_name)
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
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-{pdm}'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'])],
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
    cmd = 'python -u recalibrate-feature-mass.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'])
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
        comet_output = '{experiment_base}/comet-output-{pdm}-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'], run_name=run_name)
        target_l.append(comet_output)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {experiment_base}/comet-output-{pdm}-recalibrated'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'])],
        'verbosity': 2
    }

def task_identify_searched_features_recalibrated():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    for run_name in run_names_l:
        comet_output = '{experiment_base}/comet-output-{pdm}-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'], run_name=run_name)
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
        'clean': ['rm -rf {}'.format(IDENTIFICATIONS_DIR), 'rm -rf {experiment_base}/percolator-output-{pdm}-recalibrated'.format(experiment_base=EXPERIMENT_DIR, pdm=config['precursor_definition_method'])],
        'verbosity': 2
    }


####################################
# build the peptide sequence library
####################################

def task_build_sequence_library():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.feather'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    depend_l = [IDENTIFICATIONS_FILE]
    # cmd
    cmd = 'python -u build-sequence-library.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -pdm {precursor_definition_method}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'], precursor_definition_method=config['precursor_definition_method'])
    cmd_l.append(cmd)
    # output
    SEQUENCE_LIBRARY_DIR = "{}/sequence-library-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.feather".format(SEQUENCE_LIBRARY_DIR)
    target_l.append(SEQUENCE_LIBRARY_FILE_NAME)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(SEQUENCE_LIBRARY_DIR)],
        'verbosity': 2
    }


#############################
# build coordinate estimators
#############################

def task_build_coordinate_estimators():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    SEQUENCE_LIBRARY_DIR = "{}/sequence-library-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.feather".format(SEQUENCE_LIBRARY_DIR)
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.feather'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    depend_l = [SEQUENCE_LIBRARY_FILE_NAME,IDENTIFICATIONS_FILE]
    # cmd
    cmd = 'python -u build-run-coordinate-estimators.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -snmp'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], INI_FILE=config['ini_file'])
    cmd_l.append(cmd)
    # output
    COORDINATE_ESTIMATORS_DIR = "{}/coordinate-estimators".format(EXPERIMENT_DIR)
    for run_name in run_names_l:
        for dim in ['mz','scan','rt']:
            ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, dim)
            target_l.append(ESTIMATOR_MODEL_FILE_NAME)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(COORDINATE_ESTIMATORS_DIR)],
        'verbosity': 2
    }


########################################
# extract features for library sequences
########################################

def task_extract_features_for_library_sequences():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    SEQUENCE_LIBRARY_DIR = "{}/sequence-library-{}".format(EXPERIMENT_DIR, config['precursor_definition_method'])
    SEQUENCE_LIBRARY_FILE_NAME = "{}/sequence-library.feather".format(SEQUENCE_LIBRARY_DIR)
    depend_l = [SEQUENCE_LIBRARY_FILE_NAME]
    COORDINATE_ESTIMATORS_DIR = "{}/coordinate-estimators".format(EXPERIMENT_DIR)
    for run_name in run_names_l:
        for dim in ['mz','scan','rt']:
            ESTIMATOR_MODEL_FILE_NAME = "{}/run-{}-{}-estimator.pkl".format(COORDINATE_ESTIMATORS_DIR, run_name, dim)
            depend_l.append(ESTIMATOR_MODEL_FILE_NAME)
    # cmd
    cmd = 'python -u bulk-extract-sequence-library-features.py -eb {experiment_base} -en {experiment_name} -rn {run_names} -ini {INI_FILE} {denoised}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_names=config['run_names'], INI_FILE=config['ini_file'], denoised=config['use_denoised_db_flag'])
    cmd_l.append(cmd)
    # output
    TARGET_DECOY_MODEL_DIR = "{}/target-decoy-models".format(EXPERIMENT_DIR)
    METRICS_DB_NAME = "{}/experiment-metrics-for-library-sequences.sqlite".format(TARGET_DECOY_MODEL_DIR)
    target_l = [METRICS_DB_NAME]

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm {}'.format(METRICS_DB_NAME)],
        'verbosity': 2
    }


#########################
# build target classifier
#########################

def task_build_target_classifier():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    TARGET_DECOY_MODEL_DIR = "{}/target-decoy-models".format(EXPERIMENT_DIR)
    METRICS_DB_NAME = "{}/experiment-metrics-for-library-sequences.sqlite".format(TARGET_DECOY_MODEL_DIR)
    depend_l = [METRICS_DB_NAME]
    # cmd
    cmd = 'python -u build-target-decoy-classifier.py -eb {experiment_base} -en {experiment_name} --minimum_number_files 5 --training_set_multiplier 1 -snmp'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'])
    cmd_l.append(cmd)
    # output
    CLASSIFIER_FILE_NAME = "{}/target-decoy-classifier.pkl".format(TARGET_DECOY_MODEL_DIR)
    target_l = [CLASSIFIER_FILE_NAME]

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm {}'.format(CLASSIFIER_FILE_NAME)],
        'verbosity': 2
    }


#############################
# classify extracted features
#############################

def task_classify_extracted_features():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    TARGET_DECOY_MODEL_DIR = "{}/target-decoy-models".format(EXPERIMENT_DIR)
    CLASSIFIER_FILE_NAME = "{}/target-decoy-classifier.pkl".format(TARGET_DECOY_MODEL_DIR)
    METRICS_DB_NAME = "{}/experiment-metrics-for-library-sequences.sqlite".format(TARGET_DECOY_MODEL_DIR)
    depend_l = [CLASSIFIER_FILE_NAME,METRICS_DB_NAME]
    # cmd
    cmd = 'python -u classify-extracted-features.py -eb {experiment_base} -en {experiment_name} -rn {run_names}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'], run_names=config['run_names'])
    cmd_l.append(cmd)
    # output
    EXTRACTED_FEATURES_DIR = "{}/extracted-features".format(EXPERIMENT_DIR)
    EXTRACTED_FEATURES_DB_NAME = "{}/extracted-features.sqlite".format(EXTRACTED_FEATURES_DIR)
    target_l = [EXTRACTED_FEATURES_DB_NAME]

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(EXTRACTED_FEATURES_DIR)],
        'verbosity': 2
    }


#############################
# generate summarised results
#############################

def task_summarise_results():
    depend_l = []
    cmd_l = []
    target_l = []
    # input
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.feather'.format(IDENTIFICATIONS_DIR, config['experiment_name'], config['precursor_definition_method'])
    EXTRACTED_FEATURES_DB_NAME = '{}/extracted-features/extracted-features.sqlite'.format(EXPERIMENT_DIR)
    depend_l = [IDENTIFICATIONS_FILE,EXTRACTED_FEATURES_DB_NAME]
    # cmd
    cmd = 'python -u generate-results.py -eb {experiment_base} -en {experiment_name}'.format(experiment_base=config['experiment_base_dir'], experiment_name=config['experiment_name'])
    cmd_l.append(cmd)
    # output
    RESULTS_DIR = "{}/summarised-results".format(EXPERIMENT_DIR)
    RESULTS_DB_FILE_NAME = '{}/results.sqlite'.format(RESULTS_DIR)
    target_l.append(RESULTS_DB_FILE_NAME)

    return {
        'file_dep': depend_l,
        'actions': cmd_l,
        'targets': target_l,
        'clean': ['rm -rf {}'.format(RESULTS_DIR)],
        'verbosity': 2
    }


###################
# backup everything
###################

def task_make_copies():
    target_directory_name = ''

    def setup_target():
        nonlocal target_directory_name

        # display total running time
        stop_run = time.time()
        print("total running time ({}): {} seconds".format(config, round(stop_run-start_run,1)))

        # set up copy directory
        d = datetime.datetime.now()
        target_directory_name = '/media/data-4t-a/results-{}/{}'.format(config['experiment_name'], d.strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(target_directory_name)
        print('copying results to {}'.format(target_directory_name))

    def features_cmd():
        source_dir = '{}/features-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
        cmd = 'cp -r {}/ {}/'.format(source_dir, target_directory_name)
        return cmd

    def mgf_cmd():
        source_dir = '{}/mgf-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
        cmd = 'cp -r {}/ {}/'.format(source_dir, target_directory_name)
        return cmd

    def identifications_cmd():
        source_dir = '{}/identifications-{}'.format(EXPERIMENT_DIR, config['precursor_definition_method'])
        cmd = 'cp -r {}/ {}/'.format(source_dir, target_directory_name)
        return cmd

    def extracted_features_cmd():
        source_dir = '{}/extracted-features'.format(EXPERIMENT_DIR)
        cmd = 'cp -r {}/ {}/'.format(source_dir, target_directory_name)
        return cmd

    def sequence_library_cmd():
        source_dir = '{}/summarised-results'.format(EXPERIMENT_DIR)
        cmd = 'cp -r {}/ {}/'.format(source_dir, target_directory_name)
        return cmd

    def log_cmd():
        cmd = 'cp {}/bulk-run.log {}/'.format(expanduser('~'), target_directory_name)
        return cmd

    # input
    RESULTS_DIR = "{}/summarised-results".format(EXPERIMENT_DIR)
    RESULTS_DB_FILE_NAME = '{}/results.sqlite'.format(RESULTS_DIR)

    return {
        'file_dep': [RESULTS_DB_FILE_NAME],
        'actions': [setup_target, CmdAction(features_cmd), CmdAction(mgf_cmd), CmdAction(identifications_cmd), CmdAction(extracted_features_cmd), CmdAction(sequence_library_cmd), CmdAction(log_cmd)],
        'verbosity': 2
    }
