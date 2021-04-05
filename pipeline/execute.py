# This is the set of tasks to take a raw instrument database and create a list of peptides

experiment_base_dir = '/media/big-ssd/experiments'
experiment_name = 'P3856'
run_name = 'P3856_YHE211_1_Slot1-1_1_5104'
ini_file = 'pasef-process-short-gradient.ini'
fasta_file_name = '../fasta/Human_Yeast_Ecoli.fasta'
feature_detection_method = 'pasef'

EXPERIMENT_DIR = "{}/{}".format(experiment_base_dir, experiment_name)

####################
# initial search
####################

def task_render_mgf():
    # input
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, feature_detection_method)
    FEATURES_FILE = '{}/exp-{}-run-{}-features-{}.pkl'.format(FEATURES_DIR, experiment_name, run_name, feature_detection_method)
    # command
    cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -fdm pasef -ns'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, run_name=run_name)
    # output
    MGF_DIR = "{}/mgf".format(EXPERIMENT_DIR)
    MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, experiment_name, run_name, feature_detection_method)
    return {
        'file_dep': ['render-features-as-mgf.py',FEATURES_FILE,INI_NAME],
        'actions': [cmd],
        'targets': [MGF_FILE],
        'verbosity': 2
    }

def task_search_mgf():
    # input
    MGF_DIR = "{}/mgf".format(EXPERIMENT_DIR)
    MGF_FILE = '{}/exp-{}-run-{}-features-{}.mgf'.format(MGF_DIR, experiment_name, run_name, feature_detection_method)
    # cmd
    cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -fdm pasef -ns'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, run_name=run_name, INI_FILE=INI_NAME, fasta_name=fasta_file_name)
    # output
    comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
    return {
        'file_dep': ['search-mgf-against-sequence-db.py',MGF_FILE,INI_NAME,fasta_file_name],
        'actions': [cmd],
        'targets': [comet_output],
        'verbosity': 2
    }

def task_identify_searched_features():
    # input
    comet_output = '{experiment_base}/comet-output-pasef/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -fdm pasef -ns'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, INI_FILE=INI_NAME, fasta_name=fasta_file_name)
    # output
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, feature_detection_method)
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.pkl'.format(IDENTIFICATIONS_DIR, experiment_name, feature_detection_method)
    return {
        'file_dep': ['identify-searched-features.py',comet_output,INI_NAME,fasta_file_name],
        'actions': [cmd],
        'targets': [IDENTIFICATIONS_FILE],
        'verbosity': 2
    }

####################
# mass recalibration
####################

def task_mass_recalibration():
    # input
    IDENTIFICATIONS_DIR = '{}/identifications-{}'.format(EXPERIMENT_DIR, feature_detection_method)
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}.pkl'.format(IDENTIFICATIONS_DIR, experiment_name, feature_detection_method)
    # command
    cmd = 'python -u recalibrate-feature-mass.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -fdm pasef -rm cluster'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, INI_FILE=INI_NAME)
    # output
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, feature_detection_method)
    RECAL_FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.pkl'.format(FEATURES_DIR, experiment_name, run_name, feature_detection_method)
    return {
        'file_dep': ['recalibrate-feature-mass.py',IDENTIFICATIONS_FILE,INI_NAME],
        'actions': [cmd],
        'targets': [RECAL_FEATURES_FILE],
        'verbosity': 2
    }

####################
# second search
####################

def task_render_mgf_recalibrated():
    # input
    FEATURES_DIR = '{}/features-{}'.format(EXPERIMENT_DIR, feature_detection_method)
    FEATURES_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.pkl'.format(FEATURES_DIR, experiment_name, run_name, feature_detection_method)
    # command
    cmd = 'python -u render-features-as-mgf.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -fdm pasef -ns -recal'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, run_name=run_name)
    # output
    MGF_DIR = "{}/mgf-recalibrated".format(EXPERIMENT_DIR)
    MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, experiment_name, run_name, feature_detection_method)
    return {
        'file_dep': ['render-features-as-mgf.py',FEATURES_FILE,INI_NAME],
        'actions': [cmd],
        'targets': [MGF_FILE],
        'verbosity': 2
    }

def task_search_mgf_recalibrated():
    # input
    MGF_DIR = "{}/mgf-recalibrated".format(EXPERIMENT_DIR)
    MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, experiment_name, run_name, feature_detection_method)
    # cmd
    cmd = 'python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {INI_FILE} -ff {fasta_name} -fdm pasef -ns -recal'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, run_name=run_name, INI_FILE=INI_NAME, fasta_name=fasta_file_name)
    # output
    comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
    return {
        'file_dep': ['search-mgf-against-sequence-db.py',MGF_FILE,INI_NAME,fasta_file_name],
        'actions': [cmd],
        'targets': [comet_output],
        'verbosity': 2
    }

def task_identify_searched_features_recalibrated():
    # input
    comet_output = '{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=run_name)
    # cmd
    cmd = 'python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {INI_FILE} -ff {fasta_name} -fdm pasef -ns -recal'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, INI_FILE=INI_NAME, fasta_name=fasta_file_name)
    # output
    IDENTIFICATIONS_FILE = '{}/exp-{}-identifications-{}-recalibrated.pkl'.format(IDENTIFICATIONS_DIR, experiment_name, feature_detection_method)
    return {
        'file_dep': ['identify-searched-features.py',comet_output,INI_NAME,fasta_file_name],
        'actions': [cmd],
        'targets': [IDENTIFICATIONS_FILE],
        'verbosity': 2
    }
