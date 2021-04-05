# This is the set of tasks to take a raw instrument database and create a list of peptides

experiment_base_dir = '/media/big-ssd/experiments'
experiment_name = 'P3856'
feature_detection_method = 'pasef'

EXPERIMENT_DIR = "{}/{}".format(experiment_base_dir, experiment_name)
RUN_NAME = 'P3856_YHE211_1_Slot1-1_1_5104'
INI_NAME = 'pasef-process-short-gradient.ini'
FASTA_NAME = '../fasta/Human_Yeast_Ecoli.fasta'

def task_identify_searched_features_recalibrated():
    return {
        'file_dep': ['identify-searched-features.py',INI_NAME,FASTA_NAME,'{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=RUN_NAME)],
        'actions': ['python -u identify-searched-features.py -eb {experiment_base} -en {experiment_name} -ini {ini_name} -ff {fasta_name} -fdm pasef -ns -recal'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, ini_name=INI_NAME, fasta_name=FASTA_NAME)],
        'targets': ['{}/identifications-pasef/exp-{}-identifications-pasef-recalibrated.pkl'.format(EXPERIMENT_DIR, experiment_name)],
        'verbosity': 2
    }

def task_search_mgf_recalibrated():
    MGF_DIR = "{}/mgf-recalibrated".format(EXPERIMENT_DIR)
    MGF_FILE = '{}/exp-{}-run-{}-features-{}-recalibrated.mgf'.format(MGF_DIR, experiment_name, run_name, feature_detection_method)
    return {
        'file_dep': ['search-mgf-against-sequence-db.py',INI_NAME,FASTA_NAME,'{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=RUN_NAME)],
        'actions': ['python -u search-mgf-against-sequence-db.py -eb {experiment_base} -en {experiment_name} -rn {run_name} -ini {ini_name} -ff {fasta_name} -fdm pasef -ns -recal'.format(experiment_base=experiment_base_dir, experiment_name=experiment_name, run_name=RUN_NAME, ini_name=INI_NAME, fasta_name=FASTA_NAME)],
        'targets': ['{}/identifications-pasef/exp-{}-identifications-pasef-recalibrated.pkl'.format(EXPERIMENT_DIR, experiment_name)],
        'verbosity': 2
    }
