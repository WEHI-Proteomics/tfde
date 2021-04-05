# This is the set of tasks to take a raw instrument database and create a list of peptides

experiment_base_dir = '/media/big-ssd/experiments'
experiment_name = 'P3856'
EXPERIMENT_DIR = "{}/{}".format(experiment_base_dir, experiment_name)
RUN_NAME = 'P3856_YHE211_1_Slot1-1_1_5104'

def task_identify_searched_features():
    return {
        'file_dep': ['identify-searched-features.py','pasef-process-short-gradient.ini','../fasta/Human_Yeast_Ecoli.fasta','{experiment_base}/comet-output-pasef-recalibrated/{run_name}.comet.log.txt'.format(experiment_base=EXPERIMENT_DIR, run_name=RUN_NAME)],
        'actions': ['python -u identify-searched-features.py -eb {} -en {} -ini pasef-process-short-gradient.ini -ff ../fasta/Human_Yeast_Ecoli.fasta -fdm pasef -ns -recal'.format(experiment_base_dir, experiment_name)],
        'targets': ['{}/identifications-pasef/exp-{}-identifications-pasef-recalibrated.pkl'.format(experiment_name)],
        'verbosity': 2
    }
