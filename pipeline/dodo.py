# This is the set of tasks to take a raw instrument database and create a list of peptides

def task_identify_searched_features():
    return {
        'file_dep': ['identify-searched-features.py'],
        'actions': ['python -u identify-searched-features.py -eb /media/big-ssd/experiments -en P3856 -ini pasef-process-short-gradient.ini -ff ../fasta/Human_Yeast_Ecoli.fasta -fdm pasef -ns -recal'],
        'targets': ['/media/big-ssd/experiments/P3856/identifications-pasef/exp-P3856-identifications-pasef-recalibrated.pkl']
    }
