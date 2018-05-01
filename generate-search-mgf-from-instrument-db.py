import os
from multiprocessing import Pool

processes = (
    'sum-frames-ms1.py -sdb /media/data-drive/Hela_20A_20R_500.sqlite -ddb test-Hela_20A_20R_500-1-5.sqlite -ce 7 -fl 1 -fu 5', 
    'sum-frames-ms1.py -sdb /media/data-drive/Hela_20A_20R_500.sqlite -ddb test-Hela_20A_20R_500-6-10.sqlite -ce 7 -fl 6 -fu 10',
    )
# other = ('process3.py',)

def run_process(process):
    os.system('python {}'.format(process))


pool = Pool(processes=3)
pool.map(run_process, processes)
# pool.map(run_process, other)
