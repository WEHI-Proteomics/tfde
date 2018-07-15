import glob
import sqlite3
import pandas as pd
import os
import multiprocessing as mp
from multiprocessing import Pool
import time

def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)

processing_start_time = time.time()

SOURCE_DATABASE_DIRECTORY = "~/UPS2_allion_Slot1-2_01_699"
file_list = glob.glob("{}/UPS2_allion_Slot1-2_01_699-features-*-*.sqlite".format(SOURCE_DATABASE_DIRECTORY))

merge_processes = []
for source_db in file_list:
    merge_processes.append("pgloader {} postgresql://dwm:password@recombine-test.ct0qrar1ezs6.us-west-1.rds.amazonaws.com/recombine_test".format(source_db))

# Set up the processing pool
pool = Pool()
pool.map(run_process, merge_processes)

processing_stop_time = time.time()
print("merge processing time: {} seconds".format(processing_stop_time-processing_start_time))
