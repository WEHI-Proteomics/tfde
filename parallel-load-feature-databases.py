
# coding: utf-8

# In[1]:


import glob
import sqlite3
import pandas as pd
import os
import multiprocessing as mp
from multiprocessing import Pool


# In[2]:


def run_process(process):
    print("Executing: {}".format(process))
    os.system(process)


# In[3]:


feature_database_name = "./UPS2_allion_Slot1-2_01_699/UPS2_allion_Slot1-2_01_699.sqlite"


# In[4]:


SOURCE_DATABASE_DIRECTORY = "./UPS2_allion_Slot1-2_01_699"
file_list = glob.glob("{}/UPS2_allion_Slot1-2_01_699-features-*-*.sqlite".format(SOURCE_DATABASE_DIRECTORY))


# In[5]:


merge_processes = []
for source_db in file_list:
    merge_processes.append("pgloader {} postgresql://dwm:password@recombine-test.ct0qrar1ezs6.us-west-1.rds.amazonaws.com/recombine_test".format(source_db))


# In[ ]:


# Set up the processing pool
pool = Pool()
pool.map(run_process, merge_processes)

