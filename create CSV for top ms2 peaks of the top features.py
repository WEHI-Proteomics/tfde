
# coding: utf-8

# In[ ]:


import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import json
import os


# Write out the top peaks for the top features

# In[ ]:


DB_NAME = '/home/ubuntu/UPS2_allion/UPS2_allion-features-1-1097.sqlite'
CONV_DB_NAME = '/home/ubuntu/UPS2_allion/UPS2_allion.sqlite'

NUMBER_OF_TOP_FEATURES = 1000
NUMBER_OF_TOP_PEAKS = 1000

# get the 200 top-matching ms2 peaks for the top 200 features
db_conn = sqlite3.connect(CONV_DB_NAME)
top_features_df = pd.read_sql_query("select feature_id from features order by feature_id ASC limit {}".format(NUMBER_OF_TOP_FEATURES), db_conn)
db_conn.close()


# In[ ]:


db_conn = sqlite3.connect(DB_NAME)
filename = "~/top_{}_peaks_for_top_{}_features.csv".format(NUMBER_OF_TOP_PEAKS, NUMBER_OF_TOP_FEATURES)
if os.path.isfile(filename):
    os.remove(filename)
    
for idx in range(len(top_features_df)):
    feature_id = top_features_df.loc[idx].feature_id
    print("feature ID {}".format(feature_id))
    df_1 = pd.read_sql_query("select feature_id,peak_id,centroid_mz from ms2_peaks where feature_id || '-' || peak_id in (select feature_id || '-' || ms2_peak_id from peak_correlation where feature_id == {} and abs(rt_distance) <= {} and abs(scan_distance) <= {} order by ms2_peak_id limit {})".format(feature_id, MAX_RT_DISTANCE, MAX_SCAN_DISTANCE, NUMBER_OF_TOP_PEAKS), db_conn)
    df_2 = pd.read_sql_query("select * from peak_correlation where feature_id=={} and abs(rt_distance) <= {} and abs(scan_distance) <= {} order by ms2_peak_id limit {}".format(feature_id, MAX_RT_DISTANCE, MAX_SCAN_DISTANCE, NUMBER_OF_TOP_PEAKS), db_conn)
    df = pd.merge(df_1, df_2, left_on=['feature_id','peak_id'], right_on=['feature_id','ms2_peak_id'])
    df.drop(['peak_id','correlation'], inplace=True, axis=1)
    # write the CSV
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', sep=',', index=False, header=False)
    else:
        df.to_csv(filename, mode='a', sep=',', index=False, header=True)

db_conn.close()

