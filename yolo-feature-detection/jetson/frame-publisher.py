import sqlite3
import pandas as pd
import numpy as np
import sys
import argparse
import os, shutil
import time

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

CONVERTED_DATABASE_NAME = '/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite'
PUBLISHED_FRAMES_DIR = '/data/published-frames'

if os.path.exists(PUBLISHED_FRAMES_DIR):
    shutil.rmtree(PUBLISHED_FRAMES_DIR)
os.makedirs(PUBLISHED_FRAMES_DIR)

# create indexes
print("creating indexes on {}".format(CONVERTED_DATABASE_NAME))
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
src_c = db_conn.cursor()
src_c.execute("create index if not exists idx_frame_publisher_1 on frames (frame_id,scan)")
db_conn.close()

# load the ms1 frame IDs
print("load the frame ids")
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select Id,Time from frame_properties where MsMsType == {} order by Time".format(FRAME_TYPE_MS1), db_conn)
ms1_frame_ids = tuple(ms1_frame_properties_df.Id)
db_conn.close()

# publish the frames
print("load the frame ids")
for frame_idx,frame_id in enumerate(ms1_frame_ids[:50]):
    print("frame id {} ({} of {})".format(frame_id, frame_idx+1, len(ms1_frame_ids)))
    # load the frame's data
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    frame_df = pd.read_sql_query("select * from frames where frame_id == {} order by scan".format(frame_id), db_conn)
    db_conn.close()

    # save it to a file
    frame_file_name = '{}/{}'.format(PUBLISHED_FRAMES_DIR, 'frame-{}.pkl'.format(frame_id))
    frame_df.to_pickle(frame_file_name)

    # post a message with some metadata
    