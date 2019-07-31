import glob
import os
from PIL import Image
import sqlite3
import pandas as pd
import shutil

BASE_DIR = '/home/ubuntu/yolo-movie-rt-3000-3600'
INDIVIDUAL_TILES_DIR = '{}/individual-tiles'.format(BASE_DIR)
ANIMATION_FRAMES_DIR = '{}/animation-frames'.format(BASE_DIR)
VIDEO_DIR = '{}/video'.format(BASE_DIR)

TILE_START = 10
TILE_END = 70
CONVERTED_DATABASE = '/home/ubuntu/HeLa_20KInt-rt-3000-3600-denoised/HeLa_20KInt.sqlite'
RT_LOWER = 3000
RT_UPPER = 3600
MS1_CE = 10

# initialise the directories required
if os.path.exists(INDIVIDUAL_TILES_DIR):
    shutil.rmtree(INDIVIDUAL_TILES_DIR)
os.makedirs(INDIVIDUAL_TILES_DIR)

if os.path.exists(ANIMATION_FRAMES_DIR):
    shutil.rmtree(ANIMATION_FRAMES_DIR)
os.makedirs(ANIMATION_FRAMES_DIR)

if os.path.exists(VIDEO_DIR):
    shutil.rmtree(VIDEO_DIR)
os.makedirs(VIDEO_DIR)

db_conn = sqlite3.connect(CONVERTED_DATABASE)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(RT_LOWER, RT_UPPER, MS1_CE), db_conn)
db_conn.close()

for idx in range(len(ms1_frame_properties_df)):
    frame_id = int(ms1_frame_properties_df.iloc[idx].frame_id)
    file_list = []
    for tile_id in range(TILE_START, TILE_END+1):
        file_list.append(glob.glob("{}/frame-{}-tile-{}-mz-*.png".format(INDIVIDUAL_TILES_DIR, frame_id, tile_id))[0])
    images = list(map(Image.open, file_list))

    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save('{}/frame-{:04d}.png'.format(ANIMATION_FRAMES_DIR, idx))
