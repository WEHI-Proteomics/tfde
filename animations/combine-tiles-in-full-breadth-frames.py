import glob
import os
from PIL import Image
import sqlite3
import pandas as pd
import shutil
import argparse

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-cdb','--converted_database', type=str, help='Path to the raw converted database.', required=True)
parser.add_argument('-tb','--tile_base', type=str, help='Path to the base directory of the tiles.', required=True)
parser.add_argument('-fd','--frame_directory', type=str, help='Path to the directory of the rendered frames.', required=True)
parser.add_argument('-rtl','--rt_lower', type=int, help='Lower bound of the RT range.', required=True)
parser.add_argument('-rtu','--rt_upper', type=int, help='Upper bound of the RT range.', required=True)
parser.add_argument('-tl','--tile_lower', type=int, help='Lower bound of the tile range.', required=True)
parser.add_argument('-tu','--tile_upper', type=int, help='Upper bound of the tile range.', required=True)
parser.add_argument('-x','--x_pixels', type=int, default=910, help='Resize to this x dimension.', required=False)
parser.add_argument('-y','--y_pixels', type=int, default=910, help='Resize to this y dimension.', required=False)
parser.add_argument('-pb','--with_prediction_boxes', action='store_true', help='Overlay the prediction boxes.')
args = parser.parse_args()

if args.with_prediction_boxes:
    INDIVIDUAL_TILES_DIR = '{}/overlay'.format(args.tile_base)
else:
    INDIVIDUAL_TILES_DIR = '{}/pre-assigned'.format(args.tile_base)

ANIMATION_FRAMES_DIR = args.frame_directory

TILE_START = args.tile_lower
TILE_END = args.tile_upper
CONVERTED_DATABASE = args.converted_database
RT_LOWER = args.rt_lower
RT_UPPER = args.rt_upper
MS1_CE = 10

# initialise the directories required
print("preparing directories")
if os.path.exists(ANIMATION_FRAMES_DIR):
    shutil.rmtree(ANIMATION_FRAMES_DIR)
os.makedirs(ANIMATION_FRAMES_DIR)

db_conn = sqlite3.connect(CONVERTED_DATABASE)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {}".format(RT_LOWER, RT_UPPER, MS1_CE), db_conn)
print("loaded {} frame ids".format(len(ms1_frame_properties_df)))
db_conn.close()

for idx in range(len(ms1_frame_properties_df)):
    frame_id = int(ms1_frame_properties_df.iloc[idx].frame_id)
    file_list = []
    for tile_id in range(TILE_START, TILE_END+1):
        print("processing frame {}, tile {}".format(frame_id, tile_id))
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

    new_im = new_im.resize((args.x_pixels, args.y_pixels))
    new_im.save('{}/frame-{:04d}.png'.format(ANIMATION_FRAMES_DIR, idx))

print()
print("wrote {} frames to {}".format(len(ms1_frame_properties_df), ANIMATION_FRAMES_DIR))
