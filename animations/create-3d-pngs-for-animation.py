import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np
import os
import shutil


CONVERTED_DATABASE_NAME = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite'
RT_LOWER = 200
RT_UPPER = 400

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
frames_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} order by frame_id ASC, scan ASC, mz ASC".format(FRAME_TYPE_MS1, RT_LOWER, RT_UPPER), db_conn)
db_conn.close()

print("loaded {} points from {}".format(len(frames_df), CONVERTED_DATABASE_NAME))

# set a filename, run the logistic model, and create the plot
gif_filename = '190719_Hela_Ecoli_1to1_01'
save_folder = '3d'
working_folder = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/tiles/{}/{}'.format(save_folder, gif_filename)
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

azimuth = 230
frame_counter = 0

for frame_id,frame_df in frames_df.groupby('frame_id'):
    print("rendering frame {}".format(frame_id))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_figheight(10)
    fig.set_figwidth(15)
    ax.patch.set_facecolor('silver')

    ax.elev = 20.0
    ax.azim = azimuth
    ax.dist = 10.0

    ax.set_xlim(left=200, right=1800)
    ax.set_ylim(bottom=900, top=0)
    ax.set_zlim(bottom=0, top=10000)

    # plt.gca().invert_yaxis()
    plt.xlabel('m/z')
    plt.ylabel('scan')

    ax.scatter(frame_df.mz, frame_df.scan, frame_df.intensity, s=2**2, c=np.log(frame_df.intensity), cmap=plt.get_cmap('cool'))
    fig.suptitle('frame {}, retention time (secs) {}'.format(frame_id, round(frame_df.iloc[0].retention_time_secs, 1)), fontsize=16, x=0.5, y=0.85)
    plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight')
    plt.close()

    frame_counter += 1
