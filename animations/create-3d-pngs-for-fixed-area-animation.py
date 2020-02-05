import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np
import os
import shutil
import colorcet as cc

CONVERTED_DATABASE_NAME = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite'
RT_LOWER = 100
RT_UPPER = 300

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# set a filename, run the logistic model, and create the plot
gif_filename = '190719_Hela_Ecoli_1to1_01'
save_folder = '3d-fixed-area'
working_folder = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/tiles/{}/{}'.format(save_folder, gif_filename)
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

azimuth = 230
frame_counter = 0

# m/z extent
mz_lower = 695
mz_upper = 740

# CCS extent
scan_lower = 500
scan_upper = 800

# RT extent
rt_lower = 200
rt_upper = 300

# load the points from the frame range
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
frames_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} and mz >= {} and mz <= {} and scan >= {} and scan <= {} order by frame_id ASC, scan ASC, mz ASC".format(FRAME_TYPE_MS1, rt_lower, rt_upper, mz_lower, mz_upper, scan_lower, scan_upper), db_conn)
db_conn.close()

print("loaded {} points from {}".format(len(frames_df), CONVERTED_DATABASE_NAME))

frames_df['normalised_intensity'] = frames_df.intensity / frames_df.intensity.max()

for frame_id,frame_df in frames_df.groupby('frame_id'):
    if len(frame_df) > 0:
        retention_time_secs = frame_df.iloc[0].retention_time_secs

        print("rendering frame {}".format(frame_counter))

        fig = plt.figure()
        fig.set_facecolor('darkgray')
        ax = fig.add_subplot(111, projection='3d')
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax.patch.set_facecolor('darkgray')
        ax.w_xaxis.set_pane_color((0.75, 0.75, 0.75, 0.8))

        ax.elev = 20.0
        ax.azim = azimuth
        ax.dist = 9.0

        ax.set_xlim(left=mz_lower, right=mz_upper)
        ax.set_ylim(bottom=scan_upper, top=scan_lower)
        ax.set_zlim(bottom=0, top=1.0)

        # plt.gca().invert_yaxis()
        plt.xlabel('m/z', fontsize=20)
        plt.ylabel('CCS', fontsize=20)
        ax.set_zlabel('normalised intensity', fontsize=20)
        plt.tick_params(labelsize=18)

        ax.scatter(frame_df.mz, frame_df.scan, frame_df.normalised_intensity, s=2**2, c=np.log2(frame_df.intensity), cmap=plt.get_cmap('cet_rainbow'))
        # fig.suptitle('frame id {}, retention time (secs) {}'.format(frame_id, round(frame_df.iloc[0].retention_time_secs, 1)), fontsize=16, x=0.5, y=0.85)
        plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        frame_counter += 1
