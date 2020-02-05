import matplotlib
matplotlib.use("Agg")
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
save_folder = '2d-raw-points'
working_folder = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/tiles/{}/{}'.format(save_folder, gif_filename)
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

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
frames_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} and mz >= {} and mz <= {} and scan >= {} and scan <= {}".format(FRAME_TYPE_MS1, rt_lower, rt_upper, mz_lower, mz_upper, scan_lower, scan_upper), db_conn)
db_conn.close()

print("loaded {} points from {}".format(len(frames_df), CONVERTED_DATABASE_NAME))

frames_df['normalised_intensity'] = frames_df.intensity / frames_df.intensity.max()
frames_df.sort_values(by=['frame_id','mz'], ascending=True, inplace=True)

intensity_upper = 1.0

for frame_id,frame_df in frames_df.groupby('frame_id'):
    if len(frame_df) > 0:
        retention_time_secs = frame_df.iloc[0].retention_time_secs

        print("rendering frame {}".format(frame_counter))

        f, ax = plt.subplots()
        f.set_facecolor('darkgray')
        plt.scatter(frame_df.mz, frame_df.normalised_intensity, c=np.log2(frame_df.intensity), cmap=plt.get_cmap('cet_rainbow'), alpha=0.70, edgecolors='face')
        plt.xlabel('m/z', fontsize=20)
        plt.ylabel('normalised intensity', fontsize=20)
        plt.tick_params(labelsize=18)
        ax.patch.set_facecolor('silver')

        plt.xlim((mz_lower,mz_upper))
        plt.ylim((0,intensity_upper))

        #removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        f.set_figheight(15)
        f.set_figwidth(15)

        plt.margins(0.06)
        # plt.suptitle('Peaks in the area predicted for sequence {}, charge {}'.format(sequence_name, sequence_charge))

        plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight', pad_inches=1.0, facecolor=f.get_facecolor())
        plt.close()

        frame_counter += 1
