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
save_folder = '3d-zoomed'
working_folder = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/tiles/{}/{}'.format(save_folder, gif_filename)
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

azimuth = 230
frame_counter = 0

# m/z zoom
from_mz_lower = 200
to_mz_lower = 705.6307692307693
from_mz_upper = 1800
to_mz_upper = 708.6175824175824

# scan zoom
from_scan_lower = 0
to_scan_lower = 577
from_scan_upper = 900
to_scan_upper = 705

number_of_seconds_to_zoom = 60
rt_to_finish_zoom = 205  # set this to the start of the feature
number_of_seconds_before_zoom = 20
number_of_seconds_after_zoom = 60

# how quickly to zoom in
levels_of_zoom = 100

# rt range to extract
rt_start_zoom = rt_to_finish_zoom - number_of_seconds_to_zoom
rt_lower = rt_start_zoom - number_of_seconds_before_zoom
rt_upper = rt_to_finish_zoom + number_of_seconds_after_zoom

print("rt lower {}, zoom start rt {}, zoom end rt {}, rt upper {}".format(round(rt_lower,1), round(rt_start_zoom,1), round(rt_to_finish_zoom,1), round(rt_upper,1)))

mz_lower_zoom_steps = np.linspace(from_mz_lower, to_mz_lower, levels_of_zoom)
mz_upper_zoom_steps = np.linspace(from_mz_upper, to_mz_upper, levels_of_zoom)
scan_lower_zoom_steps = np.linspace(from_scan_lower, to_scan_lower, levels_of_zoom)
scan_upper_zoom_steps = np.linspace(from_scan_upper, to_scan_upper, levels_of_zoom)

# load the points from the frame range
db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
frames_df = pd.read_sql_query("select frame_id,mz,scan,intensity,retention_time_secs from frames where frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {} order by frame_id ASC, scan ASC, mz ASC".format(FRAME_TYPE_MS1, rt_lower, rt_upper), db_conn)
db_conn.close()

print("loaded {} points from {}".format(len(frames_df), CONVERTED_DATABASE_NAME))

for frame_id,frame_df in frames_df.groupby('frame_id'):
    retention_time_secs = frame_df.iloc[0].retention_time_secs

    if retention_time_secs < rt_start_zoom:
        zoom_index = 0
    elif retention_time_secs > rt_to_finish_zoom:
        zoom_index = -1
    else:
        # how far into the zoom are we?
        zoom_index = int((retention_time_secs - rt_start_zoom) / number_of_seconds_to_zoom * levels_of_zoom)

    mz_lower = mz_lower_zoom_steps[zoom_index]
    mz_upper = mz_upper_zoom_steps[zoom_index]
    scan_lower = scan_lower_zoom_steps[zoom_index]
    scan_upper = scan_upper_zoom_steps[zoom_index]

    # filter out everything in the frame that is not in the plot
    frame_df = frame_df[(frame_df.mz >= mz_lower) & (frame_df.mz <= mz_upper) & (frame_df.scan >= scan_lower) & (frame_df.scan <= scan_upper)]

    print("rendering frame {} at zoom index {}".format(frame_id, zoom_index))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_figheight(10)
    fig.set_figwidth(15)
    ax.patch.set_facecolor('darkgray')
    ax.w_xaxis.set_pane_color((0.75, 0.75, 0.75, 0.8))

    ax.elev = 20.0
    ax.azim = azimuth
    ax.dist = 10.0

    ax.set_xlim(left=mz_lower, right=mz_upper)
    ax.set_ylim(bottom=scan_upper, top=scan_lower)
    ax.set_zlim(bottom=0, top=10000)

    # plt.gca().invert_yaxis()
    plt.xlabel('m/z')
    plt.ylabel('scan')

    ax.scatter(frame_df.mz, frame_df.scan, frame_df.intensity, s=2**2, c=np.log(frame_df.intensity), cmap=plt.get_cmap('cet_rainbow'))
    fig.suptitle('frame id {}, retention time (secs) {}'.format(frame_id, round(frame_df.iloc[0].retention_time_secs, 1)), fontsize=16, x=0.5, y=0.85)
    plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight')
    plt.close()

    frame_counter += 1
