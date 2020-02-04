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
import peakutils

# the width to use for intensity descent, in m/z
MS1_PEAK_DELTA = 0.1

# ms1_peaks_a is a numpy array of [mz,intensity]
# returns a numpy array of [mz_centroid,summed_intensity]
def ms1_intensity_descent(ms1_peaks_a):
    # intensity descent
    ms1_peaks_l = []
    while len(ms1_peaks_a) > 0:
        # find the most intense point
        max_intensity_index = np.argmax(ms1_peaks_a[:,1])
        peak_mz = ms1_peaks_a[max_intensity_index,0]
        peak_mz_lower = peak_mz - MS1_PEAK_DELTA
        peak_mz_upper = peak_mz + MS1_PEAK_DELTA

        # get all the raw points within this m/z region
        peak_indexes = np.where((ms1_peaks_a[:,0] >= peak_mz_lower) & (ms1_peaks_a[:,0] <= peak_mz_upper))[0]
        if len(peak_indexes) > 0:
            mz_cent = peakutils.centroid(ms1_peaks_a[peak_indexes,0], ms1_peaks_a[peak_indexes,1])
            summed_intensity = ms1_peaks_a[peak_indexes,1].sum()
            ms1_peaks_l.append((mz_cent, summed_intensity))
            # remove the raw points assigned to this peak
            ms1_peaks_a = np.delete(ms1_peaks_a, peak_indexes, axis=0)
    return np.array(ms1_peaks_l)


CONVERTED_DATABASE_NAME = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite'
RT_LOWER = 100
RT_UPPER = 300

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

# set a filename, run the logistic model, and create the plot
gif_filename = '190719_Hela_Ecoli_1to1_01'
save_folder = '2d-stem-plot'
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

        ms1_peaks_a = ms1_intensity_descent(frame_df[['mz','normalised_intensity']].to_numpy())

        print("rendering frame {}".format(frame_counter))

        f, ax = plt.subplots()
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        markerline, stemlines, baseline = ax.stem(ms1_peaks_a[:,0], ms1_peaks_a[:,1], 'g', use_line_collection=True)
        plt.setp(markerline, 'color', colors[2])
        plt.setp(stemlines, 'color', colors[2])
        plt.setp(baseline, 'color', colors[7])

        plt.xlabel('m/z centroid')
        plt.ylabel('summed intensity')

        plt.xlim((mz_lower,mz_upper))
        plt.ylim((0,40))

        f.set_figheight(10)
        f.set_figwidth(15)

        plt.margins(0.06)
        # plt.suptitle('Peaks in the area predicted for sequence {}, charge {}'.format(sequence_name, sequence_charge))

        plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight')
        plt.close()

        frame_counter += 1
