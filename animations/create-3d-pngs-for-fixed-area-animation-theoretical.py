import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np
import os
import shutil

#
# This program uses Matplotlib 3D plots to generate a synthetic peptide's isotopic peak series to show its shape in each dimension
#

# Gaussian function
def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

# set a filename, run the logistic model, and create the plot
working_folder = '/Users/darylwilding-mcbride/Downloads/peptide/frames'
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

azimuth = 230
frame_counter = 0

# m/z extent
mz_lower = 695
mz_upper = 706

# CCS extent
scan_lower = 500
scan_upper = 800

# RT extent
rt_lower = 200
rt_upper = 300


rt_values = np.arange(rt_lower, rt_upper, 1.0)
ccs_values = np.arange(scan_lower, scan_upper, 1.0)

peak_rt = ((rt_upper-rt_lower)/2)+rt_lower
feature_intensity_values = func(rt_values, 1.0, peak_rt, 5.0)

for frame_id,rt in enumerate(rt_values):
    print("rendering frame {}".format(frame_counter))

    fig = plt.figure()
    fig.set_facecolor('whitesmoke')
    ax = fig.add_subplot(111, projection='3d')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax.patch.set_facecolor('whitesmoke')
    ax.w_xaxis.set_pane_color((0.96, 0.96, 0.96, 0.8))
    ax.w_yaxis.set_pane_color((0.96, 0.96, 0.96, 0.8))
    ax.w_zaxis.set_pane_color((0.96, 0.96, 0.96, 0.8))

    ax.elev = 20.0
    ax.azim = azimuth
    ax.dist = 9.0

    ax.set_xlim(left=mz_lower, right=mz_upper)
    ax.set_ylim(bottom=scan_upper, top=scan_lower)
    ax.set_zlim(bottom=0, top=1.0)

    ax.set_zticks([])

    # plt.gca().invert_yaxis()
    plt.xlabel('m/z', fontsize=18)
    plt.ylabel('CCS', fontsize=18)
    # ax.set_zlabel('normalised intensity', fontsize=20)
    plt.tick_params(labelsize=12)

    mz = 700.0
    mz_values = np.zeros((len(ccs_values),), dtype=float)+mz

    peak_ccs = ((scan_upper - scan_lower) / 2) + scan_lower
    std_dev = 10

    iso_1 = func(ccs_values, feature_intensity_values[frame_id]*1.0, peak_ccs, std_dev)
    iso_2 = func(ccs_values, feature_intensity_values[frame_id]*0.6, peak_ccs, std_dev)
    iso_3 = func(ccs_values, feature_intensity_values[frame_id]*0.4, peak_ccs, std_dev)
    iso_4 = func(ccs_values, feature_intensity_values[frame_id]*0.2, peak_ccs, std_dev)
    iso_5 = func(ccs_values, feature_intensity_values[frame_id]*0.1, peak_ccs, std_dev)

    ax.scatter(mz_values, ccs_values, iso_1, s=4**2, c=iso_1, cmap=plt.get_cmap('cool'), alpha=1.0)
    ax.scatter(mz_values+0.5, ccs_values, iso_2, s=4**2, c=iso_2, cmap=plt.get_cmap('cool'), alpha=1.0)
    ax.scatter(mz_values+1.0, ccs_values, iso_3, s=4**2, c=iso_3, cmap=plt.get_cmap('cool'), alpha=1.0)
    ax.scatter(mz_values+1.5, ccs_values, iso_4, s=4**2, c=iso_4, cmap=plt.get_cmap('cool'), alpha=1.0)
    ax.scatter(mz_values+2.0, ccs_values, iso_5, s=4**2, c=iso_5, cmap=plt.get_cmap('cool'), alpha=1.0)

    # fig.suptitle('frame id {}, retention time (secs) {}'.format(frame_id, round(frame_df.iloc[0].retention_time_secs, 1)), fontsize=16, x=0.5, y=0.85)
    plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    frame_counter += 1
