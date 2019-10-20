import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np
import os
import shutil


CONVERTED_DATABASE_NAME = '/Users/darylwilding-mcbride/Downloads/experiments/190719_Hela_Ecoli/converted-databases/190719_Hela_Ecoli_1to3_06-converted.sqlite'
RT_LOWER = 200
RT_UPPER = 400
MS1_COLLISION_ENERGY = 10

db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
ms1_frame_properties_df = pd.read_sql_query("select frame_id,retention_time_secs from frame_properties where retention_time_secs >= {} and retention_time_secs <= {} and collision_energy == {} order by retention_time_secs".format(RT_LOWER, RT_UPPER, MS1_COLLISION_ENERGY), db_conn)
ms1_frame_ids = tuple(ms1_frame_properties_df.frame_id)
db_conn.close()

db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
frames_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id in {} order by frame_id ASC, scan ASC, mz ASC".format(ms1_frame_ids), db_conn)
db_conn.close()

# set a filename, run the logistic model, and create the plot
gif_filename = '190719_Hela_Ecoli_1to3_06'
save_folder = '3d'
working_folder = '/Users/darylwilding-mcbride/Downloads/experiments/190719_Hela_Ecoli/tiles/{}/{}'.format(save_folder, gif_filename)
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

azimuth = 230

for frame_idx,frame_id in enumerate(ms1_frame_ids):
    frame_df = frames_df[frames_df.frame_id==frame_id]

    print("rendering frame {} of {}".format(frame_idx+1, len(ms1_frame_ids)))

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
    fig.suptitle('Frame {}'.format(frame_id), fontsize=16, x=0.5, y=0.85)
    plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_idx), bbox_inches='tight')
    plt.close()
