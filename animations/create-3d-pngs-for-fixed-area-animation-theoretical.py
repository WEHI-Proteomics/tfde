import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

#
# This program uses Matplotlib 3D plots to generate a synthetic peptide's isotopic peak series to show its shape in each dimension
#

# Gaussian function
def func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

# azimuth values by scene
azimuth_d = {}
azimuth_d[1] = 230
azimuth_d[2] = 230
azimuth_d[3] = np.linspace(230, 180, num=50)
azimuth_d[4] = 180
azimuth_d[5] = np.linspace(180, 270, num=100)

# elevation values by scene
elevation_d = {}
elevation_d[1] = 20
elevation_d[2] = 20
elevation_d[3] = np.linspace(20, 1, num=len(azimuth_d[3]))  # keyed to the number of azimuth values in the scene
elevation_d[4] = 1
elevation_d[5] = 1


def calculate_azimuth(scene, frame):
    if type(azimuth_d[scene]) == int:
        azimuth = azimuth_d[scene]
    else:
        azimuth = azimuth_d[scene][frame]
    return azimuth

def calculate_elevation(scene, frame):
    if type(elevation_d[scene]) == int:
        elevation = elevation_d[scene]
    else:
        elevation = elevation_d[scene][frame]
    return elevation

def calculate_feature_intensity(scene, frame):
    if scene==1 or scene==2:
        intensity = feature_intensity_values[frame]
    else:
        intensity = 1.0
    return intensity


# set a filename, run the logistic model, and create the plot
working_folder = '/Users/darylwilding-mcbride/Downloads/peptide/frames'
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

frame_counter = 0

# m/z extent
mz_lower = 698
mz_upper = 704

# CCS extent
scan_lower = 530
scan_upper = 770

peak_ccs = ((scan_upper - scan_lower) / 2) + scan_lower
std_dev_ccs = 20

# RT extent
rt_lower = 200
rt_upper = 300

# marker size
marker_size = 1**2

rt_values = np.arange(rt_lower, rt_upper, 0.8)
ccs_values = np.arange(scan_lower, scan_upper, 1.0)

peak_rt = ((rt_upper-rt_lower)/2)+rt_lower
std_dev_rt = 13

feature_intensity_values = func(rt_values, 1.0, peak_rt, std_dev_rt)
intensity_threshold = 0.001

# set up the number of frames in each scene
scenes = [len(rt_values), int(len(rt_values)/2), len(azimuth_d[3]), 40, len(azimuth_d[5])]

for scene_idx,number_of_frames in enumerate(scenes):
    for frame_id in range(number_of_frames):
        print("rendering frame {}".format(frame_counter))

        fig = plt.figure()
        fig.set_facecolor('whitesmoke')
        ax = fig.add_subplot(111, projection='3d')
        fig.set_figheight(5)
        fig.set_figwidth(5)
        ax.patch.set_facecolor('whitesmoke')
        ax.w_xaxis.set_pane_color((0.3, 0.3, 0.3, 0.8))
        ax.w_yaxis.set_pane_color((0.3, 0.3, 0.3, 0.8))
        ax.w_zaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))

        ax.elev = calculate_elevation(scene=scene_idx+1, frame=frame_id)
        ax.azim = calculate_azimuth(scene=scene_idx+1, frame=frame_id)
        ax.dist = 9.0

        ax.set_xlim(left=mz_lower, right=mz_upper)
        ax.set_ylim(bottom=scan_upper, top=scan_lower)
        ax.set_zlim(bottom=0, top=1.1)

        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

        plt.xlabel('m/z', fontsize=12)
        plt.ylabel('CCS', fontsize=12)
        plt.tick_params(labelsize=10)

        mz = 700.0
        mz_values = np.zeros((len(ccs_values),), dtype=float)+mz

        feature_intensity = calculate_feature_intensity(scene=scene_idx+1, frame=frame_id)

        iso_intensity_values = feature_intensity*1.0
        x = mz_values
        y = ccs_values
        z = func(y, iso_intensity_values, peak_ccs, std_dev_ccs)
        idx = z > intensity_threshold
        ax.scatter(x[idx], y[idx], z[idx], s=marker_size, c=z[idx], cmap=plt.get_cmap('cool'), alpha=1.0)

        iso_intensity_values = feature_intensity*0.8
        nonzero_idx = iso_intensity_values > intensity_threshold
        x = mz_values+0.5
        y = ccs_values
        z = func(y, iso_intensity_values, peak_ccs, std_dev_ccs)
        idx = z > intensity_threshold
        ax.scatter(x[idx], y[idx], z[idx], s=marker_size, c=z[idx], cmap=plt.get_cmap('cool'), alpha=1.0)

        iso_intensity_values = feature_intensity*0.4
        nonzero_idx = iso_intensity_values > intensity_threshold
        x = mz_values+1.0
        y = ccs_values
        z = func(y, iso_intensity_values, peak_ccs, std_dev_ccs)
        idx = z > intensity_threshold
        ax.scatter(x[idx], y[idx], z[idx], s=marker_size, c=z[idx], cmap=plt.get_cmap('cool'), alpha=1.0)

        iso_intensity_values = feature_intensity*0.2
        nonzero_idx = iso_intensity_values > intensity_threshold
        x = mz_values+1.5
        y = ccs_values
        z = func(y, iso_intensity_values, peak_ccs, std_dev_ccs)
        idx = z > intensity_threshold
        ax.scatter(x[idx], y[idx], z[idx], s=marker_size, c=z[idx], cmap=plt.get_cmap('cool'), alpha=1.0)

        iso_intensity_values = feature_intensity*0.1
        nonzero_idx = iso_intensity_values > intensity_threshold
        x = mz_values+2.0
        y = ccs_values
        z = func(y, iso_intensity_values, peak_ccs, std_dev_ccs)
        idx = z > intensity_threshold
        ax.scatter(x[idx], y[idx], z[idx], s=marker_size, c=z[idx], cmap=plt.get_cmap('cool'), alpha=1.0)

        # plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.savefig('{}/img-{:04d}.tiff'.format(working_folder, frame_counter), bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=600)
        plt.close()

        frame_counter += 1
