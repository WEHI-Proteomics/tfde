import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.cluster import DBSCAN
import time
from scipy import signal
import peakutils

# Process slices of a whole frame using DBSCAN, and then plot only the area of interest.

# FRAME_ID = 30000

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

IMAGE_X_PIXELS = 1600
IMAGE_Y_PIXELS = 900

EPSILON = 16.5
MIN_POINTS_IN_CLUSTER = 4

# scaling factors derived from manual inspection of good peak spacing in the subset plots
SCALING_FACTOR_X = 2500.0/5.0
SCALING_FACTOR_Y = 800.0/100.0

def bbox(points):
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    return a


# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-30000-30010.csv")

for frame_id in range(30000, 30011):

    print('Generating frame {}'.format(frame_id))

    # Create a slice
    slice_df = rows_df[(rows_df.frame == frame_id) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)].sort_values(['scan', 'mz'], ascending=[True, True])

    X_pretransform = slice_df[['mz','scan']].values

    X = np.copy(X_pretransform)
    X[:,0] = X[:,0]*SCALING_FACTOR_X
    X[:,1] = X[:,1]*SCALING_FACTOR_Y

    db = DBSCAN(eps=EPSILON, min_samples=MIN_POINTS_IN_CLUSTER, n_jobs=-1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Plot the area of interest
    fig = plt.figure()
    dpi = fig.get_dpi()
    fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))
    axes = fig.add_subplot(111)
    axes.set_xlim(xmin=LOW_MZ, xmax=HIGH_MZ)
    axes.set_ylim(ymin=HIGH_SCAN, ymax=LOW_SCAN)
    plt.xlabel('m/z')
    plt.ylabel('scan')

    # # Plot the intensity profile
    # fig2 = plt.figure()
    # plt.title('Intensity profile')
    # axes2 = fig2.add_subplot(111)
    # # axes2.set_xlim(xmin=LOW_MZ, xmax=HIGH_MZ)
    # axes2.set_ylim(ymin=HIGH_SCAN, ymax=LOW_SCAN)
    # plt.xlabel('intensity')
    # plt.ylabel('scan')

    xy = X_pretransform[core_samples_mask]
    axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)

    xy = X_pretransform[~core_samples_mask]
    axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='black', markeredgecolor='black', markeredgewidth=0.0, markersize=2)

    # Process the clusters
    clusters = [X_pretransform[labels == i] for i in xrange(n_clusters_)]
    for cluster in clusters:

        # find the bounding box of each cluster
        bb = bbox(cluster)
        x1 = bb[0,0]
        y1 = bb[1,0]
        x2 = bb[0,1]
        y2 = bb[1,1]

        # draw a bounding box around each cluster
        p = patches.Rectangle((x1, y1), x2-x1, y2-y1, fc = 'none', ec = 'green', linewidth=1)
        axes.add_patch(p)

        # get the intensity values by scan
        cluster_df = slice_df[(slice_df.mz >= x1) & (slice_df.mz <= x2) & (slice_df.scan >= y1) & (slice_df.scan <= y2)]
        cluster_intensity = cluster_df[['scan','intensity']].values
        cluster_mz = cluster_df[['mz','scan']].values

        # filter the intensity with a Gaussian filter
        window = signal.gaussian(20, std=5)
        filtered = signal.convolve(cluster_intensity[:, 1], window, mode='same') / sum(window)
        indexes = peakutils.indexes(filtered, thres=0.2, min_dist=10)
        # Plot the maxmima for this cluster
        for index in indexes:
            axes.plot(cluster_mz[index,0], cluster_mz[index,1], 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        # Find the minimum intensity between the maxima
        if len(indexes) == 2:
            minimum_intensity_index = np.argmin(filtered[indexes[0]:indexes[1]+1])+indexes[0]
            axes.plot(cluster_mz[minimum_intensity_index,0], cluster_mz[minimum_intensity_index,1], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)

        # # If this cluster is within the AOI, plot the intensity profile
        # if ((x1 >= LOW_MZ) & (x2 <= HIGH_MZ) & (y1 >= LOW_SCAN) & (y2 <= HIGH_SCAN)):
        #     axes2.plot(filtered, cluster_mz[:,1], '-', color='orange', linewidth=1, markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=1)
        #     for index in indexes:
        #         axes2.plot(filtered[index], cluster_mz[index,1], 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        #     if len(indexes) == 2:
        #         minimum_intensity_index = np.argmin(filtered[indexes[0]:indexes[1]+1])+indexes[0]
        #         axes2.plot(filtered[minimum_intensity_index], cluster_mz[minimum_intensity_index,1], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)

    # ax = plt.gca()
    # ax.axis('tight')
    # plt.tight_layout()
    # # plt.show()
    fig.suptitle('Frame {:0>4}'.format(frame_id), fontsize=20)
    fig.savefig('./frames/frame-{:0>4}.png'.format(frame_id), pad_inches = 0.0, dpi='figure')
