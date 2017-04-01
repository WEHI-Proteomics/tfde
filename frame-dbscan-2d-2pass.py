import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.cluster import DBSCAN
import time
from scipy import signal
import peakutils
import copy

# Process a whole frame using DBSCAN, and then plot only the area of interest.

FRAME_ID = 30000

# Peak of interest
LOW_MZ = 566.2
HIGH_MZ = 566.44
LOW_SCAN = 516
HIGH_SCAN = 579

THRESHOLD = 85

EPSILON = 2.5
MIN_POINTS_IN_CLUSTER = 4

# scaling factors derived from manual inspection of good peak spacing in the subset plots
SCALING_FACTOR_X = 50.0
SCALING_FACTOR_Y = 1.0

def bbox(points):
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    return a

# Split the provided bounding box at the given y point, and return two bounding boxes
def split_bbox_y(bbox, y):
    a = copy.deepcopy(bbox)
    b = copy.deepcopy(bbox)
    a[1,1] = y  # y2 = y (y becomes the lower limit for the top bbox)
    b[1,0] = y  # y1 = y (y becomes the upper limit for the bottom bbox)
    return a, b

def bbox_points(bbox):
    return {'x1':bbox[0,0], 'y1':bbox[1,0], 'x2':bbox[0,1], 'y2':bbox[1,1]}    


# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-th-{}-30000-30000.csv".format(THRESHOLD))

# Create a frame
frame_df = rows_df[(rows_df.frame == FRAME_ID)].sort_values(['scan', 'mz'], ascending=[True, True])

X_pretransform = frame_df[['mz','scan']].values

start = time.time()
X = np.copy(X_pretransform)
X[:,0] = X[:,0]*SCALING_FACTOR_X
X[:,1] = X[:,1]*SCALING_FACTOR_Y

db = DBSCAN(eps=EPSILON, min_samples=MIN_POINTS_IN_CLUSTER, n_jobs=-1).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

end = time.time()
print("elapsed time = {} sec".format(end-start))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Plot the area of interest
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

plt.title('Epsilon={}, samples={}, clusters={}'.format(EPSILON, MIN_POINTS_IN_CLUSTER, n_clusters_))

ax1.set_xlim(xmin=LOW_MZ, xmax=HIGH_MZ)
ax1.set_ylim(ymin=HIGH_SCAN, ymax=LOW_SCAN)
plt.xlabel('m/z')
plt.ylabel('scan')

ax2.set_ylim(ymin=HIGH_SCAN, ymax=LOW_SCAN)
plt.xlabel('intensity')
plt.ylabel('scan')

xy = X_pretransform[core_samples_mask]
ax1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)

xy = X_pretransform[~core_samples_mask]
ax1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='black', markeredgecolor='black', markeredgewidth=0.0, markersize=2)

# Process the clusters
mono_peaks_df = pd.DataFrame()  # where we record all the monoisotopic peaks we find
clusters = [X_pretransform[labels == i] for i in xrange(n_clusters_)]
for cluster in clusters:

    # find the bounding box of each cluster
    bb = bbox(cluster)
    x1 = bbox_points(bb)['x1']
    y1 = bbox_points(bb)['y1']
    x2 = bbox_points(bb)['x2']
    y2 = bbox_points(bb)['y2']

    # get the intensity values by scan
    cluster_df = frame_df[(frame_df.mz >= x1) & (frame_df.mz <= x2) & (frame_df.scan >= y1) & (frame_df.scan <= y2)]
    cluster_intensity = cluster_df[['scan','intensity']].values
    cluster_mz = cluster_df[['mz','scan']].values

    # filter the intensity with a Gaussian filter
    window = signal.gaussian(20, std=5)
    filtered = signal.convolve(cluster_intensity[:, 1], window, mode='same') / sum(window)

    # find the maxima
    indexes = peakutils.indexes(filtered, thres=0.1, min_dist=10)

    # ... and plot them
    for index in indexes:
        ax1.plot(cluster_mz[index,0], cluster_mz[index,1], 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)

    # Find the minimum intensity between the maxima
    if len(indexes) == 2:
        minimum_intensity_index = np.argmin(filtered[indexes[0]:indexes[1]+1])+indexes[0]
        ax1.plot(cluster_mz[minimum_intensity_index,0], cluster_mz[minimum_intensity_index,1], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        # split the bounding box at the minima
        a_peak, b_peak = split_bbox_y(bbox=bb, y=cluster_mz[minimum_intensity_index,1])
        # draw a bounding box around a_peak
        a_x1 = bbox_points(a_peak)['x1']
        a_y1 = bbox_points(a_peak)['y1']
        a_x2 = bbox_points(a_peak)['x2']
        a_y2 = bbox_points(a_peak)['y2']
        p = patches.Rectangle((a_x1, a_y1), a_x2-a_x1, a_y2-a_y1, fc = 'none', ec = 'green', linewidth=1)
        ax1.add_patch(p)
        # draw a bounding box around b_peak
        b_x1 = bbox_points(b_peak)['x1']
        b_y1 = bbox_points(b_peak)['y1']
        b_x2 = bbox_points(b_peak)['x2']
        b_y2 = bbox_points(b_peak)['y2']
        p = patches.Rectangle((b_x1, b_y1), b_x2-b_x1, b_y2-b_y1, fc = 'none', ec = 'green', linewidth=1)
        ax1.add_patch(p)
        # add the peaks to the collection
        mono_peaks_df = mono_peaks_df.append(bbox_points(a_peak), ignore_index=True)
        mono_peaks_df = mono_peaks_df.append(bbox_points(b_peak), ignore_index=True)
    else:
        # draw a bounding box around the peak
        p = patches.Rectangle((x1, y1), x2-x1, y2-y1, fc = 'none', ec = 'green', linewidth=1)
        # add the peak to the collection
        mono_peaks_df = mono_peaks_df.append(bbox_points(bb), ignore_index=True)

    # If this cluster is within the AOI, plot the intensity profile
    if ((x1 >= LOW_MZ) & (x2 <= HIGH_MZ) & (y1 >= LOW_SCAN) & (y2 <= HIGH_SCAN)):
        ax2.plot(filtered, cluster_mz[:,1], '-', color='orange', linewidth=1, markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=1)
        for index in indexes:
            ax2.plot(filtered[index], cluster_mz[index,1], 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        if len(indexes) == 2:
            minimum_intensity_index = np.argmin(filtered[indexes[0]:indexes[1]+1])+indexes[0]
            ax2.plot(filtered[minimum_intensity_index], cluster_mz[minimum_intensity_index,1], 'o', markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.0, markersize=6)

plt.show()
plt.close('all')
