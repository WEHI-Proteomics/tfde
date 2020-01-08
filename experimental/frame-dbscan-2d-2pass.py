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

EPSILON_PEAK = 2.5
MIN_POINTS_IN_PEAK = 4

EPSILON_CLUSTER = 2.0
MIN_POINTS_IN_CLUSTER = 2

# scaling factors derived from manual inspection of good peak spacing in the subset plots
SCALING_FACTOR_PEAKS_X = 50.0
SCALING_FACTOR_PEAKS_Y = 1.0

SCALING_FACTOR_CLUSTERS_X = 1.0
SCALING_FACTOR_CLUSTERS_Y = 0.2

def bbox(points):
    # print("points {}".format(points))
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    # print("bb {}".format(a))
    return a

# Split the provided bounding box at the given y point, and return two bounding boxes
def split_bbox_y(bbox, y):
    a = copy.deepcopy(bbox)
    b = copy.deepcopy(bbox)
    a[1,1] = y  # y2 = y (y becomes the lower limit for the top bbox)
    b[1,0] = y  # y1 = y (y becomes the upper limit for the bottom bbox)
    return a, b

def bbox_points(bbox):
    return {'tl':{'x':bbox[0,0], 'y':bbox[1,0]}, 'br':{'x':bbox[0,1], 'y':bbox[1,1]}}

def bbox_centroid(bbox):
    return {'x':bbox[0,0]+(bbox[0,1]-bbox[0,0])/2, 'y':bbox[1,0]+(bbox[1,1]-bbox[1,0])/2}


# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-th-{}-30000-30000.csv".format(THRESHOLD))

# Create a frame
frame_df = rows_df[(rows_df.frame == FRAME_ID)].sort_values(['scan', 'mz'], ascending=[True, True])

X_pretransform = frame_df[['mz','scan']].values

start = time.time()
X = np.copy(X_pretransform)
X[:,0] = X[:,0]*SCALING_FACTOR_PEAKS_X
X[:,1] = X[:,1]*SCALING_FACTOR_PEAKS_Y

db = DBSCAN(eps=EPSILON_PEAK, min_samples=MIN_POINTS_IN_PEAK, n_jobs=-1).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

end = time.time()
print("elapsed time = {} sec".format(end-start))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Plot the area of interest
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f = plt.figure()
ax1 = f.add_subplot(111)

plt.title('Epsilon={}, samples={}, clusters={}'.format(EPSILON_PEAK, MIN_POINTS_IN_PEAK, n_clusters_))

ax1.set_xlim(xmin=LOW_MZ, xmax=HIGH_MZ)
ax1.set_ylim(ymin=HIGH_SCAN, ymax=LOW_SCAN)
plt.xlabel('m/z')
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
    x1 = bbox_points(bb)['tl']['x']
    y1 = bbox_points(bb)['tl']['y']
    x2 = bbox_points(bb)['br']['x']
    y2 = bbox_points(bb)['br']['y']

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
        a_x1 = bbox_points(a_peak)['tl']['x']
        a_y1 = bbox_points(a_peak)['tl']['y']
        a_x2 = bbox_points(a_peak)['br']['x']
        a_y2 = bbox_points(a_peak)['br']['y']
        p = patches.Rectangle((a_x1, a_y1), a_x2-a_x1, a_y2-a_y1, fc = 'none', ec = 'green', linewidth=1)
        ax1.add_patch(p)
        # draw a bounding box around b_peak
        b_x1 = bbox_points(b_peak)['tl']['x']
        b_y1 = bbox_points(b_peak)['tl']['y']
        b_x2 = bbox_points(b_peak)['br']['x']
        b_y2 = bbox_points(b_peak)['br']['y']
        p = patches.Rectangle((b_x1, b_y1), b_x2-b_x1, b_y2-b_y1, fc = 'none', ec = 'green', linewidth=1)
        ax1.add_patch(p)
        # plot the centroids
        ax1.plot(bbox_centroid(a_peak)['x'], bbox_centroid(a_peak)['y'], 'D', markerfacecolor='magenta', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        ax1.plot(bbox_centroid(b_peak)['x'], bbox_centroid(b_peak)['y'], 'D', markerfacecolor='magenta', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        # add the peaks to the collection
        mono_peaks_df = mono_peaks_df.append({'peak':bbox_points(a_peak), 'centroid':(bbox_centroid(a_peak))}, ignore_index=True)
        mono_peaks_df = mono_peaks_df.append({'peak':bbox_points(b_peak), 'centroid':(bbox_centroid(b_peak))}, ignore_index=True)
    else:
        # draw a bounding box around the peak
        p = patches.Rectangle((x1, y1), x2-x1, y2-y1, fc = 'none', ec = 'green', linewidth=1)
        ax1.add_patch(p)
        # plot the centroid
        ax1.plot(bbox_centroid(bb)['x'], bbox_centroid(bb)['y'], 'D', markerfacecolor='magenta', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
        # add the peak to the collection
        mono_peaks_df = mono_peaks_df.append({'peak':bbox_points(bb), 'centroid':(bbox_centroid(bb))}, ignore_index=True)

# # Cluster the peaks we found
# Y_pretransform = pd.concat([mono_peaks_df, pd.DataFrame((d for idx, d in mono_peaks_df['centroid'].iteritems()))], axis=1)[['x','y']].values
# peak_bbs_tl = pd.concat([mono_peaks_df, pd.DataFrame((d['tl'] for idx, d in mono_peaks_df['peak'].iteritems()))], axis=1)[['x','y']].values
# peak_bbs_br = pd.concat([mono_peaks_df, pd.DataFrame((d['br'] for idx, d in mono_peaks_df['peak'].iteritems()))], axis=1)[['x','y']].values

# Y = np.copy(Y_pretransform)
# Y[:,0] = Y[:,0]*SCALING_FACTOR_CLUSTERS_X
# Y[:,1] = Y[:,1]*SCALING_FACTOR_CLUSTERS_Y

# db = DBSCAN(eps=EPSILON_CLUSTER, min_samples=MIN_POINTS_IN_CLUSTER, n_jobs=-1).fit(Y)
# labels = db.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print("found {} clusters".format(n_clusters_))
# c1 = (peak_bbs_tl[labels == i] for i in xrange(n_clusters_))
# c2 = (peak_bbs_br[labels == i] for i in xrange(n_clusters_))
# clusters = [np.concatenate((e1, e2), axis=0) for e1, e2 in zip(c1, c2)]
# for cluster in clusters:
#     # find the bounding box of each cluster
#     bb = bbox(cluster)
#     x1 = bbox_points(bb)['tl']['x']
#     y1 = bbox_points(bb)['tl']['y']
#     x2 = bbox_points(bb)['br']['x']
#     y2 = bbox_points(bb)['br']['y']
#     # draw a bounding box around the cluster
#     p = patches.Rectangle((x1, y1), x2-x1, y2-y1, fc = 'none', ec = 'blue', linewidth=1)
#     ax1.add_patch(p)


# Show the final plot
plt.show()
plt.close('all')
