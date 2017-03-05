import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.cluster import DBSCAN
import time

# Process a whole frame using DBSCAN, and then plot only the area of interest.

FRAME_ID = 30000

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

EPSILON = 0.00005
MIN_POINTS_IN_CLUSTER = 4

# scaling factors derived from manual inspection of good peak spacing in the subset plots
SCALING_FACTOR_X = 2500.0/5.0
SCALING_FACTOR_Y = 800.0/100.0

P = 0.5
C = 9.81

def bbox(points):
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    return a

def metric(p1, p2):
    num = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**P
    den = C*p1[2]*p2[2]
    return num/den

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-30000-30010.csv")

# Create a subset
frame = rows_df[(rows_df.frame == FRAME_ID)]

X_pretransform = frame[['mz','scan','intensity']].values

start = time.time()
X = np.copy(X_pretransform)
X[:,0] = X[:,0]*SCALING_FACTOR_X
X[:,1] = X[:,1]*SCALING_FACTOR_Y

db = DBSCAN(eps=EPSILON, min_samples=MIN_POINTS_IN_CLUSTER, metric=metric).fit(X)
# db = DBSCAN(eps=EPSILON, min_samples=MIN_POINTS_IN_CLUSTER).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print('labels {}'.format(np.shape(labels)))

end = time.time()
print("elapsed time = {} sec".format(end-start))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Plot the area of interest
fig = plt.figure()
axes = fig.add_subplot(111)

xy = X_pretransform[core_samples_mask]
axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)

xy = X_pretransform[~core_samples_mask]
axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor='black', markeredgecolor='black', markeredgewidth=0.0, markersize=2)

# draw a bounding box around each cluster
clusters = [X_pretransform[labels == i] for i in xrange(n_clusters_)]
for cluster in clusters:
    bb = bbox(cluster[:,0:2])  # take the first two columns (scan, mz)
    x1 = bb[0,0]
    y1 = bb[1,0]
    x2 = bb[0,1]
    y2 = bb[1,1]
    p = patches.Rectangle((x1, y1), x2-x1, y2-y1, fc = 'none', ec = 'green', linewidth=2)
    axes.add_patch(p)


plt.title('Epsilon={}, samples={}, clusters={}'.format(EPSILON, MIN_POINTS_IN_CLUSTER, n_clusters_))
ax = plt.gca()
ax.axis('tight')
ax.set_xlim(xmin=LOW_MZ, xmax=HIGH_MZ)
ax.set_ylim(ymin=HIGH_SCAN, ymax=LOW_SCAN)
plt.xlabel('m/z')
plt.ylabel('scan')
plt.tight_layout()
plt.show()
# fig.savefig('./images/frame-dbscan-eps-{}-sam-{}.png'.format(EPSILON, MIN_POINTS_IN_CLUSTER), pad_inches = 0.0, dpi='figure')
plt.close('all')
