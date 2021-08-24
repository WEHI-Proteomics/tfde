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

# FRAME_ID = 708
# LOW_MZ = 000.0
# HIGH_MZ = 3000.0
# LOW_SCAN = 1
# HIGH_SCAN = 350


EPSILON = 2.5
MIN_POINTS_IN_CLUSTER = 4

# Take the scan lines as the reference spacing - monoisotopic peaks are about 0.5 m/z apart, so need to apply more
SCALING_FACTOR_X = 50.0
SCALING_FACTOR_Y = 1.0

def bbox(points):
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    return a

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-th-100-30000-30000.csv")
# rows_df = pd.read_csv("./data/tunemix/all-frames/frames-tune-mix-00001-01566.csv")

# Create a subset
frame = rows_df[(rows_df.frame == FRAME_ID)].sort_values(['scan','mz'], ascending=[True,True])

X_pretransform = frame[['mz','scan']].values

start = time.time()
X = np.copy(X_pretransform)
X[:,0] = X[:,0]*SCALING_FACTOR_X
X[:,1] = X[:,1]*SCALING_FACTOR_Y

db = DBSCAN(eps=EPSILON, min_samples=MIN_POINTS_IN_CLUSTER).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

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
boxes_df = pd.DataFrame()
clusters = [X_pretransform[labels == i] for i in xrange(n_clusters_)]
for cluster in clusters:
    # draw the bounding box
    bb = bbox(cluster)
    x1 = bb[0,0]
    y1 = bb[1,0]
    x2 = bb[0,1]
    y2 = bb[1,1]
    p = patches.Rectangle((x1, y1), x2-x1, y2-y1, fc = 'none', ec = 'green', linewidth=1)
    axes.add_patch(p)
    # add it to the dataframe for writing out to the CSV
    boxes_df = boxes_df.append({'frame_id':FRAME_ID,'low_mz':x1,'high_mz':x2,'low_scan':y1,'high_scan':y2,'cluster_id':1,'confidence':1.0}, ignore_index=True)

# write out the boxes CSV
boxes_df.frame_id = boxes_df.frame_id.astype(np.uint16)
boxes_df.low_scan = boxes_df.low_scan.astype(np.uint16)
boxes_df.high_scan = boxes_df.high_scan.astype(np.uint16)
boxes_df.cluster_id = boxes_df.cluster_id.astype(np.uint16)
boxes_df.to_csv("./data/frames-th-100-30000-30000-boxes.csv", sep=',', index=False)

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
