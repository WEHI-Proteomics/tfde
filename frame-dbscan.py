import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time

# Process a whole frame using DBSCAN, and then plot only the area of interest.

FRAME_ID = 30000

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

EPSILON = 17.0
MIN_POINTS_IN_CLUSTER = 4

SCALING_FACTOR = 20

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-30000-30010.csv")

# Create a subset
frame = rows_df[(rows_df.frame == FRAME_ID)]

X_pretransform = frame[['mz','scan']].values

start = time.time()
X = np.copy(X_pretransform)
X[:,0] = X[:,0]*500.0
X[:,1] = X[:,1]*8.0

# db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric=mydistance).fit(X)
db = DBSCAN(eps=EPSILON, min_samples=MIN_POINTS_IN_CLUSTER).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print('labels {}'.format(np.shape(labels)))

end = time.time()
print("elapsed time = {} sec".format(end-start))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Plot the area of interest
# Black is removed and used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
fig = plt.figure()
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X_pretransform[class_member_mask & core_samples_mask]
    # xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markeredgewidth=0.0, markersize=4)

    xy = X_pretransform[class_member_mask & ~core_samples_mask]
    # xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markeredgewidth=0.0, markersize=2)

plt.title('Epsilon={}, samples={}, clusters={}'.format(EPSILON, MIN_POINTS_IN_CLUSTER, n_clusters_))

# plt.xlim(LOW_MZ, HIGH_MZ)
# plt.ylim(HIGH_SCAN, LOW_SCAN)

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
