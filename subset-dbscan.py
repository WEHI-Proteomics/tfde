import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time

FRAME_ID = 30000

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

EPSILON = 15
MIN_SAMPLES = 4

def mydistance(x,y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-30000-30010.csv")

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]
X = subset[['mz','scan']].values

start = time.time()
X = StandardScaler().fit_transform(X)

# db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric=mydistance).fit(X)
db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

end = time.time()
print("elapsed time = {} sec".format(end-start))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
fig = plt.figure()
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markeredgewidth=0.0, markersize=4)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markeredgewidth=0.0, markersize=2)

plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.xlim(subset.mz.min(), subset.mz.max())
plt.ylim(subset.scan.max(), subset.scan.min())

ax = plt.gca()
ax.axis('tight')
plt.xlabel('m/z')
plt.ylabel('scan')
plt.tight_layout()
fig.savefig('./images/subset-dbscan-eps-{}-sam-{}.png'.format(EPSILON, MIN_SAMPLES), pad_inches = 0.0, dpi='figure')
plt.close('all')
