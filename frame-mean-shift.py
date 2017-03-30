import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd
from itertools import cycle
import time

# The frame of interest
# FRAME_ID = 708
FRAME_ID = 30000

# Peak A
# LOW_MZ = 0
# HIGH_MZ = 3000.0
# LOW_SCAN = 1
# HIGH_SCAN = 350

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

# Read in the frames CSV
# rows_df = pd.read_csv("./data/tunemix/all-frames/frames-tune-mix-00001-01566.csv")
# rows_df = pd.read_csv("./data/frames-30000-30010.csv")
rows_df = pd.read_csv("./data/frames-th-100-30000-30000.csv")

# Create a subset, ordered by scan number and m/z
frame = rows_df[(rows_df.frame == FRAME_ID)].sort_values(['scan','mz'], ascending=[True,True])
X = frame[['mz','scan']].values

start = time.time()

# bandwidth = estimate_bandwidth(X, quantile=0.1)
# print ("bandwidth={}".format(bandwidth))

bandwidth = 5.6

ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
 
n_clusters_ = labels.max()+1
 
end = time.time()
print("elapsed time = {} sec".format(end-start))

#%% Plot result
plt.figure(1)
plt.clf()
 
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    # plt.plot(cluster_center[0], cluster_center[1],
    #          'o', markerfacecolor=col,
    #          markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
ax = plt.gca()
ax.axis('tight')
ax.set_xlim(xmin=LOW_MZ, xmax=HIGH_MZ)
ax.set_ylim(ymin=HIGH_SCAN, ymax=LOW_SCAN)
plt.xlabel('m/z')
plt.ylabel('scan')
plt.show()
plt.close('all')
