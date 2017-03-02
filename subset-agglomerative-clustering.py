import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import time

FRAME_ID = 30000

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

# Read in the frames CSV
rows_df = pd.read_csv("frames-30000-30010.csv")

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]
X = subset[['mz','scan']].values
X = StandardScaler().fit_transform(X)

knn_graph = kneighbors_graph(X, 5, include_self=False)

for connectivity in (None, knn_graph):
    for n_clusters in (30, 20, 10):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average', 'complete', 'ward')):
            plt.subplot(1, 3, index + 1)
            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=knn_graph,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.spectral)
            x = X[:, 0]
            y = X[:, 1]
            plt.xlim(x.min(), x.max())
            plt.ylim(y.max(), y.min())
            plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time), fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                left=0, right=1)
            plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)


plt.show()
