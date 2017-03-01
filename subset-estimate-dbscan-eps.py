import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import time

FRAME_ID = 30000
MIN_POINTS_IN_CLUSTER = 4

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-30000-30010.csv")

# Create a subset
frame = rows_df[(rows_df.frame == FRAME_ID)]
X = frame[['mz','scan']].values
X = StandardScaler().fit_transform(X)

nbrs = NearestNeighbors(n_neighbors=MIN_POINTS_IN_CLUSTER+1, algorithm='ball_tree').fit(X)  # 4th nearest neighbour
distances, indices = nbrs.kneighbors(X)

d4 = np.sort(distances[:,MIN_POINTS_IN_CLUSTER])

plt.title('Sorted distance to 4th-nearest neighbour for each point in frame {}'.format(FRAME_ID))
plt.plot(d4, 'o', markeredgecolor='k', markeredgewidth=0.0, markersize=4)
plt.xlabel('Point index')
plt.ylabel('Distance')
plt.grid()
plt.show()
plt.close('all')
