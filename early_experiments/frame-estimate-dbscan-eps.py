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

SCALING_FACTOR = 20

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-30000-30010.csv")

# Create a subset
frame = rows_df[(rows_df.frame == FRAME_ID)]
X_pretransform = frame[['mz','scan']].values
X = np.copy(X_pretransform)
X[:,0] = X[:,0]*500.0
X[:,1] = X[:,1]*8.0

nbrs = NearestNeighbors(n_neighbors=MIN_POINTS_IN_CLUSTER+1).fit(X)  # 4th nearest neighbour
distances, indices = nbrs.kneighbors(X)

d4 = np.sort(distances[:,MIN_POINTS_IN_CLUSTER])

plt.title('Sorted distance to 4th-nearest neighbour for each point in frame {}'.format(FRAME_ID))
plt.plot(d4, 'o', markeredgecolor='k', markeredgewidth=0.0, markersize=4)
plt.xlabel('Point index')
plt.ylabel('Distance')
plt.grid()
plt.show()
plt.close('all')
