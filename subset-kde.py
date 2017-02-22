import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import pandas as pd

# Area of interest
FRAME_ID = 30000
LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

# Read in the frames CSV
rows_df = pd.read_csv("frames-30000-30010.csv")

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]

x = subset.mz
y = subset.scan
z = subset.intensity
subset_arr = subset[['mz','scan','intensity']].values

# Estimate the density
kde = stats.gaussian_kde(subset_arr.T, bw_method='scott')
density = kde(subset_arr.T)

# Plot the density
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.scatter(x, y, z, c=density)
# ax.scatter(x, y, z, c=z)

plt.xlim(subset.mz.min(), subset.mz.max())
plt.ylim(subset.scan.max(), subset.scan.min())

ax.set_xlabel('m/z')
ax.set_ylabel('scan')
ax.set_zlabel('intensity')
fig.suptitle('Density Estimate', fontsize=20)

plt.show()
