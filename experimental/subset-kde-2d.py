import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
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
subset_arr = subset[['mz','scan']].values

# Estimate the density
kde = KernelDensity(bandwidth=0.04, metric='haversine', kernel='gaussian', algorithm='ball_tree')
kde.fit(Xtrain[ytrain == i])



kde = stats.gaussian_kde(subset_arr.T, bw_method='scott')
density = kde(subset_arr.T)

# Plot the frame
fig = plt.figure()
fig.suptitle('Density Estimate', fontsize=20)

# dpi = fig.get_dpi()
# fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))

axes = fig.add_subplot(111)
axes.scatter(x=subset.mz, y=subset.scan, c=density, linewidth=0)
plt.xlim(subset.mz.min(), subset.mz.max())
plt.ylim(subset.scan.max(), subset.scan.min())

ax = plt.gca()
ax.axis('tight')
plt.xlabel('m/z')
plt.ylabel('scan')
plt.tight_layout()
plt.show()
plt.close('all')
