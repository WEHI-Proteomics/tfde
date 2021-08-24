import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

# Area of interest
FRAME_ID = 30000
LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

# Image attributes
IMAGE_X_PIXELS = 1000
IMAGE_Y_PIXELS = 1000
IMAGE_DPI = 100

# Read in the frames CSV
rows_df = pd.read_csv("frames-30000-30010.csv")

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]

# The subset in array form
x = subset.mz
y = subset.scan
z = subset.intensity
subset_arr = subset[['mz','scan','intensity']].values

# Convert the array to an image
subset_g = gaussian_filter(subset_arr.T, sigma=(1.0,1.0), order=0)

# Visualise the array
fig = plt.figure()
dpi = fig.get_dpi()
fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))
axes = fig.add_subplot(111)

# plt.imshow(subset_g, interpolation='nearest', extent=(x.min(), x.max(), y.max(), y.min()), cmap='hot', aspect='auto')
plt.imshow(subset_arr.T, interpolation='nearest', extent=(x.min(), x.max(), y.max(), y.min()), cmap='hot', aspect='auto')

ax = plt.gca()
ax.axis('tight')
plt.xlabel('m/z')
plt.ylabel('scan')
plt.tight_layout()
plt.show()
plt.close('all')
