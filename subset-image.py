import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

FRAME_ID = 30000

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

IMAGE_X_PIXELS = 1600
IMAGE_Y_PIXELS = 900

# Read in the frames CSV
rows_df = pd.read_csv("frames-30000-30010.csv")

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]

x = subset.index - subset.index.min() # offset so we only get the view we want
y = subset.scan
subset_i = np.zeros((x.max()+1, y.max()+1))
subset_i[x, y] = subset.intensity

# Apply a Gaussian filter
# subset_g = gaussian_filter(subset_array, sigma=(10,10), order=0)

# Plot the frame
fig = plt.figure()
dpi = fig.get_dpi()
fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))

axes = fig.add_subplot(111)
axes.imshow(subset_i.T, interpolation='nearest', cmap='hot', extent=[subset.mz.min(), subset.mz.max(), subset.scan.max(), subset.scan.min()])

ax = plt.gca()
ax.axis('tight')
plt.xlabel('m/z')
plt.ylabel('scan')
plt.tight_layout()
plt.show()
