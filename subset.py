import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd

FRAME_ID = 30000

LOW_MZ = 560
HIGH_MZ = 582
LOW_SCAN = 300
HIGH_SCAN = 650

IMAGE_X_PIXELS = 1600
IMAGE_Y_PIXELS = 900


# Read in the frames CSV
rows_df = pd.read_csv("frames-30000-30010.csv")

# Create a subset for m/z 560-582, scans 300-650
#   0      1    2    3     4
# frame ID,mz,index,scan,intensity
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]


# Apply a Gaussian filter
# frame_g = gaussian_filter(frame, sigma=(10,10), order=0)

# Plot the frame
fig = plt.figure()
dpi = fig.get_dpi()
fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))
axes = fig.add_subplot(111)

axes.scatter(subset.mz, subset.scan)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
ax.axis('tight')
plt.tight_layout()
plt.show()
