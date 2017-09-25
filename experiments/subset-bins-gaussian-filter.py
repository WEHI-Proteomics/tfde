import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy import ndimage
import time

# Area of interest
FRAME_ID = 30000
LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

MAX_BINS = 2000

SIGMA_X = 4.0
SIGMA_Y = 2.0
GAUSSIAN_THRESHOLD = 0.035

IMAGE_X_PIXELS = 1600
IMAGE_Y_PIXELS = 900


# Read in the frames CSV
rows_df = pd.read_csv("frames-30000-30010.csv")

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]

# The subset in array form
x = subset.mz
y = subset.scan
z = subset.intensity
subset_arr = subset[['mz','scan','intensity']].values

bins = np.linspace(subset.mz.min(),subset.mz.max(), num=MAX_BINS)
inds = np.digitize(subset.mz, bins)

subset_a = np.zeros((MAX_BINS+1, subset.scan.max()+1))
subset_a[inds, subset.scan] = subset.intensity

start = time.time()

# Convert the array to an image
subset_g = gaussian_filter(subset_a, sigma=(SIGMA_X,SIGMA_Y), order=0)

# Find the bounding boxes of the peaks
mask = subset_g > subset_g.max()*GAUSSIAN_THRESHOLD
im, number_of_objects = ndimage.label(mask.T)
blobs = ndimage.find_objects(im)

end = time.time()
print("elapsed time = {} sec".format(end-start))

# Visualise the array
fig = plt.figure()
axes = fig.add_subplot(111)

plt.imshow(subset_g.T, cmap='hot', aspect='auto')
plt.xlim(0, MAX_BINS)
plt.ylim(subset.scan.max(), subset.scan.min())

# Add the bounding boxes to the image plot
for blob in blobs:
    x1 = int(blob[1].start)
    width = int(blob[1].stop - blob[1].start)
    y1 = int(blob[0].start)
    height = int(blob[0].stop - blob[0].start)
    print("x1={}, y1={}, w={}, h={}".format(x1, y1, width, height))
    p = patches.Rectangle((x1, y1), width, height, fc = 'none', ec = 'green', linewidth=2)
    axes.add_patch(p)

plt.xlabel('m/z bin')
plt.ylabel('scan')
plt.tight_layout()
fig.savefig('./subset-gaussian-{}-{}-{}.png'.format(SIGMA_X,SIGMA_Y,GAUSSIAN_THRESHOLD), pad_inches = 0.0, dpi='figure')
plt.close('all')

# Plot the frame as a scatter plot
fig = plt.figure()
dpi = fig.get_dpi()
fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))

axes = fig.add_subplot(111)
axes.scatter(x=subset.mz, y=subset.scan, c=subset.intensity, linewidth=0)
plt.xlim(subset.mz.min(), subset.mz.max())
plt.ylim(subset.scan.max(), subset.scan.min())

# Add the bounding boxes to the plot
for blob in blobs:
    x_start = bins[blob[1].start]
    x_stop = bins[blob[1].stop]
    x1 = x_start
    width = x_stop - x_start
    y1 = int(blob[0].start)
    height = int(blob[0].stop - blob[0].start)
    print("x1={}, y1={}, w={}, h={}".format(x1, y1, width, height))
    p = patches.Rectangle((x1, y1), width, height, fc = 'none', ec = 'orange', linewidth=2)
    axes.add_patch(p)

ax = plt.gca()
ax.axis('tight')
plt.xlabel('m/z')
plt.ylabel('scan')
plt.tight_layout()
fig.savefig('./subset-scatter-{}-{}-{}.png'.format(SIGMA_X,SIGMA_Y,GAUSSIAN_THRESHOLD), pad_inches = 0.0, dpi='figure')
plt.close('all')
