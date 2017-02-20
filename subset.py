import sys
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import csv

FRAME_ID = 30000

LOW_MZ = 560
HIGH_MZ = 582
LOW_SCAN = 300
HIGH_SCAN = 650

IMAGE_X_PIXELS = 1600
IMAGE_Y_PIXELS = 900


# Read in the frames CSV
rows = genfromtxt('frames-30000-30010.csv', delimiter=',')

# Create a subset for m/z 560-582, scans 300-650
#   0      1    2    3     4
# frame ID,mz,index,scan,intensity

MIN_INDEX = sys.maxint
MAX_INDEX = 0
MIN_SCAN = sys.maxint
MAX_SCAN = 0

# Find the lowest and highest index and scan values
for idx,row in enumerate(rows):
	if idx > 0:
		ind = int(row[2])
		MIN_INDEX = min(MIN_INDEX, ind)
		MAX_INDEX = max(MAX_INDEX, ind)
		scan = int(row[3])
		MIN_SCAN = min(MIN_SCAN, scan)
		MAX_SCAN = max(MAX_SCAN, scan)

print("max index = {}, min index = {}".format(MAX_INDEX, MIN_INDEX))
print("max scan = {}, min scan = {}".format(MAX_SCAN, MIN_SCAN))

frame = np.zeros((MAX_INDEX+1, MAX_SCAN+1))

for idx,row in enumerate(rows):
	if idx > 0:
		frame_id = int(row[0])
		mz = float(row[1])
		index = int(row[2])
		scan = int(row[3])
		intensity = int(row[4])
		if frame_id == FRAME_ID:
			frame[index,scan] = intensity

# Apply a Gaussian filter
frame_g = gaussian_filter(frame, sigma=(10,10), order=0)

# Plot the frame
fig = plt.figure()
dpi = fig.get_dpi()
fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))
axes = fig.add_subplot(111)

axes.imshow(frame_g, interpolation='nearest', cmap='hot', aspect='auto')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.tight_layout()
plt.show()
