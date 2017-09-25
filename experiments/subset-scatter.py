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

THRESHOLD = 85

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-th-{}-30000-30000.csv".format(THRESHOLD))

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]
# subset = rows_df[(rows_df.frame == FRAME_ID)]

# Plot the frame
fig = plt.figure()
colourmap = plt.get_cmap("brg")

axes = fig.add_subplot(111)
sc = axes.scatter(x=subset.mz, y=subset.scan, c=subset.intensity, linewidth=0, s=8, cmap=colourmap)
cb = plt.colorbar(sc)
cb.set_label('Intensity')
# plt.xlim(LOW_MZ, HIGH_MZ)
# plt.ylim(HIGH_SCAN, LOW_SCAN)
plt.xlim([LOW_MZ,HIGH_MZ])
plt.ylim([HIGH_SCAN,LOW_SCAN])

# ax = plt.gca()
# ax.axis('tight')

plt.title('Frame {}, Noise Threshold = {}'.format(FRAME_ID, THRESHOLD))
plt.xlabel('m/z')
plt.ylabel('scan')
plt.tight_layout()
plt.show()
plt.close('all')
