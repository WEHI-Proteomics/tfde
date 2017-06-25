import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# The frame of interest
FRAME_ID = 30000

# Peak 1
LOW_MZ = 565.7
HIGH_MZ = 565.9
LOW_SCAN = 513
HIGH_SCAN = 600

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-th-85-30000-30000.csv")

# Create a subset, ordered by scan number and m/z
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]
mz_intensity_arr = subset[['mz','intensity']].sort_values(['mz'], ascending=[True]).values
scan_intensity_arr = subset[['scan','intensity']].sort_values(['scan'], ascending=[True]).values

temp = mz_intensity_arr[:,0]    # just the m/z values
mz_arr = temp[:, np.newaxis]    # an Nx1 array of m/z values
temp = scan_intensity_arr[:,0]  # just the scan number values
scan_arr = temp[:, np.newaxis]  # an Nx1 array of scan number values

# Plot two 2D graphs on a common intensity y-axis
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
colourmap = plt.get_cmap("brg")

# Plot the actual peaks
# ax1.plot(mz_intensity_arr[:, 0], mz_intensity_arr[:, 1], 'o', markersize=2, c=mz_intensity_arr[:, 1])
# ax2.plot(scan_intensity_arr[:, 0], scan_intensity_arr[:, 1], 'o', markersize=2)

sc1 = ax1.scatter(x=subset.mz, y=subset.intensity, c=subset.intensity, linewidth=0, s=8, cmap=colourmap)
sc2 = ax2.scatter(x=subset.scan, y=subset.intensity, c=subset.intensity, linewidth=0, s=8, cmap=colourmap)

# cb = plt.colorbar(sc2)
# cb.set_label('Intensity')

ax1.set_xlabel('m/z')
ax1.set_ylabel('intensity')
ax2.set_xlabel('scan')

plt.show()
plt.close('all')
