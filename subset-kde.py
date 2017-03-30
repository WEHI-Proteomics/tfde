import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd

# The frame of interest
FRAME_ID = 30000

# Peak 1
LOW_MZ = 566.7
HIGH_MZ = 566.9
LOW_SCAN = 517
HIGH_SCAN = 549

# Read in the frames CSV
rows_df = pd.read_csv("./data/frames-th-100-30000-30000.csv")

# Create a subset, ordered by scan number and m/z
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]
mz_intensity_arr = subset[['mz','intensity']].sort_values(['mz'], ascending=[True]).values
scan_intensity_arr = subset[['scan','intensity']].sort_values(['scan'], ascending=[True]).values

temp = mz_intensity_arr[:,0]    # just the m/z values
mz_arr = temp[:, np.newaxis]    # an Nx1 array of m/z values
temp = scan_intensity_arr[:,0]  # just the scan number values
scan_arr = temp[:, np.newaxis]  # an Nx1 array of scan number values

# Generate the range of points to use in visualisation of the density
mz_range_extend = (HIGH_MZ-LOW_MZ)
scan_range_extend = (HIGH_SCAN-LOW_SCAN)
xfit_mz = np.linspace(LOW_MZ-mz_range_extend, HIGH_MZ+mz_range_extend, 1000)
xfit_scan = np.linspace(LOW_SCAN-scan_range_extend, HIGH_SCAN+scan_range_extend, 1000)

# Change to Nx1 array because that's what KDE wants
Xfit_mz = xfit_mz[:, np.newaxis]
Xfit_scan = xfit_scan[:, np.newaxis]


# # Use a grid search to find the optimal bandwidth for the mz axis
# print("Estimating bandwidth in the m/z dimension")
# grid_mz = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.01, HIGH_MZ, 1000)}, cv=20)
# grid_mz.fit(mz_arr)
# print grid_mz.best_params_

# # Use a grid search to find the optimal bandwidth for the scan axis
# print("Estimating bandwidth in the scan dimension")
# grid_scan = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.01, HIGH_SCAN, 1000)}, cv=20)
# grid_scan.fit(scan_arr)
# print grid_scan.best_params_


# Evaluate the density model on the data
kde_mz = KernelDensity(kernel='gaussian', bandwidth=0.03)  # bandwidth was determined from GridSearchCV
kde_mz.fit(mz_arr)
density_mz = np.exp(kde_mz.score_samples(Xfit_mz))

kde_scan = KernelDensity(kernel='gaussian', bandwidth=5.6) # bandwidth was determined from GridSearchCV
kde_scan.fit(scan_arr)
density_scan = np.exp(kde_scan.score_samples(Xfit_scan))

# Plot two 2D graphs on a common intensity y-axis: scan & kde_scan, and mz & kde_mz
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1b = ax1.twinx()
ax2b = ax2.twinx()

# Plot the actual peaks
ax1.plot(mz_intensity_arr[:, 0], mz_intensity_arr[:, 1], 'ok', markersize=2)
ax2.plot(scan_intensity_arr[:, 0], scan_intensity_arr[:, 1], 'ok', markersize=2)

# Plot the density estimates
ax1b.plot(xfit_mz, density_mz, '-g', lw=1)
ax2b.plot(xfit_scan, density_scan, '-g', lw=1)

ax1.set_xlabel('m/z')
ax2.set_xlabel('scan')

plt.show()
plt.close('all')
