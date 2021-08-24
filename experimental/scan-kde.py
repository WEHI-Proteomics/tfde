import sys
import numpy as np
import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


FRAME_START = 137000
FRAME_END = 137004
SCAN = 90
SQLITE_FILE = "\\temp\\frames-20ms-th-0-137000-138000-V4.sqlite"

conn = sqlite3.connect(SQLITE_FILE)
frame_df = pd.read_sql_query("select mz,intensity,scan from frames where frame_id>={} AND frame_id<={} ORDER BY MZ ASC;".format(FRAME_START, FRAME_END), conn)
conn.close()

scan_min = frame_df.scan.min()
scan_max = frame_df.scan.max()
print("scans from {} to {}".format(scan_min, scan_max))

mz_min = frame_df.mz.min()
mz_max = frame_df.mz.max()

points_df = frame_df[frame_df.scan == SCAN]

mz_intensity_arr = points_df[['mz','intensity']].sort_values(['mz'], ascending=[True]).values

temp = mz_intensity_arr[:,0]    # just the m/z values
mz_arr = temp[:, np.newaxis]    # an Nx1 array of m/z values

xfit_mz = np.linspace(0.0, mz_max, 4000)
Xfit_mz = xfit_mz[:, np.newaxis]

kde_mz = KernelDensity(kernel='gaussian', bandwidth=0.03)
kde_mz.fit(mz_arr)
density_mz = np.exp(kde_mz.score_samples(Xfit_mz))

# Plot two 2D graphs on a common intensity y-axis: scan & kde_scan, and mz & kde_mz
f, ax1 = plt.subplots(1, 1, sharey=True)

ax1b = ax1.twinx()

# Plot the actual peaks
ax1.plot(mz_intensity_arr[:, 0], mz_intensity_arr[:, 1], 'ok', markersize=6, alpha=0.25)

# Plot the density estimates
ax1b.plot(xfit_mz, density_mz, '-g', lw=1)

ax1.set_xlabel('m/z')
plt.show()
plt.close('all')
