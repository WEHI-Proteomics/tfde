# -*- coding: utf-8 -*-
"""Test program using Python wrapper for timsdata.dll"""

import sys

if len(sys.argv) < 2:
    raise RuntimeError("need arguments: tdf_directory")

analysis_dir = sys.argv[1]

if sys.version_info.major == 2:
    analysis_dir = unicode(analysis_dir)

import timsdata, sqlite3, sys, time
import numpy as np, matplotlib.pyplot as plt

td = timsdata.TimsData(analysis_dir)
conn = td.conn


# Get total frame count:
q = conn.execute("SELECT COUNT(*) FROM Frames")
row = q.fetchone()
N = row[0]
print("Analysis has {0} frames.".format(N))


# Get a projected mass spectrum:
frame_id = 10
q = conn.execute("SELECT NumScans FROM Frames WHERE Id={0}".format(frame_id))
num_scans = q.fetchone()[0]

numplotbins = 500;
min_mz = 0
max_mz = 3000
mzbins = np.linspace(min_mz, max_mz, numplotbins)
midmz = (mzbins[0:numplotbins-1] + mzbins[1:numplotbins]) / 2
summed_intensities = np.zeros(numplotbins+1)

for scan in td.readScans(frame_id, 0, num_scans):
    index = np.array(scan[0], dtype=np.float64)
    mz = td.indexToMz(frame_id, index)
    if len(mz) > 0:
        plotbins = np.digitize(mz, mzbins)
        intens = scan[1]
        for i in range(0, len(intens)):
            summed_intensities[plotbins[i]] += intens[i]

# Get list of scanned mobilities
scan_number_axis = np.arange(num_scans, dtype=np.float64)

ook0_axis = td.scanNumToOneOverK0(frame_id, scan_number_axis)
scan_number_from_ook0_axis = td.oneOverK0ToScanNum(frame_id, ook0_axis)
voltage_axis = td.scanNumToVoltage(frame_id, scan_number_axis)
scan_number_from_voltage_axis = td.voltageToScanNum(frame_id, voltage_axis)

print(scan_number_axis[0], scan_number_axis[-1])
print(ook0_axis[0], ook0_axis[-1], scan_number_from_ook0_axis[0], scan_number_from_ook0_axis[-1])
print(voltage_axis[0], voltage_axis[-1], scan_number_from_voltage_axis[0], scan_number_from_voltage_axis[-1])

plt.figure(1, figsize=(10,5))
plt.clf()
plt.stem(midmz, summed_intensities[1:numplotbins])
plt.xlim(min_mz, max_mz)
plt.show()
