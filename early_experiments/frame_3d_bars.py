import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import timsdata

if len(sys.argv) < 2:
    raise RuntimeError("need arguments: tdf_directory")

analysis_dir = sys.argv[1]
if sys.version_info.major == 2:
    analysis_dir = unicode(analysis_dir)

timsfile = timsdata.TimsData(analysis_dir)

frame_id = 30000

(frame_data_id, num_scans) = timsfile.conn.execute(
      "SELECT TimsId, NumScans FROM Frames WHERE Id = {}".format(frame_id)).fetchone()

frame = timsfile.readScans(frame_data_id, 0, num_scans)

scan_begin = 0
scan_end = num_scans
threshold = 100

def threshold_scan_transform(threshold, indicies, intensities):
    np_mz = timsfile.indexToMz(frame_id, np.array(indicies, dtype=np.float64))
    np_int = np.asarray(intensities, dtype=np.int32)
    low_indices = np.where(np_int < threshold)
    np_mz = np.delete(np_mz, low_indices)
    np_int = np.delete(np_int, low_indices)
    return np_mz, np_int


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(scan_begin, scan_end):
    scan = frame[i]
    if len(scan[0]) > 0:
        mz, intensities = threshold_scan_transform(threshold, scan[0], scan[1])
        peaks = len(mz)
        if peaks > 0:
            ax.bar(mz, intensities, zs=i, zdir='y', color='b', alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
