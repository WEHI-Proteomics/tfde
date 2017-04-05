import sys
import numpy as np
from numpy import ix_
import timsdata
import sqlite3


THRESHOLD = 85
FRAME_START = 1
FRAME_END = 67843

def threshold_scan_transform(threshold, indicies, intensities):
    np_mz = timsfile.indexToMz(frame_id, np.array(indicies, dtype=np.float64))
    np_int = np.asarray(intensities, dtype=np.int32)
    low_indices = np.where(np_int < threshold)
    np_mz = np.delete(np_mz, low_indices)
    np_int = np.delete(np_int, low_indices)
    return np_mz, np_int

if len(sys.argv) < 2:
    raise RuntimeError("need arguments: tdf_directory")

analysis_dir = sys.argv[1]
if sys.version_info.major == 2:
    analysis_dir = unicode(analysis_dir)

timsfile = timsdata.TimsData(analysis_dir)

# Get total frame count
q = timsfile.conn.execute("SELECT COUNT(*) FROM Frames")
row = q.fetchone()
frame_count = row[0]
print("Analysis has {0} frames.".format(frame_count))

# Connecting to the database file
sqlite_file = "\\temp\\frames-th-{}.sqlite".format(THRESHOLD)
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Create the table
c.execute('''DROP TABLE IF EXISTS frames''')
c.execute('''CREATE TABLE frames (frame_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER)''')

for frame_id in range(FRAME_START, FRAME_END+1):
    print("Frame {:0>5} of {}".format(frame_id, frame_count))
    points = []

    (frame_data_id, num_scans) = timsfile.conn.execute(
          "SELECT TimsId, NumScans FROM Frames WHERE Id = {}".format(frame_id)).fetchone()
    scans = timsfile.readScans(frame_data_id, 0, num_scans)
    scan_begin = 0
    scan_end = num_scans

    for i in range(scan_begin, scan_end):
        scan = scans[i]
        if len(scan[0]) > 0:
            mz, intensities = threshold_scan_transform(THRESHOLD, scan[0], scan[1])
            indices = np.round(timsfile.mzToIndex(frame_id, mz)).astype(int)
            peaks = len(mz)
            if peaks > 0:
            	for index in range(0,len(mz)):
                    points.append((frame_id, mz[index], i, intensities[index]))
    c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?)", points)

# Commit changes and close the connection
conn.commit()
conn.close()
