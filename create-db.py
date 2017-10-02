import sys
import numpy as np
from numpy import ix_
import timsdata
import sqlite3
import os

# Usage: python create-db.py D:\Bruker\Databases\Hela200ng100msMSonlyPP23pro_Slot1-5_01_57.d

THRESHOLD = 0
FRAME_START = 1
FRAME_END = 283906
DB_VERSION = 6
COLLISION_ENERGY_PROPERTY = 1454

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
sqlite_file = "S:\\data\\Projects\\ProtemicsLab\\Bruker timsTOF\\extracts\\{}-frames-{}-{}-V{}.sqlite".format(os.path.basename(analysis_dir).split('.')[0], THRESHOLD, FRAME_START, FRAME_END, DB_VERSION)
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Create the table
c.execute('''DROP TABLE IF EXISTS frames''')
c.execute('''CREATE TABLE frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)''')
c.execute('''DROP INDEX IF EXISTS idx_frames''')
c.execute('''CREATE INDEX idx_frames ON frames (frame_id)''')
c.execute('''DROP TABLE IF EXISTS frame_properties''')
c.execute('''CREATE TABLE frame_properties (frame_id INTEGER, collision_energy REAL)''')
c.execute('''DROP INDEX IF EXISTS idx_frame_properties''')
c.execute('''CREATE INDEX idx_frame_properties ON frame_properties (frame_id)''')

points = []
frame_properties = []
peak_id = 0
for frame_id in range(FRAME_START, FRAME_END+1):
    print("Frame {:0>5} of {}".format(frame_id, frame_count))

    (frame_data_id, num_scans) = timsfile.conn.execute(
          "SELECT TimsId, NumScans FROM Frames WHERE Id = {}".format(frame_id)).fetchone()
    scans = timsfile.readScans(frame_data_id, 0, num_scans)
    scan_begin = 0
    scan_end = num_scans
    pointId = 0

    (collision_energy,) = timsfile.conn.execute(
          "SELECT Value FROM FrameProperties WHERE Frame={} AND Property={}".format(frame_id, COLLISION_ENERGY_PROPERTY)).fetchone()
    frame_properties.append((frame_id, collision_energy))

    for i in range(scan_begin, scan_end):
        scan = scans[i]
        if len(scan[0]) > 0:
            mz, intensities = threshold_scan_transform(THRESHOLD, scan[0], scan[1])
            indices = np.round(timsfile.mzToIndex(frame_id, mz)).astype(int)
            peaks = len(mz)
            if peaks > 0:
            	for index in range(0,len(mz)):
                    pointId += 1
                    points.append((frame_id, pointId, mz[index], i, intensities[index], peak_id))
    if frame_id % 1000 == 0:
        print("Writing 1000 frames...")
        c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", points)
        conn.commit()
        points = []

# Write what we have left
if len(points) > 0:
    c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", points)

c.executemany("INSERT INTO frame_properties VALUES (?, ?)", frame_properties)

# Commit changes and close the connection
conn.commit()
conn.close()
