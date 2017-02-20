import sys
import numpy as np
from numpy import ix_
import timsdata
import csv


MAX_INDEX = 3200
MAX_SCANS = 814
MAX_INTENSITY = 25000
THRESHOLD = 100
PEAK_THRESHOLD = 500
FRAME_START = 30000
FRAME_END = 30010

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

filename = "frames-{:0>5}-{:0>5}.csv".format(FRAME_START, FRAME_END)
with open(filename, 'wb') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(["frame ID", "mz", "index", "scan", "intensity"])

	for frame_id in range(FRAME_START, FRAME_END+1):
	    print("Frame {:0>5} of {}".format(frame_id, frame_count))

	    (frame_data_id, num_scans) = timsfile.conn.execute(
	          "SELECT TimsId, NumScans FROM Frames WHERE Id = {}".format(frame_id)).fetchone()
	    scans = timsfile.readScans(frame_data_id, 0, num_scans)
	    scan_begin = 0
	    scan_end = num_scans

	    frame = np.zeros((MAX_INDEX, MAX_SCANS))

	    for i in range(scan_begin, scan_end):
	        scan = scans[i]
	        if len(scan[0]) > 0:
	            mz, intensities = threshold_scan_transform(THRESHOLD, scan[0], scan[1])
	            indices = np.round(timsfile.mzToIndex(frame_id, mz)).astype(int)
	            peaks = len(mz)
	            if peaks > 0:
	            	for index in range(0,len(mz)):
	            		#                frame ID  mz         index           scan  intensity
	            		writer.writerow([frame_id, mz[index], indices[index], i,    intensities[index]])
