import sys
import numpy as np
import pandas as pd
import time
import sqlite3
from pyteomics import mgf
import operator
import os.path

THRESHOLD = 200
FRAME_START = 137000
FRAME_END = 137020
DB_VERSION = 4
FRAMES_TO_SUM = 5
INSTRUMENT_RESOLUTION = 40000.0
MZ_INCREMENT = 0.000001
SOURCE_SQLITE_FILE = "\\temp\\frames-20ms-th-0-137000-138000-V4.sqlite"
DEST_SQLITE_FILE = "\\temp\\summed-frames-5-th-200-137000-138000-V4.sqlite"

# Formula from https://en.wikipedia.org/wiki/Gaussian_function
def gaussian(x, amplitude, peak):
    stddev = (peak / INSTRUMENT_RESOLUTION) / 2.35482
    num = np.power((x-peak), 2.)
    den = 2. * np.power(stddev, 2.)
    return amplitude * np.exp(-num/den)


source_conn = sqlite3.connect(SOURCE_SQLITE_FILE)
source_c = source_conn.cursor()

dest_conn = sqlite3.connect(DEST_SQLITE_FILE)
dest_c = dest_conn.cursor()

dest_c.execute('''DROP TABLE IF EXISTS frames''')
dest_c.execute('''CREATE TABLE frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)''')
dest_c.execute('''DROP INDEX IF EXISTS idx_frames''')
dest_c.execute('''CREATE INDEX idx_frames ON frames (frame_id)''')

points = []
peakId = 0
pointId = 0
summedFrameId = 0
for base_frame_id in range(FRAME_START, FRAME_END, FRAMES_TO_SUM):
    print("Reading {} component frames starting at {} from database {}...".format(FRAMES_TO_SUM, base_frame_id, SOURCE_SQLITE_FILE))
    frame_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id>={} AND frame_id<={} ORDER BY FRAME_ID, MZ, SCAN ASC;".format(base_frame_id, base_frame_id+FRAMES_TO_SUM-1), source_conn)
    # Calculate the standard deviation for a Gaussian peak centred at each m/z value
    # Relationsip of std dev to FWHM (full width at half maximum) from https://en.wikipedia.org/wiki/Gaussian_function
    print("Calculating the standard deviation for a Gaussian peak centred at each m/z value ({} points)".format(len(frame_df)))
    frame_df["std_dev"] = frame_df[["mz"]].apply(lambda mz: (mz / INSTRUMENT_RESOLUTION) / 2.35482)

    # Find the scan range for these frames
    scan_min = frame_df.scan.min()
    scan_max = frame_df.scan.max()

    # Find the m/z range for these frames
    mz_min = frame_df.mz.min()
    mz_max = frame_df.mz.max()

    for frame in range(base_frame_id, base_frame_id+FRAMES_TO_SUM):
        for scan in range(scan_min, scan_max+1):
            # Get all the points for this scan
            points_df = frame_df[(frame_df.scan == scan) & (frame_df.frame == frame)].sort('mz', ascending=True)




    summedFrameId += 1
    # Find the scan range for these frames
    scan_min = frame_df.scan.min()
    scan_max = frame_df.scan.max()
    for scan in range(scan_min, scan_max+1):
        # Get all the points in the 5 frames for this scan
        points_df = frame_df[frame_df.scan == scan].sort('mz', ascending=True)
        print("")
        print("Processing scan {} in frame {}. There are {} points for this scan.".format(scan, summedFrameId, len(points_df)))
        # Cluster the points along the m/z axis. A point belongs in the same cluster if it overlaps the next one along.
        for i in range(0, points_df.shape[0]-1):
        	point = points_df.irow(i)
        	next_point = points_df.irow(i+1)
        	if (point.mz + point.std_dev*2) >= (next_point.mz - next_point.std_dev*2):
        		point['cluster_id'] = clusterId
        		next_point['cluster_id'] = clusterId
        	else
        		clusterId += 1



        # For each point, step from -2 sigma to +2 sigma at 1x10-6 increments, summing the intensity from each other point in that range
        for index, point in points_df.iterrows():


            # Find the m/z range over which we sum the intensity for this point
            std_dev_2 = point.std_dev * 2.0
            mz_min = point.mz - std_dev_2
            mz_max = point.mz + std_dev_2
            # Find all the points whose gaussian peaks overlap
            overlapping_points_df = points_df[points_df.apply(lambda row: ((point.mz <= row.mz) and ((row.mz-(row.std_dev*2)) <= mz_max)) or ((point.mz >= row.mz) and ((row.mz+(row.std_dev*2)) >= mz_min)), axis=1)]
            print("There are {} points ({}) overlapping with point {} ({}).".format(len(overlapping_points_df), overlapping_points_df.mz.values, index, point.mz))
            # If there are no overlapping points and the sum of each point's intensity is lower than THRESHOLD, we don't need to sum nor write it out
            if (overlapping_points_df.intensity.sum() < THRESHOLD):
                print("Maximum intensity of overlapping points is below threshold - skipping")
            else:
                print("Summing intensity")
                for mz in np.arange(mz_min, mz_max, MZ_INCREMENT):
                    # Evaluate the intensity function for all overlapping points at this m/z value
                    intensity = 0.0
                    for i, p in overlapping_points_df.iterrows():
                        intensity += gaussian(mz, p.intensity, p.mz, p.std_dev)
                    if intensity >= THRESHOLD:
                        pointId += 1
                        points.append((summedFrameId, pointId, mz, scan, int(intensity), peakId))
    # Write out the points for this set of frames
    print("Writing {} points to the database.".format(len(points)))
    dest_c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", points)
    points = []

# Commit changes and close the connection
dest_conn.commit()
source_conn.close()
dest_conn.close()
