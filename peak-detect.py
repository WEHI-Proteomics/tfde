import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import time
from scipy import signal
import peakutils
import copy
import sqlite3

# Process a set of frames using DBSCAN, and save the detected peaks to a database

FRAME_START = 29900
FRAME_END = 30100

THRESHOLD = 85

EPSILON_PEAK = 2.5
MIN_POINTS_IN_PEAK = 4

EPSILON_CLUSTER = 2.0
MIN_POINTS_IN_CLUSTER = 2

# scaling factors derived from manual inspection of good peak spacing in the subset plots
SCALING_FACTOR_PEAKS_X = 50.0
SCALING_FACTOR_PEAKS_Y = 1.0

SCALING_FACTOR_CLUSTERS_X = 1.0
SCALING_FACTOR_CLUSTERS_Y = 0.2

def bbox(points):
    # print("points {}".format(points))
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    # print("bb {}".format(a))
    return a

# Split the provided bounding box at the given y point, and return two bounding boxes
def split_bbox_y(bbox, y):
    a = copy.deepcopy(bbox)
    b = copy.deepcopy(bbox)
    a[1,1] = y  # y2 = y (y becomes the lower limit for the top bbox)
    b[1,0] = y  # y1 = y (y becomes the upper limit for the bottom bbox)
    return a, b

def bbox_points(bbox):
    return {'tl':{'x':bbox[0,0], 'y':bbox[1,0]}, 'br':{'x':bbox[0,1], 'y':bbox[1,1]}}

def bbox_centroid(bbox):
    return {'x':bbox[0,0]+(bbox[0,1]-bbox[0,0])/2, 'y':bbox[1,0]+(bbox[1,1]-bbox[1,0])/2}



# Connect to the database file
sqlite_file = "\\temp\\frames-th-{}-{}-{}.sqlite".format(THRESHOLD, FRAME_START, FRAME_END)
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Set up the table for detected peaks
c.execute('''DROP TABLE IF EXISTS peaks''')
c.execute('''CREATE TABLE peaks (frame_id INTEGER, peak_id INTEGER, state TEXT, centroid_mz REAL, centroid_scan INTEGER)''')

c.execute('''DROP TABLE IF EXISTS peak_points''')
c.execute('''CREATE TABLE peak_points (frame_id INTEGER, peak_id INTEGER, point_id INTEGER)''')
# Indexes
c.execute('''DROP INDEX IF EXISTS idx_peaks''')
c.execute('''CREATE INDEX idx_peaks ON peaks (frame_id)''')

c.execute('''DROP INDEX IF EXISTS idx_peak_points''')
c.execute('''CREATE INDEX idx_peak_points ON peak_points (frame_id)''')

c.execute('''DROP INDEX IF EXISTS idx_frame_point''')
c.execute('''CREATE INDEX idx_frame_point ON frames (frame_id,point_id)''')

for frame_id in range(FRAME_START, FRAME_END+1):

    print("Reading frame {} from database {}...".format(frame_id, sqlite_file))
    frame_df = pd.read_sql_query("select * from frames where frame_id={} ORDER BY MZ, SCAN ASC;".format(frame_id), conn)

    X_pretransform = frame_df[['mz','scan']].values

    start = time.time()
    X = np.copy(X_pretransform)
    X[:,0] = X[:,0]*SCALING_FACTOR_PEAKS_X
    X[:,1] = X[:,1]*SCALING_FACTOR_PEAKS_Y

    db = DBSCAN(eps=EPSILON_PEAK, min_samples=MIN_POINTS_IN_PEAK, n_jobs=-1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    end = time.time()
    print("found {} peaks in frame {} in {} seconds.".format(n_clusters_, frame_id, end-start))

    # Process the clusters
    mono_peaks = []  # where we record all the monoisotopic peaks we find
    peak_points = []
    clusters = [X_pretransform[labels == i] for i in xrange(n_clusters_)]
    peak_id = 0
    for cluster in clusters:
        peak_id += 1
        centroid = bbox_centroid(bbox(cluster))
        # add the peak to the collection
        mono_peaks.append((frame_id, peak_id, "", centroid['x'], int(centroid['y'])))
        # write out all the points in the cluster
        for point in cluster:
            # Find the point's pointID
            row = frame_df[(frame_df.mz == point[0]) & (frame_df.scan == point[1])]
            point_id = int(row.point_id.values[0])
            peak_points.append((frame_id, peak_id, point_id))
            # Update the point's peak_id in the database
            values = (peak_id, frame_id, point_id)
            c.execute("update frames set peak_id=? where frame_id=? and point_id=?", values)
    # Write out all the peaks to the database
    c.executemany("INSERT INTO peaks VALUES (?, ?, ?, ?, ?)", mono_peaks)
    c.executemany("INSERT INTO peak_points VALUES (?, ?, ?)", peak_points)
    conn.commit()

conn.close()
