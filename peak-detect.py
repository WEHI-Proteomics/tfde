import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import time
from scipy import signal
import peakutils
import copy
import sqlite3

# Process a whole frame using DBSCAN, and then plot only the area of interest.

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
c.execute('''CREATE TABLE peaks (peak_id INTEGER, frame_id INTEGER, low_mz REAL, high_mz REAL, low_scan INTEGER, high_scan INTEGER, state TEXT)''')
c.execute('''CREATE INDEX idx_peaks ON peaks (peak_id, frame_id)''')

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
    clusters = [X_pretransform[labels == i] for i in xrange(n_clusters_)]
    peak_id = 0
    frame_id = frame_id
    for cluster in clusters:

        peak_id += 1

        # find the bounding box of each cluster
        bb = bbox(cluster)
        x1 = bbox_points(bb)['tl']['x']
        y1 = bbox_points(bb)['tl']['y']
        x2 = bbox_points(bb)['br']['x']
        y2 = bbox_points(bb)['br']['y']

        # get the intensity values by scan
        cluster_df = frame_df[(frame_df.mz >= x1) & (frame_df.mz <= x2) & (frame_df.scan >= y1) & (frame_df.scan <= y2)]
        cluster_intensity = cluster_df[['scan','intensity']].values
        cluster_mz = cluster_df[['mz','scan']].values

        # filter the intensity with a Gaussian filter
        window = signal.gaussian(20, std=5)
        filtered = signal.convolve(cluster_intensity[:, 1], window, mode='same') / sum(window)

        # find the maxima
        indexes = peakutils.indexes(filtered, thres=0.1, min_dist=10)

        # Find the minimum intensity between the maxima
        if len(indexes) == 2:
            minimum_intensity_index = np.argmin(filtered[indexes[0]:indexes[1]+1])+indexes[0]
            # split the bounding box at the minima
            a_peak, b_peak = split_bbox_y(bbox=bb, y=cluster_mz[minimum_intensity_index,1])
            # draw a bounding box around a_peak
            a_x1 = bbox_points(a_peak)['tl']['x']
            a_y1 = bbox_points(a_peak)['tl']['y']
            a_x2 = bbox_points(a_peak)['br']['x']
            a_y2 = bbox_points(a_peak)['br']['y']
            # draw a bounding box around b_peak
            b_x1 = bbox_points(b_peak)['tl']['x']
            b_y1 = bbox_points(b_peak)['tl']['y']
            b_x2 = bbox_points(b_peak)['br']['x']
            b_y2 = bbox_points(b_peak)['br']['y']
            # add the peaks to the collection
            mono_peaks.append((peak_id, frame_id, a_x1, a_x2, a_y1, a_y2, ""))
            mono_peaks.append((peak_id, frame_id, b_x1, b_x2, b_y1, b_y2, ""))
        else:
            # add the peak to the collection
            mono_peaks.append((peak_id, frame_id, x1, x2, y1, y2, ""))

    # Write out all the peaks to the database
    c.executemany("INSERT INTO peaks VALUES (?, ?, ?, ?, ?, ?, ?)", mono_peaks)
    conn.commit()

conn.close()
