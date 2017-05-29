import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import time
import sqlite3
import copy

EPSILON_CLUSTER = 2.0
MIN_POINTS_IN_CLUSTER = 2

SCALING_FACTOR_CLUSTERS_X = 1.0
SCALING_FACTOR_CLUSTERS_Y = 0.2

FRAME_START = 29900
FRAME_END = 30100
THRESHOLD = 85
DB_VERSION = 4

# Connect to the database file
sqlite_file = "\\temp\\frames-th-{}-{}-{}-V{}.sqlite".format(THRESHOLD, FRAME_START, FRAME_END, DB_VERSION)
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

print("Setting up tables and indexes")
c.execute('''DROP TABLE IF EXISTS clusters''')
c.execute('''CREATE TABLE `clusters` ( `frame_id` INTEGER, `cluster_id` INTEGER, `centroid_mz` REAL, `centroid_scan` INTEGER, `state` TEXT, PRIMARY KEY(`cluster_id`,`frame_id`) )''')
c.execute('''DROP INDEX IF EXISTS idx_clusters''')
c.execute('''CREATE INDEX idx_clusters ON clusters (frame_id,cluster_id)''')
c.execute("update peaks set cluster_id=0")

for frame_id in range(FRAME_START, FRAME_END+1):

    print("Reading frame {} from database {}...".format(frame_id, sqlite_file))

    # Load the peaks for this frame
    peaks_df = pd.read_sql_query("select * from peaks where frame_id={};".format(frame_id), conn)
    Y_pretransform = peaks_df[['centroid_mz','centroid_scan']].values
    Y = np.copy(Y_pretransform)

    # Scale the data
    Y[:,0] = Y[:,0]*SCALING_FACTOR_CLUSTERS_X
    Y[:,1] = Y[:,1]*SCALING_FACTOR_CLUSTERS_Y

    # Cluster the peaks we found
    db = DBSCAN(eps=EPSILON_CLUSTER, min_samples=MIN_POINTS_IN_CLUSTER, n_jobs=-1).fit(Y)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("found {} clusters".format(n_clusters_))

    # Process the clusters
    peak_clusters = []  # where we record all the peak clusters we find
    clusters = [Y_pretransform[labels == i] for i in xrange(n_clusters_)]
    cluster_id = 0
    for cluster in clusters:
        cluster_id += 1
        # assign all the peaks to this cluster
        centroid_mz_max = 0.0
        centroid_mz_min = sys.float_info.max
        centroid_scan_max = 0
        centroid_scan_min = sys.maxint
        for peak in cluster:
            row = peaks_df[(peaks_df.centroid_mz == peak[0]) & (peaks_df.centroid_scan == peak[1])]
            # Find the cluster's extents
            centroid_mz_max = max(centroid_mz_max, row.centroid_mz.values[0])
            centroid_mz_min = min(centroid_mz_min, row.centroid_mz.values[0])
            centroid_scan_max = max(centroid_scan_max, row.centroid_scan.values[0])
            centroid_scan_min = min(centroid_scan_min, row.centroid_scan.values[0])
            # Find the peak's peakID
            peak_id = int(row.peak_id.values[0])
            # Update the point's peak_id in the database
            values = (cluster_id, frame_id, peak_id)
            c.execute("update peaks set cluster_id=? where frame_id=? and peak_id=?", values)
        # find the cluster's centroid
        centroid_mz = centroid_mz_min + (centroid_mz_max-centroid_mz_min)/2.0
        centroid_scan = int(centroid_scan_min + (centroid_scan_max-centroid_scan_min)/2.0)
        # add the cluster to the collection
        peak_clusters.append((frame_id, cluster_id, centroid_mz, centroid_scan, ""))
    # Write out all the clusters to the database
    c.executemany("INSERT INTO clusters VALUES (?, ?, ?, ?, ?)", peak_clusters)
    conn.commit()

conn.close()
