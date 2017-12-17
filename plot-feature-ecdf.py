import sys
import numpy as np
import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot the empirical cumulative distribution function.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

# Connect to the database file
conn = sqlite3.connect(args.database_name)
c = conn.cursor()
c.execute("select intensity_sum from clusters where (frame_id,cluster_id) in (select base_frame_id,base_cluster_id from features limit 10);")
intensity_v = np.array(c.fetchall(), dtype=np.float32)
conn.close()

x = np.sort(intensity_v)
y = np.arange(len(x))/float(len(x))

plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('intensity')
plt.ylabel('ECDF')
plt.margins(0.02)
plt.title("Feature ECDF (Empirical Cumulative Distribution Function")

plt.show()
