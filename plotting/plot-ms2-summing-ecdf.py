import sys
import numpy as np
import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
import argparse
import statsmodels.api as sm

parser = argparse.ArgumentParser(description='Plot the empirical cumulative distribution function.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

# Connect to the database file
conn = sqlite3.connect(args.database_name)
c = conn.cursor()
c.execute("select number_frames from summed_ms2_regions;")
intensity_v = np.array(c.fetchall(), dtype=np.float32)
intensity_v.shape = (len(intensity_v),)
conn.close()

ecdf = sm.distributions.ECDF(intensity_v)
plt.plot(ecdf.x, ecdf.y)
# plt.xscale('log')
plt.xlabel('Number of source frames')
plt.ylabel('ECDF')
plt.margins(0.02)
plt.title("MS2 number of frames a point appears ECDF (Empirical Cumulative Distribution Function")

# x = np.sort(intensity_v)
# y = np.arange(1, len(x)+1)/float(len(x))

# plt.plot(x, y, marker='<', markerfacecolor='none')
# plt.xlabel('intensity')
# plt.ylabel('ECDF')
# plt.xscale('log')
# plt.margins(0.02)
# plt.title("Feature ECDF (Empirical Cumulative Distribution Function")

plt.show()
