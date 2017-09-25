import sys
import numpy as np
import pandas as pd
import sqlite3
from matplotlib import pyplot as plt

SQLITE_FILE = "\\temp\\summed-frames-20ms-th-0-137000-137004-V4.sqlite"

conn = sqlite3.connect(SQLITE_FILE)

intensity_df = pd.read_sql_query("select * from frames where frame_id=={};".format(1), conn)
conn.close()

x = np.sort(intensity_df.intensity.values)
y = np.arange(len(x))/float(len(x))

plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('intensity')
plt.ylabel('ECDF')
plt.margins(0.02)

plt.show()
