import pandas as pd
import matplotlib.pyplot as plt


# Read in the frames CSV
rows_df = pd.read_csv("./data/frame_points.csv")

series = rows_df[['frame_id','points']].values

# Plot the area of interest
fig = plt.figure()
axes = fig.add_subplot(111)

axes.plot(series[:,0], series[:,1], '-', color='orange', linewidth=1, markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=1)
plt.title('Number of data points above threshold (85) by frame')
plt.xlabel('Frame')
plt.ylabel('Number of points above threshold')
plt.show()
plt.close('all')
