import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import OpticsClusterArea as OP
from itertools import *
import AutomaticClustering as AutoC
from sklearn.preprocessing import StandardScaler

FRAME_ID = 30000

LOW_MZ = 565
HIGH_MZ = 570
LOW_SCAN = 500
HIGH_SCAN = 600

# Read in the frames CSV
rows_df = pd.read_csv("frames-30000-30010.csv")

# Create a subset
subset = rows_df[(rows_df.frame == FRAME_ID) & (rows_df.mz >= LOW_MZ) & (rows_df.mz <= HIGH_MZ) & (rows_df.scan >= LOW_SCAN) & (rows_df.scan <= HIGH_SCAN)]
X = subset[['mz','scan']].values
X = StandardScaler().fit_transform(X)


#run the OPTICS algorithm on the points, using a smoothing value (0 = no smoothing)
RD, CD, order = OP.optics(X,8)

RPlot = []
RPoints = []
        
for item in order:
    RPlot.append(RD[item]) #Reachability Plot
    RPoints.append([X[item][0],X[item][1]]) #points in their order determined by OPTICS

#hierarchically cluster the data
rootNode = AutoC.automaticCluster(RPlot, RPoints)

#print Tree (DFS)
AutoC.printTree(rootNode, 0)

#graph reachability plot and tree
AutoC.graphTree(rootNode, RPlot)

#array of the TreeNode objects, position in the array is the TreeNode's level in the tree
array = AutoC.getArray(rootNode, 0, [0])

#get only the leaves of the tree
leaves = AutoC.getLeaves(rootNode, [])

#graph the points and the leaf clusters that have been found by OPTICS
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X[:,0], X[:,1], 'y.')
colors = cycle('gmkrcbgrcmk')
for item, c in zip(leaves, colors):
    node = []
    for v in range(item.start,item.end):
        node.append(RPoints[v])
    node = np.array(node)
    ax.plot(node[:,0],node[:,1], c+'o', ms=5)

plt.savefig('subset-optics.png', dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches=None, pad_inches=0.1)
# plt.show()
plt.close('all')





# plt.title('Estimated number of clusters: %d' % n_clusters_)

# plt.xlim(subset.mz.min(), subset.mz.max())
# plt.ylim(subset.scan.max(), subset.scan.min())

# ax = plt.gca()
# ax.axis('tight')
# plt.xlabel('m/z')
# plt.ylabel('scan')
# plt.tight_layout()
# plt.show()
# plt.close('all')
