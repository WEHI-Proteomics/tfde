
# coding: utf-8

# In[9]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os


# In[2]:


CONVERTED_DATABASE_NAME = '/home/ubuntu/HeLa_20KInt/HeLa_20KInt.sqlite'


# In[3]:


db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
frames_df = pd.read_sql_query("select * from summed_frames where retention_time_secs <= 600 and intensity > 100 order by retention_time_secs", db_conn)
db_conn.close()


# In[4]:


frames_df.frame_id.min()


# In[5]:


frames_df.frame_id.max()


# In[6]:


frame_id = 1


# In[7]:


frame_df = frames_df[frames_df.frame_id==frame_id]


# In[10]:


# set a filename, run the logistic model, and create the plot
gif_filename = 'HeLa_20KInt'
save_folder = 'animation'
working_folder = '/home/ubuntu/{}/{}'.format(save_folder, gif_filename)
if not os.path.exists(working_folder):
    os.makedirs(working_folder)

fig = plt.figure()
ax = Axes3D(fig)
plt.gca().invert_yaxis()
plt.xlabel('m/z')
plt.ylabel('scan')

for frame_id in range(1,250):
    frame_df = frames_df[frames_df.frame_id==frame_id]
    ax.scatter(frame_df.mz, frame_df.scan, frame_df.intensity, c=np.log(frame_df.intensity), cmap='cool')
    fig.suptitle('Frame {}'.format(frame_id), fontsize=16, x=0.5, y=0.85)
    plt.savefig('{}/img{:03d}.png'.format(working_folder, frame_id), bbox_inches='tight')

plt.close()

