
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import os
from PIL import Image
import glob


# In[ ]:


gif_filename = 'HeLa_20KInt'
save_folder = 'animation'
working_folder = '/home/ubuntu/{}/{}'.format(save_folder, gif_filename)


# In[ ]:


# load all the static images into a list then save as an animated gif
gif_filepath = '{}/{}.gif'.format(working_folder, gif_filename)
images = [Image.open(image) for image in glob.glob('{}/*.png'.format(working_folder))]
images[0].save(gif_filepath,
               save_all=True,
               append_images=images[1:],
               duration=100,
               loop=0)
