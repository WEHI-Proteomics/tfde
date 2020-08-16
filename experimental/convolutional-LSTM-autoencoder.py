#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


# In[ ]:


import pandas as pd
import numpy as np
import glob
from PIL import Image


# In[ ]:


# EXPERIMENT_DIR = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test'
EXPERIMENT_DIR = '/home/daryl/experiments/dwm-test'
RUN_NAME = '190719_Hela_Ecoli_1to1_01'


# In[ ]:


ENCODED_FEATURES_DIR = '{}/encoded-features/{}'.format(EXPERIMENT_DIR, RUN_NAME)
FEATURE_SLICES_DIR = '{}/slices'.format(ENCODED_FEATURES_DIR)


# In[ ]:


# load the feature IDs that we processed earlier
FEATURE_ID_LIST_FILE = '{}/feature_ids.pkl'.format(ENCODED_FEATURES_DIR)
feature_list_df = pd.read_pickle(FEATURE_ID_LIST_FILE)
features_l = feature_list_df.feature_id.tolist()


# In[ ]:


feature_list_df.head()


# In[ ]:


# load the feature slices
feature_movies_l = []
for feature_id in features_l:
    # load the slices for the feature
    slices_l = sorted(glob.glob("{}/feature-{}-slice-*.png".format(FEATURE_SLICES_DIR, feature_id)))
    feature_slices_l = []
    for feature_slice in slices_l:
        # load the image and generate the feature vector
        img = Image.open(feature_slice)
        x = image.img_to_array(img.resize((256,256)))
        feature_slices_l.append(x)
    feature_slices = np.array(feature_slices_l)
    feature_slices = feature_slices.astype('float32') / 255.
    feature_movies_l.append(feature_slices)
feature_movies = np.array(feature_movies_l)


# In[ ]:


feature_movies.shape


# In[ ]:


# build the model
seq = Sequential()

seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 20, 256, 256, 3)))  # 20 images of 256x256x3
seq.add(LayerNormalization())

seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
seq.add(LayerNormalization())
# # # # #
seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
seq.add(LayerNormalization())

seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
seq.add(LayerNormalization())

seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
seq.add(LayerNormalization(name='encoded'))

# # # # #
seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
seq.add(LayerNormalization())

seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
seq.add(LayerNormalization())

seq.add(TimeDistributed(Conv2D(3, (11, 11), activation="sigmoid", padding="same")))
print(seq.summary())


# In[ ]:


encoder = Model(inputs=seq.inputs, outputs=seq.get_layer(name='encoded').output)


# In[ ]:


seq.compile(loss='mse', optimizer=Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))


# In[ ]:


seq.fit(feature_movies, feature_movies, batch_size=5, epochs=20, shuffle=False)


# In[ ]:


seq.save('{}/')

