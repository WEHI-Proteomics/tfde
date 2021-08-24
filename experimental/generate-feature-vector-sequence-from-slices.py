import numpy as np
from PIL import Image
import glob
import sys
import os
import shutil
import pickle
import pandas as pd
from datetime import datetime
from matplotlib import colors, cm, pyplot as plt
from sklearn.model_selection import train_test_split
import argparse

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard


##################################
parser = argparse.ArgumentParser(description='Optionally train a new model to encode feature slices to vector sequences.')
parser.add_argument('-tnm','--train_new_model', action='store_true', help='Train a new model.')
args = parser.parse_args()


EXPERIMENT_DIR = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test'
RUN_NAME = '190719_Hela_Ecoli_1to1_01'

# check the slices directory
ENCODED_FEATURES_DIR = '{}/encoded-features/{}'.format(EXPERIMENT_DIR, RUN_NAME)
FEATURE_SLICES_DIR = '{}/slices'.format(ENCODED_FEATURES_DIR)
if not os.path.exists(FEATURE_SLICES_DIR):
    print("The slices directory is required but does not exist: {}".format(FEATURE_SLICES_DIR))
    sys.exit(1)

# check the model directory
CONV_MODEL_DIR = '{}/conv-encoder'.format(ENCODED_FEATURES_DIR)
if args.train_new_model:
    if os.path.exists(CONV_MODEL_DIR):
        shutil.rmtree(CONV_MODEL_DIR)
    os.makedirs(CONV_MODEL_DIR)
else:
    if not os.path.exists(CONV_MODEL_DIR):
        print("The model directory is required but does not exist: {}".format(CONV_MODEL_DIR))
        sys.exit(1)

# make the vectors directory
VECTORS_DIR = '{}/vectors'.format(ENCODED_FEATURES_DIR)
if os.path.exists(VECTORS_DIR):
    shutil.rmtree(VECTORS_DIR)
os.makedirs(VECTORS_DIR)

# make the sequences directory
VECTOR_SEQUENCES_DIR = '{}/vector-sequences'.format(ENCODED_FEATURES_DIR)
if os.path.exists(VECTOR_SEQUENCES_DIR):
    shutil.rmtree(VECTOR_SEQUENCES_DIR)
os.makedirs(VECTOR_SEQUENCES_DIR)

# load the feature IDs that we processed earlier
FEATURE_ID_LIST_FILE = '{}/feature_ids.pkl'.format(ENCODED_FEATURES_DIR)
feature_list_df = pd.read_pickle(FEATURE_ID_LIST_FILE)
features_l = feature_list_df.feature_id.tolist()

# load the feature slices
feature_slices_l = glob.glob("{}/feature-*-slice-*.png".format(FEATURE_SLICES_DIR))
image_arrays_l = []
slice_basenames_l = []
for feature_slice in sorted(feature_slices_l):
    # load the image and generate the feature vector
    img = Image.open(feature_slice)
    x = image.img_to_array(img.resize((112,112)))
    image_arrays_l.append(x)
    # derive the basename for the slice
    slice_basenames_l.append(os.path.basename(feature_slice).split('.')[0])
image_arrays = np.array(image_arrays_l)
image_arrays = image_arrays.astype('float32') / 255.

# set up the model
CONV_MODEL_NAME = '{}/conv-encoder'.format(CONV_MODEL_DIR)
if args.train_new_model:
    # create the model
    input_img = Input(shape=(112, 112, 3))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    model = Model(inputs = input_img, outputs = decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    encoder = Model(inputs = input_img, outputs = encoded)

    # split the data set into training and test
    X_train, X_test, _, _ = train_test_split(image_arrays, image_arrays, test_size=0.10)
    X_train = np.reshape(X_train, (len(X_train), 112, 112, 3))
    X_test = np.reshape(X_test, (len(X_test), 112, 112, 3))

    # divide X_test into validation and test
    split_num = int(len(X_test)/2)
    X_val = X_test[:split_num]
    X_test = X_test[split_num:]

    # set up TensorBoard
    logdir = "/tmp/tb/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    # train the model
    batch_size = 32
    epochs = 50
    print("training the model")
    history = model.fit(X_train, X_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_val, X_val),
                        callbacks=[tensorboard_callback],
                        shuffle=True)

    # evaluate performance against the test set
    score = model.evaluate(X_test, X_test, verbose=1)
    print("evaluating the model against the test set:")
    print(score)

    # save the encoder
    encoder.save(CONV_MODEL_NAME)
else:
    encoder = tf.keras.models.load_model(CONV_MODEL_NAME)

# encode the image set
print("encode the feature slices")
encoded_imgs = encoder.predict(image_arrays)

# record the feature vectors for each slice
print("saving the encoded slices to {}".format(VECTORS_DIR))
for slice_idx,slice_name in enumerate(slice_basenames_l):
    slice_vector_filename = '{}/{}.npy'.format(VECTORS_DIR, slice_name)
    slice_vector = encoded_imgs[slice_idx].flatten()
    np.save(slice_vector_filename, slice_vector)

# convert the feature vectors to vector sequences
print("converting the encoded slices to sequences in {}".format(VECTOR_SEQUENCES_DIR))
for feature_id in features_l:
    # load the slices for the feature
    slices_l = sorted(glob.glob("{}/feature-{}-slice-*.npy".format(VECTORS_DIR, feature_id)))
    sequence_arrays_l = []
    for slc in slices_l:
        print("processing vector {}".format(slc))
        sequence_arrays_l.append(np.load(slc))
    print()
    
    # save the vector sequence for this feature
    VECTOR_SEQUENCE_FILE = '{}/vector-sequence-feature-{}.pkl'.format(VECTOR_SEQUENCES_DIR, feature_id)
    with open(VECTOR_SEQUENCE_FILE, 'wb') as f:
        pickle.dump(sequence_arrays_l, f)
