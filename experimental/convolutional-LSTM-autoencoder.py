from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np
import glob
from PIL import Image
from datetime import datetime
from sklearn.model_selection import train_test_split
import ray
import multiprocessing as mp
import argparse

@ray.remote
def load_feature_movie(feature_id, feature_idx, num_features):
    print('feature {} of {}'.format(feature_idx+1, num_features))
    # load the slices for the feature
    slices_l = sorted(glob.glob("{}/feature-{}-slice-*.png".format(FEATURE_SLICES_DIR, feature_id)))
    feature_slices_l = []
    for feature_slice in slices_l:
        # load the image and generate the feature vector
        img = Image.open(feature_slice)
        x = image.img_to_array(img.resize((PIXELS_X,PIXELS_Y)))
        feature_slices_l.append(x)
    feature_slices = np.array(feature_slices_l)
    feature_slices = feature_slices.astype('float32') / 255.
    return feature_slices

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(0.8 * number_of_cores)
    return number_of_workers

#################################
parser = argparse.ArgumentParser(description='Train the convolutional LSTM autoencoder.')
parser.add_argument('-cts','--create_training_set', action='store_true', help='Create a new training set.')
parser.add_argument('-tnm','--train_model', action='store_true', help='Train a new model.')
args = parser.parse_args()


# EXPERIMENT_DIR = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test'
EXPERIMENT_DIR = '/home/daryl/experiments/dwm-test'
RUN_NAME = '190719_Hela_Ecoli_1to1_01'
ENCODED_FEATURES_DIR = '{}/encoded-features/{}'.format(EXPERIMENT_DIR, RUN_NAME)
FEATURE_SLICES_DIR = '{}/slices'.format(ENCODED_FEATURES_DIR)

# image dimensions
PIXELS_X = 128
PIXELS_Y = 128

# number of frames for the feature movies
NUMBER_OF_FRAMES = 20

# number of features to use
NUMBER_OF_FEATURES = 100

if args.create_training_set:
    # load the feature IDs that we processed earlier
    FEATURE_ID_LIST_FILE = '{}/feature_ids.pkl'.format(ENCODED_FEATURES_DIR)
    feature_list_df = pd.read_pickle(FEATURE_ID_LIST_FILE)
    features_l = feature_list_df.sample(n=NUMBER_OF_FEATURES).feature_id.tolist()

    print("setting up Ray")
    if not ray.is_initialized():
        ray.init(object_store_memory=20000000000, redis_max_memory=25000000000, num_cpus=number_of_workers())

    # load the feature slices
    print('loading the feature slices')
    feature_movies_l = ray.get([load_feature_movie.remote(feature_id=feature_id, feature_idx=feature_idx, num_features=len(features_l)) for feature_idx,feature_id in enumerate(features_l)])
    feature_movies = np.array(feature_movies_l)

    print("shutting down ray")
    ray.shutdown()

    # split the data into training, validation, test
    print('preparing the training set')
    X_train, X_test, _, _ = train_test_split(feature_movies, feature_movies, test_size=0.20)
    del feature_movies
    X_train = np.reshape(X_train, (len(X_train), NUMBER_OF_FRAMES, PIXELS_X, PIXELS_Y, 3))
    X_test = np.reshape(X_test, (len(X_test), NUMBER_OF_FRAMES, PIXELS_X, PIXELS_Y, 3))

    # divide X_test into validation and test
    split_num = int(len(X_test)/2)
    X_val = X_test[:split_num]
    X_test = X_test[split_num:]

    # save the training set
    print('storing the training set.')
    np.save('{}/train.npy'.format(ENCODED_FEATURES_DIR), X_train, allow_pickle=False)
    np.save('{}/test.npy'.format(ENCODED_FEATURES_DIR), X_test, allow_pickle=False)
    np.save('{}/validation.npy'.format(ENCODED_FEATURES_DIR), X_val, allow_pickle=False)
else:
    # load the training set
    print('loading the training set.')
    X_train = np.load('{}/train.npy'.format(ENCODED_FEATURES_DIR), allow_pickle=False)
    X_test = np.load('{}/test.npy'.format(ENCODED_FEATURES_DIR), allow_pickle=False)
    X_val = np.load('{}/validation.npy'.format(ENCODED_FEATURES_DIR), allow_pickle=False)

print('train {}, validation {}, test {}'.format(X_train.shape, X_val.shape, X_test.shape))

if args.train_model:
    # build the model
    seq = Sequential()

    # spatial encoder
    seq.add(TimeDistributed(Conv2D(filters=128, kernel_size=(11, 11), strides=4, padding="same"), batch_input_shape=(None, NUMBER_OF_FRAMES, PIXELS_X, PIXELS_Y, 3)))
    seq.add(LayerNormalization())

    seq.add(TimeDistributed(Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())

    # temporal encoder
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())

    # bottleneck
    seq.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization(name='encoded'))

    # temporal decoder
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())

    # spatial decoder
    seq.add(TimeDistributed(Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=2, padding="same")))
    seq.add(LayerNormalization())

    seq.add(TimeDistributed(Conv2DTranspose(filters=128, kernel_size=(11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())

    seq.add(TimeDistributed(Conv2D(filters=3, kernel_size=(11, 11), activation="sigmoid", padding="same")))

    print(seq.summary())

    encoder = Model(inputs=seq.inputs, outputs=seq.get_layer(name='encoded').output)

    seq.compile(loss='mse', optimizer=Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

    # set up TensorBoard
    logdir = "/tmp/tb/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    # train the model
    print('training the model')
    seq.fit(X_train, X_train, batch_size=5, epochs=20, verbose=1, validation_data=(X_val, X_val), callbacks=[tensorboard_callback], shuffle=False)

    # save the model
    print('saving the model')
    seq.save('{}/model'.format(ENCODED_FEATURES_DIR))
else:
    # load the model
    print('loading the model')
    seq = load_model('{}/model'.format(ENCODED_FEATURES_DIR))
    print(seq.summary())

# evaluate against the test set
X_test = [X_test[i] for i in range(X_test.shape[0])]  # convert from array of arrays to list of arrays
loss, acc = seq.evaluate(X_test, X_test, verbose = 0)
print('loss {}, accuracy {}'.format(loss, acc))
