import pandas as pd
import sqlite3
import numpy as np
from PIL import Image
from matplotlib import colors, cm, pyplot as plt
import os
import shutil
import pickle

# image dimensions
PIXELS_X = 224
PIXELS_Y = 224

# number of frames for the feature movies
NUMBER_OF_FRAMES = 20

# frame types for PASEF mode
FRAME_TYPE_MS1 = 0
FRAME_TYPE_MS2 = 8

def pixel_xy(mz, scan, mz_lower, mz_upper, scan_lower, scan_upper):
    x_pixels_per_mz = (PIXELS_X-1) / (mz_upper - mz_lower)
    y_pixels_per_scan = (PIXELS_Y-1) / (scan_upper - scan_lower)
    
    pixel_x = int((mz - mz_lower) * x_pixels_per_mz)
    pixel_y = int((scan - scan_lower) * y_pixels_per_scan)
    return (pixel_x, pixel_y)

# determine the mapping between the percolator index and the run file name
def get_percolator_run_mapping(mapping_file_name):
    df = pd.read_csv(mapping_file_name)
    mapping_l = [tuple(r) for r in df.to_numpy()]
    return mapping_l

##################################
# EXPERIMENT_DIR = '/Users/darylwilding-mcbride/Downloads/experiments/dwm-test'
EXPERIMENT_DIR = '/home/daryl/experiments/dwm-test'
PERCOLATOR_OUTPUT_DIR = "{}/recalibrated-percolator-output".format(EXPERIMENT_DIR)
MAPPING_FILE_NAME = "{}/percolator-idx-mapping.csv".format(PERCOLATOR_OUTPUT_DIR)
RUN_NAME = '190719_Hela_Ecoli_1to1_01'
FEATURES_DIR = '{}/features/{}'.format(EXPERIMENT_DIR, RUN_NAME)
CONVERTED_DB = '{}/converted-databases/exp-dwm-test-run-{}-converted.sqlite'.format(EXPERIMENT_DIR, RUN_NAME)

# check the encoded features directory
ENCODED_FEATURES_DIR = '{}/encoded-features/{}'.format(EXPERIMENT_DIR, RUN_NAME)
if os.path.exists(ENCODED_FEATURES_DIR):
    shutil.rmtree(ENCODED_FEATURES_DIR)
os.makedirs(ENCODED_FEATURES_DIR)

# make the slices directory
FEATURE_SLICES_DIR = '{}/slices'.format(ENCODED_FEATURES_DIR)
os.makedirs(FEATURE_SLICES_DIR)

# create the colour mapping
colour_map = plt.get_cmap('rainbow')
norm = colors.LogNorm(vmin=1, vmax=1000, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

# get the frame properties
db_conn = sqlite3.connect(CONVERTED_DB)
frame_properties_df = pd.read_sql_query('select Id,Time from frame_properties where MsMsType == {}'.format(FRAME_TYPE_MS1), db_conn)
db_conn.close()

# read the peptide sequence identifications for the experiment
df = pd.read_pickle('{}/sequence-library/percolator-id-feature-mapping.pkl'.format(EXPERIMENT_DIR))
# only keep the sequences with non-rubbish peptide masses
df = df[df['peptide mass'] > 0]

# add the run name
mapping_l = get_percolator_run_mapping(MAPPING_FILE_NAME)
mapping_df = pd.DataFrame(mapping_l, columns=['file_idx','run_name'])
combined_df = pd.merge(df, mapping_df, on='file_idx', how='inner')

# keep a record of the feature IDs we process
features_l = []

# filter by the run we're interested in
combined_df = combined_df[combined_df.run_name == RUN_NAME]
# combined_df = combined_df.sample(n=100)  # limit the number of features for debugging purposes

for precursor_idx,precursor in enumerate(combined_df.itertuples()):
    precursor_id = precursor.precursor_id
    feature_pkl = '{}/exp-dwm-test-run-{}-features-precursor-{}.pkl'.format(FEATURES_DIR, RUN_NAME, precursor_id)
    print("processing precursor {} of {}".format(precursor_idx+1, len(combined_df)))

    # load the features for this precursor
    features_df = pd.read_pickle(feature_pkl)
    # for each feature, generate image slices for its cuboid
    for feature in features_df.itertuples():
        feature_id = feature.feature_id
        print("feature ID {}".format(feature_id))

        # determine the feature cuboid dimensions
        mz_lower = feature.envelope[0][0] - 0.5
        mz_upper = feature.envelope[-1][0] + 0.5
        scan_lower = feature.scan_lower
        scan_upper = feature.scan_upper
        rt_apex = feature.rt_apex
        rt_lower = feature.rt_lower
        rt_upper = feature.rt_upper
        monoisotopic_mz = feature.monoisotopic_mz

        # get the raw data for this feature
        db_conn = sqlite3.connect(CONVERTED_DB)
        raw_df = pd.read_sql_query('select mz,scan,intensity,frame_id,retention_time_secs from frames where mz >= {} and mz <= {} and scan >= {} and scan <= {} and frame_type == {} and retention_time_secs >= {} and retention_time_secs <= {}'.format(mz_lower, mz_upper, scan_lower, scan_upper, FRAME_TYPE_MS1, rt_lower, rt_upper), db_conn)
        if len(raw_df) == 0:
            print("found no raw points for feature {}".format(feature_id))
        db_conn.close()

        if len(raw_df) > 0:
            # calculate the raw point coordinates in scaled pixels
            pixel_df = pd.DataFrame(raw_df.apply(lambda row: pixel_xy(row.mz, row.scan, monoisotopic_mz-0.5, monoisotopic_mz+(10*0.5), scan_lower, scan_upper), axis=1).tolist(), columns=['pixel_x','pixel_y'])
            raw_pixel_df = pd.concat([raw_df, pixel_df], axis=1)

            # sum the intensity of raw points that have been assigned to each pixel
            pixel_intensity_df = raw_pixel_df.groupby(by=['frame_id', 'pixel_x', 'pixel_y'], as_index=False).intensity.sum()

            # calculate the colour to represent the intensity
            colours_l = []
            for i in pixel_intensity_df.intensity.unique():
                colours_l.append((i, colour_map(norm(i), bytes=True)[:3]))
            colours_df = pd.DataFrame(colours_l, columns=['intensity','colour'])
            pixel_intensity_df = pd.merge(pixel_intensity_df, colours_df, how='left', left_on=['intensity'], right_on=['intensity'])

            # get the frame IDs closest to the RT apex
            movie_frame_ids = frame_properties_df.iloc[(frame_properties_df['Time'] - rt_apex).abs().argsort()[:NUMBER_OF_FRAMES]].sort_values(by=['Time'], ascending=[True], inplace=False).Id.tolist()

            # write out the images to files
            feature_slice = 0
            for movie_frame_id in movie_frame_ids:
                frame_df = pixel_intensity_df[(pixel_intensity_df.frame_id == movie_frame_id)]
                # create an intensity array
                tile_im_array = np.zeros([PIXELS_Y, PIXELS_X, 3], dtype=np.uint8)  # container for the image
                for r in zip(frame_df.pixel_x, frame_df.pixel_y, frame_df.colour):
                    x = r[0]
                    y = r[1]
                    c = r[2]
                    tile_im_array[y,x,:] = c

                # create an image of the intensity array
                feature_slice += 1
                tile = Image.fromarray(tile_im_array, 'RGB')
                tile_file_name = '{}/feature-{}-slice-{:03d}.png'.format(FEATURE_SLICES_DIR, feature_id, feature_slice)
                tile.save(tile_file_name)

            # add this to the list
            features_l.append((precursor.sequence, precursor.charge_x, feature_id))

# save the list of feature IDs we processed
FEATURE_ID_LIST_FILE = '{}/feature_ids.pkl'.format(ENCODED_FEATURES_DIR)
feature_list_df = pd.DataFrame(features_l, columns=['sequence','charge','feature_id'])
feature_list_df.to_pickle(FEATURE_ID_LIST_FILE)
