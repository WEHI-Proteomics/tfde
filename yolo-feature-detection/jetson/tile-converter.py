import time
import zmq
import random
import pandas as pd
from matplotlib import colors, cm, pyplot as plt
import numpy as np
from PIL import Image
import os, shutil

PUBLISHED_FRAMES_DIR = '/data/published-frames'
TILES_DIR = '/data/tiles'

PIXELS_X = 910
PIXELS_Y = 910  # equal to the number of scan lines
MZ_MIN = 100.0
MZ_MAX = 1700.0
SCAN_MAX = PIXELS_Y
SCAN_MIN = 1
MZ_PER_TILE = 18.0
TILES_PER_FRAME = int((MZ_MAX - MZ_MIN) / MZ_PER_TILE) + 1
MIN_TILE_IDX = 0
MAX_TILE_IDX = TILES_PER_FRAME-1

MINIMUM_PIXEL_INTENSITY = 1
MAXIMUM_PIXEL_INTENSITY = 1000

def tile_pixel_x_from_mz(mz):
    mz_adj = mz - MZ_MIN
    tile_id = int(mz_adj / MZ_PER_TILE)
    pixel_x = int((mz_adj % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return (tile_id, pixel_x)

def render_frame(frame_id, frame_df, colour_map, norm):
    # assign a tile_id and a pixel x value to each raw point
    tile_pixel_l = []
    for row in zip(frame_df.mz):
        tile_pixel_l.append(tile_pixel_x_from_mz(row[0]))
    tile_pixels_df = pd.DataFrame(tile_pixel_l, columns=['tile_id','pixel_x'])
    raw_points_df = pd.concat([frame_df, tile_pixels_df], axis=1)
    pixel_intensity_df = raw_points_df.groupby(by=['tile_id', 'pixel_x', 'scan'], as_index=False).intensity.sum()

    # calculate the colour to represent the intensity
    colours_l = list(zip(list(pixel_intensity_df.intensity.unique()), list(map(tuple, colour_map(norm(pixel_intensity_df.intensity.unique()), bytes=True)[:,:3]))))
    colours_df = pd.DataFrame(colours_l, columns=['intensity','colour'])    
    pixel_intensity_df = pd.merge(pixel_intensity_df, colours_df, how='left', left_on=['intensity'], right_on=['intensity'])

    # extract the pixels for the frame for the specified tiles
    for tile_idx in range(MIN_TILE_IDX, MAX_TILE_IDX+1):
        tile_df = pixel_intensity_df[(pixel_intensity_df.tile_id == tile_idx)]

        # create an intensity array
        tile_im_array = np.zeros([PIXELS_Y+1, PIXELS_X+1, 3], dtype=np.uint8)  # container for the image
        for r in zip(tile_df.pixel_x, tile_df.scan, tile_df.colour):
            x = r[0]
            y = r[1]
            c = r[2]
            tile_im_array[y,x,:] = c

        # save the intensity array
        np.save('{}/frame-{}-tile-{}.png'.format(TILES_DIR, frame_id, tile_idx), tile_im_array)

def consumer():
    consumer_id = random.randrange(1,10005)
    print("consumer {}".format(consumer_id))
    context = zmq.Context()
    # receive work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://127.0.0.1:5557")

    # create the colour map to convert intensity to colour
    colour_map = plt.get_cmap('rainbow')
    norm = colors.LogNorm(vmin=MINIMUM_PIXEL_INTENSITY, vmax=MAXIMUM_PIXEL_INTENSITY, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

    # make a fresh output directory
    if os.path.exists(TILES_DIR):
        shutil.rmtree(TILES_DIR)
    os.makedirs(TILES_DIR)

    timings_l = []
    
    while True:
        message = consumer_receiver.recv_json()
        start_run = time.time()
        frame_id = message['frame_id']
        base_name = message['base_name']

        frame_file_name = '{}/{}'.format(PUBLISHED_FRAMES_DIR, base_name)
        frame_df = pd.read_pickle(frame_file_name)
        render_frame(frame_id, frame_df, colour_map, norm)
        stop_run = time.time()

        time_taken = stop_run-start_run
        timings_l.append(time_taken)
        
        print("processed frame {} in {} seconds - mean time {}".format(frame_id, round(time_taken,1), round(np.mean(timings_l),1)))

consumer()
