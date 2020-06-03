import time
import zmq
import random
import pandas as pd
from matplotlib import colors, cm, pyplot as plt
import numpy as np
from PIL import Image

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
    assert (mz >= MZ_MIN) and (mz <= MZ_MAX), "m/z not in range"

    tile_id = int((mz - MZ_MIN) / MZ_PER_TILE)
    pixel_x = int(((mz - MZ_MIN) % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return (tile_id, pixel_x)

def interpolate_pixels(tile_im_array):
    z = np.array([0,0,0])  # empty pixel check
    for x in range(4, PIXELS_X-4):
        for y in range(4, PIXELS_Y-4):
            c = tile_im_array[y,x]
            if (not c.any()) and tile_im_array[y-4:y+4,x-1:x+1].any():  # this is a zero pixel and there is a non-zero pixel in this region
                # build the list of this pixel's neighbours to average
                n = []
                n.append(tile_im_array[y+1,x-1])
                n.append(tile_im_array[y+1,x])
                n.append(tile_im_array[y+1,x+1])

                n.append(tile_im_array[y-1,x-1])
                n.append(tile_im_array[y-1,x])
                n.append(tile_im_array[y-1,x+1])

                n.append(tile_im_array[y+2,x-1])
                n.append(tile_im_array[y+2,x])
                n.append(tile_im_array[y+2,x+1])

                n.append(tile_im_array[y-2,x-1])
                n.append(tile_im_array[y-2,x])
                n.append(tile_im_array[y-2,x+1])

                n.append(tile_im_array[y+3,x-1])
                n.append(tile_im_array[y+3,x])
                n.append(tile_im_array[y+3,x+1])

                n.append(tile_im_array[y-3,x-1])
                n.append(tile_im_array[y-3,x])
                n.append(tile_im_array[y-3,x+1])

                n.append(tile_im_array[y+4,x-1])
                n.append(tile_im_array[y+4,x])
                n.append(tile_im_array[y+4,x+1])

                n.append(tile_im_array[y-4,x-1])
                n.append(tile_im_array[y-4,x])
                n.append(tile_im_array[y-4,x+1])

                n.append(tile_im_array[y,x-1])
                n.append(tile_im_array[y,x+1])

                # set the pixel's colour to be the channel-wise mean of its neighbours
                neighbours_a = np.array(n)
                tile_im_array[y,x] = np.mean(neighbours_a, axis=0)

    return tile_im_array

def render_frame(frame_id, frame_df):
    # assign a tile_id and a pixel x value to each raw point
    tile_pixels_df = pd.DataFrame(frame_df.apply(lambda row: tile_pixel_x_from_mz(row.mz), axis=1).tolist(), columns=['tile_id', 'pixel_x'])
    raw_points_df = pd.concat([frame_df, tile_pixels_df], axis=1)
    pixel_intensity_df = raw_points_df.groupby(by=['tile_id', 'pixel_x', 'scan'], as_index=False).intensity.sum()

    # create the colour map to convert intensity to colour
    colour_map = plt.get_cmap('rainbow')
    norm = colors.LogNorm(vmin=MINIMUM_PIXEL_INTENSITY, vmax=MAXIMUM_PIXEL_INTENSITY, clip=True)  # aiming to get good colour variation in the lower range, and clipping everything else

    # calculate the colour to represent the intensity
    colours_l = []
    for i in pixel_intensity_df.intensity.unique():
        colours_l.append((i, colour_map(norm(i), bytes=True)[:3]))
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

        # fill in zero pixels with interpolated values
        # tile_im_array = interpolate_pixels(tile_im_array)

        # create an image of the intensity array
        tile = Image.fromarray(tile_im_array, 'RGB')
        tile.save('{}/frame-{}-tile-{}.png'.format(TILES_DIR, frame_id, tile_idx))

def consumer():
    consumer_id = random.randrange(1,10005)
    print("consumer {}".format(consumer_id))
    context = zmq.Context()
    # receive work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect("tcp://127.0.0.1:5557")
    
    while True:
        message = consumer_receiver.recv_json()
        start_run = time.time()
        frame_id = message['frame_id']
        base_name = message['base_name']
        print("rendering frame {}".format(frame_id))

        frame_file_name = '{}/{}'.format(PUBLISHED_FRAMES_DIR, base_name)
        frame_df = pd.read_pickle(frame_file_name)
        render_frame(frame_id, frame_df)
        stop_run = time.time()
        print("processed message in {} seconds".format(round(stop_run-start_run,1)))

consumer()
