# This script gets all the tiles for a particular section of m/z and creates a file
# containing their filenames for the purpose of bulk inference.

import glob,os

EXPERIMENT_NAME = 'dwm-test'
RUN_NAME = '190719_Hela_Ecoli_1to1_01'
TILE_ID = 34
TILE_DIR = '/data/experiments/{}/tiles/{}/tile-{}'.format(EXPERIMENT_NAME, RUN_NAME, TILE_ID)

OUTPUT_FILENAME = './tile-{}-files.txt'.format(TILE_ID)  # the name of the file containing the list to process

file_list = sorted(glob.glob("{}/frame-*-tile-{}-mz-*.png".format(TILE_DIR, TILE_ID)))

with open(OUTPUT_FILENAME, 'w') as f:
    for file in file_list:
        f.write('{}/{}\n'.format(TILE_DIR, os.path.basename(file)))
