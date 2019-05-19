import glob,os

TEST_DIR = './yolo-train-rt-1000-4200-15-may/test/'
TARGET_DIR = 'data/peptides/test'
TILE_ID = 33
FILE_LIST_FILENAME = './test-files.txt'

with open(FILE_LIST_FILENAME, 'w') as f:
    for file in sorted(glob.glob("{}/frame-*-tile-{}-mz-*.png".format(TEST_DIR, TILE_ID))):
        f.write('{}/{}\n'.format(TARGET_DIR, os.path.basename(file)))
