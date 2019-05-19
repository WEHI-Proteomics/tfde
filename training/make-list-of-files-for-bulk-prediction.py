import glob,os

TEST_DIR = '~/yolo-train/test/'
TARGET_DIR = 'data/peptides/test'
TILE_ID = 33
FILE_LIST_FILENAME = '~/test-files.txt'

with open(FILE_LIST_FILENAME, 'w') as f:
    for file in sorted(glob.glob("{}/frame-*-tile-{}-mz-*.png".format(PREASSIGNED_DIR, TILE_ID))):
        f.write('{}/{}\n'.format(TARGET_DIR, os.path.basename(file)))
