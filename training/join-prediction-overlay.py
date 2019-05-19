import glob,os

PREDICTION_DIR = '~/Downloads/predictions'
BASE_DIR = './yolo-train-rt-1000-4200-15-may'
TEST_DIR = '{}/test'.format(BASE_DIR)
OVERLAY_DIR = '{}/overlay'.format(BASE_DIR)
COMBINED_DIR = '~/Downloads/combined'

file_list = sorted(glob.glob("{}/frame-*-tile-{}-mz-*.png".format(PREDICTION_DIR, TILE_ID)))

for prediction_file in file_list:
    basename = os.path.basename(prediction_file)
    overlay_name = "{}/{}".format(OVERLAY_DIR, basename)
    combined_name = "{}/combined-{}".format(COMBINED_DIR, basename)
    cmd = "convert {} {} +append -background darkgrey -splice 10x0+910+0 {}".format(prediction_file, overlay_name, combined_name)
    os.system(cmd)
