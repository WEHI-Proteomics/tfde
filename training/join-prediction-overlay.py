import glob,os

PREDICTION_DIR = './darknet/result_img'
BASE_DIR = './yolo-train-rt-1000-4200-15-may'
TEST_DIR = '{}/test'.format(BASE_DIR)
OVERLAY_DIR = '{}/overlay'.format(BASE_DIR)
COMBINED_DIR = './combined'
TILE_ID = 33

file_list = sorted(glob.glob("{}/prediction-frame-*-tile-{}-mz-*.jpg".format(PREDICTION_DIR, TILE_ID)))

for prediction_file in file_list:
    basename = os.path.splitext(os.path.basename(prediction_file).replace('prediction-', ''))[0]
    print("processing {}".format(basename))
    overlay_name = "{}/{}.png".format(OVERLAY_DIR, basename)
    combined_name = "{}/combined-{}.png".format(COMBINED_DIR, basename)
    cmd = "convert {} {} +append -background darkgrey -splice 10x0+910+0 {}".format(prediction_file, overlay_name, combined_name)
    os.system(cmd)
