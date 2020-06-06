from flask import Flask, request, abort, jsonify, send_file, make_response
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tempfile
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import argparse
import os, shutil
import glob
import sys
from pathlib import Path
import time
import logging
import json

MS1_PEAK_DELTA = 0.1
MASS_DIFFERENCE_C12_C13_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.
INSTRUMENT_RESOLUTION = 40000.0
NUMBER_OF_STD_DEV_MZ = 3

MZ_MIN = 100.0
MZ_PER_TILE = 18.0

SERVER_URL = "http://spectra-server-lb-1653892276.ap-southeast-2.elb.amazonaws.com"

# This is the Flask server for the Via-based labelling tool for YOLO

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-tsn','--tile_set_name', type=str, default='tile-set', help='Name of the tile set.', required=False)
args = parser.parse_args()

# set up the logging directory
LOGGING_DIR = './logging'
if os.path.exists(LOGGING_DIR):
    shutil.rmtree(LOGGING_DIR)
os.makedirs(LOGGING_DIR)

# set up logging
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
# set the file handler
file_handler = logging.FileHandler('{}/{}.log'.format(LOGGING_DIR, parser.prog))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    logger.info("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the converted database directory exists
CONVERTED_DATABASE_DIR = '{}/converted-databases'.format(EXPERIMENT_DIR)
if not os.path.exists(CONVERTED_DATABASE_DIR):
    logger.info("The converted databases directory is required but doesn't exist: {}".format(CONVERTED_DATABASE_DIR))
    sys.exit(1)

# check the raw tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.tile_set_name)
if not os.path.exists(TILES_BASE_DIR):
    logger.info("The raw tiles base directory is required but does not exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

# create the Flask application
app = Flask(__name__)

def mz_centroid(_int_f, _mz_f):
    return ((_int_f/_int_f.sum()) * _mz_f).sum()

# ms1_peaks_a is a numpy array of [mz,intensity]
# returns a numpy array of [mz_centroid,summed_intensity]
def ms1_intensity_descent(ms1_peaks_a):
    # intensity descent
    ms1_peaks_l = []
    while len(ms1_peaks_a) > 0:
        # find the most intense point
        max_intensity_index = np.argmax(ms1_peaks_a[:,1])
        peak_mz = ms1_peaks_a[max_intensity_index,0]
        peak_mz_lower = peak_mz - MS1_PEAK_DELTA
        peak_mz_upper = peak_mz + MS1_PEAK_DELTA

        # get all the raw points within this m/z region
        peak_indexes = np.where((ms1_peaks_a[:,0] >= peak_mz_lower) & (ms1_peaks_a[:,0] <= peak_mz_upper))[0]
        if len(peak_indexes) > 0:
            mz_cent = mz_centroid(ms1_peaks_a[peak_indexes,1], ms1_peaks_a[peak_indexes,0])
            summed_intensity = ms1_peaks_a[peak_indexes,1].sum()
            ms1_peaks_l.append((mz_cent, summed_intensity))
            # remove the raw points assigned to this peak
            ms1_peaks_a = np.delete(ms1_peaks_a, peak_indexes, axis=0)
    return np.array(ms1_peaks_l)

MZ_BIN_WIDTH = 0.01978
PIXELS_PER_BIN = 1
PIXELS_FROM_EDGE = 10
MZ_FROM_EDGE = PIXELS_FROM_EDGE * PIXELS_PER_BIN * MZ_BIN_WIDTH  # number of pixels padding around the monoisotopic peak

MAX_NUMBER_OF_SULPHUR_ATOMS = 3
MAX_NUMBER_OF_PREDICTED_RATIOS = 6

S0_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=np.ndarray)
S0_r[1] = np.array([-0.00142320578040, 0.53158267080224, 0.00572776591574, -0.00040226083326, -0.00007968737684])
S0_r[2] = np.array([0.06258138406507, 0.24252967352808, 0.01729736525102, -0.00427641490976, 0.00038011211412])
S0_r[3] = np.array([0.03092092306220, 0.22353930450345, -0.02630395501009, 0.00728183023772, -0.00073155573939])
S0_r[4] = np.array([-0.02490747037406, 0.26363266501679, -0.07330346656184, 0.01876886839392, -0.00176688757979])
S0_r[5] = np.array([-0.19423148776489, 0.45952477474223, -0.18163820209523, 0.04173579115885, -0.00355426505742])
S0_r[6] = np.array([0.04574408690798, -0.05092121193598, 0.13874539944789, -0.04344815868749, 0.00449747222180])

S1_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=np.ndarray)
S1_r[1] = np.array([-0.01040584267474, 0.53121149663696, 0.00576913817747, -0.00039325152252, -0.00007954180489])
S1_r[2] = np.array([0.37339166598255, -0.15814640001919, 0.24085046064819, -0.06068695741919, 0.00563606634601])
S1_r[3] = np.array([0.06969331604484, 0.28154425636993, -0.08121643989151, 0.02372741957255, -0.00238998426027])
S1_r[4] = np.array([0.04462649178239, 0.23204790123388, -0.06083969521863, 0.01564282892512, -0.00145145206815])
S1_r[5] = np.array([-0.20727547407753, 0.53536509500863, -0.22521649838170, 0.05180965157326, -0.00439750995163])
S1_r[6] = np.array([0.27169670700251, -0.37192045082925, 0.31939855191976, -0.08668833166842, 0.00822975581940])

S2_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=np.ndarray)
S2_r[1] = np.array([-0.01937823810470, 0.53084210514216, 0.00580573751882, -0.00038281138203, -0.00007958217070])
S2_r[2] = np.array([0.68496829280011, -0.54558176102022, 0.44926662609767, -0.11154849560657, 0.01023294598884])
S2_r[3] = np.array([0.04215807391059, 0.40434195078925, -0.15884974959493, 0.04319968814535, -0.00413693825139])
S2_r[4] = np.array([0.14015578207913, 0.14407679007180, -0.01310480312503, 0.00362292256563, -0.00034189078786])
S2_r[5] = np.array([-0.02549241716294, 0.32153542852101, -0.11409513283836, 0.02617210469576, -0.00221816103608])
S2_r[6] = np.array([-0.14490868030324, 0.33629928307361, -0.08223564735018, 0.01023410734015, -0.00027717589598])

model_params = np.empty(MAX_NUMBER_OF_SULPHUR_ATOMS, dtype=np.ndarray)
model_params[0] = S0_r
model_params[1] = S1_r
model_params[2] = S2_r

# Find the ratio of H(peak_number)/H(peak_number-1) for peak_number=1..6
# peak_number = 0 refers to the monoisotopic peak
# number_of_sulphur = number of sulphur atoms in the molecule
def peak_ratio(monoisotopic_mass, peak_number, number_of_sulphur):
    ratio = None
    if (((1 <= peak_number <= 3) & (((number_of_sulphur == 0) & (498 <= monoisotopic_mass <= 3915)) |
                                    ((number_of_sulphur == 1) & (530 <= monoisotopic_mass <= 3947)) |
                                    ((number_of_sulphur == 2) & (562 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 4) & (((number_of_sulphur == 0) & (907 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (939 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (971 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 5) & (((number_of_sulphur == 0) & (1219 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (1251 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (1283 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 6) & (((number_of_sulphur == 0) & (1559 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (1591 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (1623 <= monoisotopic_mass <= 3978))))):
        beta0 = model_params[number_of_sulphur][peak_number][0]
        beta1 = model_params[number_of_sulphur][peak_number][1]
        beta2 = model_params[number_of_sulphur][peak_number][2]
        beta3 = model_params[number_of_sulphur][peak_number][3]
        beta4 = model_params[number_of_sulphur][peak_number][4]
        scaled_m = monoisotopic_mass / 1000.0
        ratio = beta0 + (beta1*scaled_m) + beta2*(scaled_m**2) + beta3*(scaled_m**3) + beta4*(scaled_m**4)
    return ratio

def calculate_monoisotopic_mass(monoisotopic_mz, charge):
    return (monoisotopic_mz * charge) - (PROTON_MASS * charge)

def calculate_peak_intensities(monoisotopic_mass, monoisotopic_intensity, isotopes, sulphurs):
    isotope_intensities = np.zeros(isotopes)
    isotope_intensities[0] = monoisotopic_intensity
    for i in range(1,isotopes):
        ratio = peak_ratio(monoisotopic_mass, i, sulphurs)
        if ratio is None:
            ratio = 0
        isotope_intensities[i] = ratio * isotope_intensities[i-1]
    return isotope_intensities

def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def standard_deviation(mz):
    FWHM = mz / INSTRUMENT_RESOLUTION
    return FWHM / 2.35482

def image_from_raw_data(data_coords, charge, isotopes):
    image_file_name = ""

    run_name = data_coords['run_name']
    frame_id = data_coords['frame_id']
    mz_lower = data_coords['mz_lower']
    mz_upper = data_coords['mz_upper']
    scan_lower = data_coords['scan_lower']
    scan_upper = data_coords['scan_upper']

    # get the raw points
    CONVERTED_DATABASE_NAME = "{}/exp-{}-run-{}-converted.sqlite".format(CONVERTED_DATABASE_DIR, args.experiment_name, run_name)
    db_conn = sqlite3.connect(CONVERTED_DATABASE_NAME)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {}".format(frame_id, mz_lower, mz_upper, scan_lower, scan_upper), db_conn)
    db_conn.close()

    # draw the chart
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(8)

    if len(raw_points_df) > 0:
        # perform intensity descent to consolidate the peaks
        raw_points_a = raw_points_df[['mz','intensity']].to_numpy()
        peaks_a = ms1_intensity_descent(raw_points_a)
        peaks_a = peaks_a[peaks_a[:,0].argsort()]  # sort by m/z

        # monoisotopic determined by the guide
        estimated_monoisotopic_mz = mz_lower + MZ_FROM_EDGE
        selected_peak_idx = find_nearest_idx(peaks_a[:,0], estimated_monoisotopic_mz)
        selected_peak_mz = peaks_a[selected_peak_idx,0]
        selected_peak_intensity = peaks_a[selected_peak_idx,1]
        estimated_monoisotopic_mass = calculate_monoisotopic_mass(estimated_monoisotopic_mz, charge)
        expected_peak_spacing_mz = MASS_DIFFERENCE_C12_C13_MZ / charge
        maximum_region_intensity = raw_points_df.intensity.max()

        isotope_intensities = np.empty(MAX_NUMBER_OF_SULPHUR_ATOMS, dtype=np.ndarray)
        for sulphurs in range(MAX_NUMBER_OF_SULPHUR_ATOMS):
            isotope_intensities[sulphurs] = calculate_peak_intensities(estimated_monoisotopic_mass, selected_peak_intensity, isotopes, sulphurs)

        plt.title('Peaks detected in the selected window')
        plt.margins(0.06)
        plt.rcParams['axes.linewidth'] = 0.1

        # plot the isotopes in the m/z dimension
        ax1 = plt.subplot2grid((2, len(peaks_a)), (0, 0), colspan=len(peaks_a))
        for sulphurs in range(MAX_NUMBER_OF_SULPHUR_ATOMS):
            for isotope in range(isotopes):
                rect_base_mz = selected_peak_mz + (isotope * expected_peak_spacing_mz) - (MS1_PEAK_DELTA/2)
                peak_intensity = isotope_intensities[sulphurs][isotope]
                rect = patches.Rectangle((rect_base_mz,0), MS1_PEAK_DELTA, peak_intensity, edgecolor='silver', linewidth=1.0, facecolor='silver', alpha=0.3, label='theoretical')
                ax1.add_patch(rect)
        markerline, stemlines, baseline = ax1.stem(peaks_a[:,0], peaks_a[:,1], colors[1], use_line_collection=True, label='peak summed from raw data')
        plt.setp(markerline, linewidth=1, color=colors[1])
        plt.setp(markerline, markersize=3, color=colors[1])
        plt.setp(stemlines, linewidth=1, color=colors[1])
        plt.setp(baseline, linewidth=0.25, color=colors[7])
        baseline.set_xdata([0,1])
        baseline.set_transform(plt.gca().get_yaxis_transform())
        plt.xlabel('m/z')
        plt.ylabel('intensity')

        # plot the isotopes in the CCS dimension
        for peak_idx,peak in enumerate(peaks_a):
            ax = plt.subplot2grid((2, len(peaks_a)), (1, peak_idx), colspan=1)

            peak_mz = peaks_a[peak_idx][0]
            mz_delta = standard_deviation(peak_mz) * NUMBER_OF_STD_DEV_MZ
            peak_mz_lower = peak_mz - mz_delta
            peak_mz_upper = peak_mz + mz_delta
            peak_points_df = raw_points_df[(raw_points_df.mz >= peak_mz_lower) & (raw_points_df.mz <= peak_mz_upper)]

            ax.scatter(peak_points_df.intensity, peak_points_df.scan, marker='o', color='tab:orange', lw=0, s=30, alpha=0.8)
            plt.ylim([scan_upper,scan_lower])
            plt.xlim([0,maximum_region_intensity])
            # turn off tick labels
            if peak_idx > 0:
                ax.set_yticklabels([])
                ax.set_yticks([])
            else:
                plt.ylabel('scan')
            ax.set_xticklabels([])
            ax.set_xticks([])

    # save the chart as an image
    image_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    logger.info("image file: {}".format(image_file_name))
    plt.savefig(image_file_name, bbox_inches='tight')
    plt.close()

    return image_file_name

def tile_coords_to_data_coords(tile_name, tile_width, tile_height, region_x, region_y, region_width, region_height, canvas_scale):
    # determine the tile id and frame id from the tile URL
    elements = tile_name.split('/')
    run_name = elements[5]
    tile_id = int(elements[7])
    frame_id = int(elements[9])

    tile_mz_lower = MZ_MIN + (tile_id * MZ_PER_TILE)
    tile_mz_upper = tile_mz_lower + MZ_PER_TILE

    # scale the tile coordinates by the canvas scale
    region_x = region_x * canvas_scale
    region_y = region_y * canvas_scale
    region_width = region_width * canvas_scale
    region_height = region_height * canvas_scale

    region_mz_lower = ((region_x / tile_width) * (tile_mz_upper - tile_mz_lower)) + tile_mz_lower
    region_mz_upper = (((region_x + region_width) / tile_width) * (tile_mz_upper - tile_mz_lower)) + tile_mz_lower
    region_scan_lower = region_y
    region_scan_upper = region_y + region_height

    d = {}
    d['run_name'] = run_name
    d['frame_id'] = frame_id
    d['mz_lower'] = region_mz_lower
    d['mz_upper'] = region_mz_upper
    d['scan_lower'] = region_scan_lower
    d['scan_upper'] = region_scan_upper
    return d

# create the indexes we need for this application
def create_indexes(db_file_name):
    db_conn = sqlite3.connect(db_file_name)
    src_c = db_conn.cursor()
    src_c.execute("create index if not exists idx_spectra_server on frames (frame_id,mz,scan)")
    db_conn.close()

@app.route('/spectra', methods=['POST'])
def spectra():
    if request.method == 'POST':
        start_time = time.time()
        # extract payload
        action = request.json['action']
        x = request.json['x']
        y = request.json['y']
        width = request.json['width']
        height = request.json['height']
        tile_name = request.json['tile_name']
        tile_width = request.json['tile_width']
        tile_height = request.json['tile_height']
        canvas_scale = request.json['canvas_scale']
        attributes = request.json['attributes']
        if len(attributes) > 0:
            charge = int(''.join(ch for ch in attributes['charge'] if ch.isdigit()))
            isotopes = int(attributes['isotopes'])
        else:
            charge = 0
            isotopes = 0
        logger.info(request.json)
        # convert to data coordinates
        data_coords = tile_coords_to_data_coords(tile_name, tile_width, tile_height, x, y, width, height, canvas_scale)
        logger.info("data coords: {}".format(data_coords))
        # create image
        filename = image_from_raw_data(data_coords, charge, isotopes)
        response = send_file(filename)
        stop_time = time.time()
        logger.info("served the request in {} seconds".format(round(stop_time-start_time,1)))
        return response
    else:
        abort(400)

# retrieve the tile-frame for this run
@app.route('/tile/run/<string:run_name>/tile/<int:tile_id>/frame/<int:frame_id>')
def tile(run_name, tile_id, frame_id):
    # determine the file name for this tile
    file_list = glob.glob("{}/run-{}-frame-{}-tile-{}.png".format(TILES_BASE_DIR, run_name, frame_id, tile_id))
    if len(file_list) > 0:
        tile_file_name = file_list[0]
        # send it to the client
        logger.info("serving {}".format(tile_file_name))
        response = send_file(tile_file_name)
        return response
    else:
        logger.info("tile for tile {} frame {} run {} does not exist in {}".format(tile_id, frame_id, run_name, TILES_BASE_DIR))
        abort(400)

# retrieve the list of tile URLs for a specific tile index
@app.route('/tile-list/run/<string:run_name>/tile/<int:tile_id>')
def tile_list(run_name, tile_id):
    tile_list = sorted(glob.glob("{}/run-{}-frame-*-tile-{}.png".format(TILES_BASE_DIR, run_name, tile_id)))
    if len(tile_list) > 0:
        temp_file_name = tempfile.NamedTemporaryFile(suffix='.txt').name
        with open(temp_file_name, 'w') as filehandle:
            for idx,tile_file_path in enumerate(tile_list):
                tile_file_name = os.path.basename(tile_file_path)
                # get the frame id for this tile
                frame_id = int(tile_file_name.split('-')[3])
                # create the URL
                logger.info(SERVER_URL)
                tile_url = "{}/tile/run/{}/tile/{}/frame/{}".format(SERVER_URL, run_name, tile_id, frame_id)
                logger.info(tile_url)
                if idx < len(tile_list):
                    filehandle.write('{}\n'.format(tile_url))
                else:
                    filehandle.write('{}'.format(tile_url))
        response = send_file(temp_file_name)
        os.remove(temp_file_name)
        return response
    else:
        logger.info("tiles in the series for run {} tile index {} do not exist in {}".format(run_name, tile_id, TILES_BASE_DIR))
        abort(400)

# retrieve the via annotation tool
@app.route('/via')
def via():
    via_file_name = 'otf-peak-detect/yolo-feature-detection/via/via.html'
    home = str(Path.home())
    response = send_file("{}/{}".format(home, via_file_name))
    return response

# set the server URL so the server can generate a list of URLs that can be referenced from the internet by the Via client
@app.route('/server_url/<string:server_url>')
def set_server_url(server_url):
    global SERVER_URL
    SERVER_URL = "http://{}".format(server_url)
    logger.info("server URL is now {}".format(SERVER_URL))
    return make_response()

@app.route('/index.html')
def index():
    resp = make_response()
    resp.status_code = 200
    return resp


if __name__ == '__main__':
    # load the tile set metadata file
    tile_set_metadata_file_name = '{}/metadata.json'.format(TILES_BASE_DIR)
    if os.path.isfile(tile_set_metadata_file_name):
        with open(tile_set_metadata_file_name) as json_file:
            tile_set_metadata = json.load(json_file)
    else:
        print("Could not find the tile set's metadata file: {}".format(tile_set_metadata_file_name))
        sys.exit(1)
    # print some information about the specified tile set
    print("tile set {}: {}".format(args.tile_set_name, tile_set_metadata['arguments']))

    # set up indexes on the converted databases in this tile set
    for run_name in tile_set_metadata['arguments']['run_names']:
        CONVERTED_DATABASE_NAME = "{}/exp-{}-run-{}-converted.sqlite".format(CONVERTED_DATABASE_DIR, args.experiment_name, run_name)
        logger.info("setting up indexes on {}".format(CONVERTED_DATABASE_NAME))
        create_indexes(CONVERTED_DATABASE_NAME)

    logger.info("running the server")
    app.run(host='0.0.0.0')
