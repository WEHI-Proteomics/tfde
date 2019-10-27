from flask import Flask, request, abort, jsonify, send_file
from flask_cors import CORS
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
import os
import glob

MS1_PEAK_DELTA = 0.1
MASS_DIFFERENCE_C12_C13_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.
INSTRUMENT_RESOLUTION = 40000.0
NUMBER_OF_STD_DEV_MZ = 3

MZ_MIN = 100.0
MZ_PER_TILE = 18.0

# This is the Flask server for the Via-based labelling tool for YOLO
# Example: python ./otf-peak-detect/yolo/via/spectra-display-server.py -eb ~/Downloads/experiments -en 190719_Hela_Ecoli -rn 190719_Hela_Ecoli_1to3_06

parser = argparse.ArgumentParser(description='Create the tiles from raw data.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-url', '--server_url', type=str, default='http://127.0.0.1:5000', help='The URL used to access the server from the client.', required=False)
args = parser.parse_args()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the converted database exists
CONVERTED_DATABASE_NAME = "{}/converted-databases/{}-converted.sqlite".format(EXPERIMENT_DIR, args.run_name)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database is required but doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

# check the tiles base directory exists
TILES_BASE_DIR = '{}/tiles/{}'.format(EXPERIMENT_DIR, args.run_name)
if not os.path.exists(TILES_BASE_DIR):
    print("The tiles base directory is required but doesn't exist: {}".format(TILES_BASE_DIR))
    sys.exit(1)

app = Flask(__name__)
# CORS(app) # This will enable CORS for all routes

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

    frame_id = data_coords['frame_id']
    mz_lower = data_coords['mz_lower']
    mz_upper = data_coords['mz_upper']
    scan_lower = data_coords['scan_lower']
    scan_upper = data_coords['scan_upper']

    # get the raw points
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

        for peak_idx,peak in enumerate(peaks_a):
            ax = plt.subplot2grid((2, len(peaks_a)), (1, peak_idx), colspan=1)

            peak_mz = peaks_a[peak_idx][0]
            mz_delta = standard_deviation(peak_mz) * NUMBER_OF_STD_DEV_MZ
            peak_mz_lower = peak_mz - mz_delta
            peak_mz_upper = peak_mz + mz_delta

            peak_points_df = raw_points_df[(raw_points_df.mz >= peak_mz_lower) & (raw_points_df.mz <= peak_mz_upper)]
            summed_df = pd.DataFrame(peak_points_df.groupby(['scan'])['intensity'].sum().reset_index())

            ax.plot(summed_df.intensity, summed_df.scan, linestyle='-', linewidth=0.5, color='tab:brown')
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
    print("image file: {}".format(image_file_name))
    plt.savefig(image_file_name, bbox_inches='tight')
    plt.close()

    return image_file_name

def tile_coords_to_data_coords(tile_name, tile_width, tile_height, region_x, region_y, region_width, region_height, canvas_scale):
    # determine the tile id and frame id from the tile URL
    elements = tile_name.split('/')
    tile_id = int(elements[4])
    frame_id = int(elements[6])

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
    d['frame_id'] = frame_id
    d['mz_lower'] = region_mz_lower
    d['mz_upper'] = region_mz_upper
    d['scan_lower'] = region_scan_lower
    d['scan_upper'] = region_scan_upper
    return d

@app.route('/spectra', methods=['POST'])
def spectra():
    if request.method == 'POST':
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
        print(request.json)
        # convert to data coordinates
        data_coords = tile_coords_to_data_coords(tile_name, tile_width, tile_height, x, y, width, height, canvas_scale)
        print("data coords: {}".format(data_coords))
        # create image
        filename = image_from_raw_data(data_coords, charge, isotopes)
        response = send_file(filename)
        return response
    else:
        abort(400)

# retrieve the tile-frame for this run
@app.route('/tile/<int:tile_id>/frame/<int:frame_id>')
def tile(tile_id, frame_id):
    # determine the file name for this tile
    file_list = glob.glob("{}/tile-{}/frame-{}-tile-{}*.png".format(TILES_BASE_DIR, tile_id, frame_id, tile_id))
    if len(file_list) > 0:
        tile_file_name = file_list[0]
        # send it to the client
        print("serving {}".format(tile_file_name))
        response = send_file(tile_file_name)
        return response
    else:
        print("tile for tile {} frame {} does not exist in {}".format(tile_id, frame_id, TILES_BASE_DIR))
        abort(400)

# retrieve the list of tile URLs for a specific tile index
@app.route('/tile-list/<int:tile_id>')
def tile_list(tile_id):
    tile_list = sorted(glob.glob("{}/tile-{}/*.png".format(TILES_BASE_DIR, tile_id)))
    if len(tile_list) > 0:
        temp_file_name = tempfile.NamedTemporaryFile(suffix='.txt').name
        with open(temp_file_name, 'w') as filehandle:
            for idx,tile_file_path in enumerate(tile_list):
                tile_file_name = os.path.basename(tile_file_path)
                print(tile_file_name)
                # get the frame id for this tile
                frame_id = int(tile_file_name.split('-')[1])
                # create the URL
                tile_url = "{}/tile/{}/frame/{}".format(args.server_url, tile_id, frame_id)
                if idx < len(tile_list):
                    filehandle.write('{}\n'.format(tile_url))
                else:
                    filehandle.write('{}'.format(tile_url))
        response = send_file(temp_file_name)
        return response
    else:
        print("tiles in the series for tile index {} do not exist in {}".format(tile_id, TILES_BASE_DIR))
        abort(400)

# retrieve the via annotation tool
@app.route('/via')
def via():
    via_file_name = 'Documents/otf-peak-detect/yolo/via/via.html'
    root_dir = os.path.dirname(os.getcwd())
    response = send_file("{}/{}".format(root_dir, via_file_name))
    return response


if __name__ == '__main__':
    app.run()
