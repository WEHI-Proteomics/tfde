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

CONVERTED_DATABASE = '/Users/darylwilding-mcbride/Downloads/190719_Hela_Ecoli/converted/190719_Hela_Ecoli_1to1_01-converted.sqlite'
IMAGE_X = 400
IMAGE_Y = 300

MS1_PEAK_DELTA = 0.1
MASS_DIFFERENCE_C12_C13_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

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
MZ_FROM_EDGE = PIXELS_FROM_EDGE * PIXELS_PER_BIN * MZ_BIN_WIDTH

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

def image_from_raw_data(data_coords, charge, isotopes):
    image_file_name = ""

    frame_id = data_coords['frame_id']
    mz_lower = data_coords['mz_lower']
    mz_upper = data_coords['mz_upper']
    scan_lower = data_coords['scan_lower']
    scan_upper = data_coords['scan_upper']

    # get the raw points
    db_conn = sqlite3.connect(CONVERTED_DATABASE)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {}".format(frame_id, mz_lower, mz_upper, scan_lower, scan_upper), db_conn)
    db_conn.close()

    if len(raw_points_df) > 0:

        # perform intensity descent to consolidate the peaks
        raw_points_a = raw_points_df[['mz','intensity']].to_numpy()
        peaks_a = ms1_intensity_descent(raw_points_a)

        # monoisotopic determined by the guide
        estimated_monoisotopic_mz = mz_lower + MZ_FROM_EDGE
        selected_peak_idx = find_nearest_idx(peaks_a[:,0], estimated_monoisotopic_mz)
        selected_peak_mz = peaks_a[selected_peak_idx,0]
        selected_peak_intensity = peaks_a[selected_peak_idx,1]
        estimated_monoisotopic_mass = calculate_monoisotopic_mass(estimated_monoisotopic_mz, charge)
        expected_peak_spacing_mz = MASS_DIFFERENCE_C12_C13_MZ / charge

        isotope_intensities = np.empty(MAX_NUMBER_OF_SULPHUR_ATOMS, dtype=np.ndarray)
        for sulphurs in range(MAX_NUMBER_OF_SULPHUR_ATOMS):
            isotope_intensities[sulphurs] = calculate_peak_intensities(estimated_monoisotopic_mass, selected_peak_intensity, isotopes, sulphurs)

    # draw the chart
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    fig = plt.figure()
    ax = plt.axes()
    fig.set_figheight(4)
    fig.set_figwidth(8)
    if len(raw_points_df) > 0:
        markerline, stemlines, baseline = ax.stem(peaks_a[:,0], peaks_a[:,1], colors[1], use_line_collection=True, label='peak summed from raw data')
        plt.setp(markerline, linewidth=1, color=colors[1])
        plt.setp(markerline, markersize=3, color=colors[1])
        plt.setp(stemlines, linewidth=1, color=colors[1])
        plt.setp(baseline, linewidth=0.25, color=colors[7])
        plt.xlim([mz_lower,mz_upper])
        baseline.set_xdata([0,1])
        baseline.set_transform(plt.gca().get_yaxis_transform())
        # draw the monoisotopic shaded area
        for sulphurs in range(MAX_NUMBER_OF_SULPHUR_ATOMS):
            for isotope in range(isotopes):
                rect_base_mz = selected_peak_mz + (isotope * expected_peak_spacing_mz) - (MS1_PEAK_DELTA/2)
                peak_intensity = isotope_intensities[sulphurs][isotope]
                if sulphurs == 0:
                    rect = patches.Rectangle((rect_base_mz,0), MS1_PEAK_DELTA, peak_intensity, linewidth=0.2, facecolor='silver', alpha=0.6, label='theoretical, 0 sulphur')
                    ax.add_patch(rect)
                else:
                    dotted_line = plt.Line2D((rect_base_mz, rect_base_mz+MS1_PEAK_DELTA), (peak_intensity, peak_intensity), color=colors[sulphurs+3], linewidth=1.0, linestyle='-', alpha=1.0, label='{} sulphur(s)'.format(sulphurs))
                    plt.gca().add_line(dotted_line)
    plt.xlabel('m/z')
    plt.ylabel('intensity')
    plt.margins(0.06)
    # plt.legend(loc='best')
    plt.title('Peaks summed with intensity descent on raw data in the selected window')
    # save the chart as an image
    image_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    print("image file: {}".format(image_file_name))
    # plt.savefig(image_file_name, bbox_inches='tight')
    plt.savefig(image_file_name)
    plt.close()

    return image_file_name

def tile_coords_to_data_coords(tile_name, tile_width, tile_height, region_x, region_y, region_width, region_height, canvas_scale):
    elements = tile_name.split('-')
    frame_id = int(elements[1])
    tile_mz_lower = int(elements[5])
    tile_mz_upper = int(elements[6])

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

@app.route('/webhook', methods=['POST'])
def webhook():
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

if __name__ == '__main__':
    app.run()
