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

    # perform intensity descent to consolidate the peaks
    raw_points_a = raw_points_df[['mz','intensity']].to_numpy()
    peaks_a = ms1_intensity_descent(raw_points_a)

    # monoisotopic determined by the guide
    estimated_monoisotopic_mz = mz_lower + MZ_FROM_EDGE
    selected_peak_idx = find_nearest_idx(peaks_a[:,0], estimated_monoisotopic_mz)
    selected_peak_mz = peaks_a[selected_peak_idx,0]
    selected_peak_intensity = peaks_a[selected_peak_idx,1]

    expected_peak_spacing_mz = MASS_DIFFERENCE_C12_C13_MZ / charge

    # draw the chart
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    fig = plt.figure()
    ax = plt.axes()
    fig.set_figheight(4)
    fig.set_figwidth(8)
    if len(raw_points_df) > 0:
        markerline, stemlines, baseline = ax.stem(peaks_a[:,0], peaks_a[:,1], 'g', use_line_collection=True)
        plt.setp(markerline, linewidth=1, color=colors[2])
        plt.setp(markerline, markersize=3, color=colors[2])
        plt.setp(stemlines, linewidth=1, color=colors[2])
        plt.setp(baseline, linewidth=0.25, color=colors[7])
        plt.xlim([mz_lower,mz_upper])
        baseline.set_xdata([0,1])
        baseline.set_transform(plt.gca().get_yaxis_transform())
        # draw the monoisotopic shaded area
        for isotope in range(isotopes):
            rect_base_mz = selected_peak_mz + (isotope * expected_peak_spacing_mz) - (MS1_PEAK_DELTA/2)
            rect = patches.Rectangle((rect_base_mz,0), MS1_PEAK_DELTA, selected_peak_intensity, linewidth=0, facecolor='silver', alpha=0.8)
            ax.add_patch(rect)
    plt.xlabel('m/z')
    plt.ylabel('intensity')
    plt.margins(0.06)
    plt.title('Peaks summed with intensity descent on raw data in the selected window')
    # save the chart as an image
    image_file_name = tempfile.NamedTemporaryFile(suffix='.png').name
    print("image file: {}".format(image_file_name))
    plt.savefig(image_file_name, bbox_inches='tight')
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
        charge = int(''.join(ch for ch in attributes['charge'] if ch.isdigit()))
        isotopes = int(attributes['isotopes'])
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
