from flask import Flask, request, abort, jsonify, send_file
from flask_cors import CORS
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CONVERTED_DATABASE = '/Users/darylwilding-mcbride/Downloads/190719_Hela_Ecoli/converted/190719_Hela_Ecoli_1to1_01-converted.sqlite'
IMAGE_X = 400
IMAGE_Y = 300

app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

# credit: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )

def image_from_raw_data(data_coords):
    frame_id = data_coords['frame_id']
    mz_lower = data_coords['mz_lower']
    mz_upper = data_coords['mz_upper']
    scan_lower = data_coords['scan_lower']
    scan_upper = data_coords['scan_upper']

    # get the raw points
    db_conn = sqlite3.connect(CONVERTED_DATABASE)
    raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {} and mz >= {} and mz <= {} and scan >= {} and scan <= {}".format(frame_id, mz_lower, mz_upper, scan_lower, scan_upper), db_conn)
    db_conn.close()

    # draw the chart
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    f, ax = plt.subplots()
    markerline, stemlines, baseline = ax.stem(raw_points_df.mz, raw_points_df.intensity, 'g')
    plt.setp(markerline, 'color', colors[2])
    plt.setp(stemlines, 'color', colors[2])
    plt.setp(baseline, 'color', colors[7])
    plt.xlabel('m/z')
    plt.ylabel('intensity')
    f.set_figheight(5)
    f.set_figwidth(15)
    plt.margins(0.06)
    plt.title('Raw data in the selected window')

    # put the chart into an image
    im = fig2img(f)
    return im

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

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
        print(request.json)
        # convert to data coordinates
        data_coords = tile_coords_to_data_coords(tile_name, tile_width, tile_height, x, y, width, height, canvas_scale)
        print("data coords: {}".format(data_coords))
        # create image
        img = image_from_raw_data(data_coords)
        img.save('/Users/darylwilding-mcbride/Downloads/image.png')
        # return the image in the response
        # return serve_pil_image(img)
    else:
        abort(400)


if __name__ == '__main__':
    app.run()
