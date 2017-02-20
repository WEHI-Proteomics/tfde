import sys
# import matplotlib
# matplotlib.use("Agg")
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy import ix_
import timsdata
from scipy import ndimage
from scipy.misc import imsave
import scipy.spatial as spatial
from scipy.ndimage import gaussian_filter

class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        '''
        (x1, y1) is the upper left corner,
        (x2, y2) is the lower right corner,
        with (0, 0) being in the upper left corner.
        '''
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def taxicab_diagonal(self):
        '''
        Return the taxicab distance from (x1,y1) to (x2,y2)
        '''
        return self.x2 - self.x1 + self.y2 - self.y1

    def overlaps(self, other):
        '''
        Return True iff self and other overlap.
        '''
        return not ((self.x1 > other.x2)
                    or (self.x2 < other.x1)
                    or (self.y1 > other.y2)
                    or (self.y2 < other.y1))

    def __eq__(self, other):
        return (self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2)

    def __format__(self, format):
        return '({},{}),({},{})'.format(self.x1, self.y1, self.x2, self.y2)

MAX_INDEX = 3200
MAX_SCANS = 814
MAX_INTENSITY = 25000
THRESHOLD = 100
IMAGE_X_PIXELS = 1600
IMAGE_Y_PIXELS = 900
PEAK_THRESHOLD = 500

def threshold_scan_transform(threshold, indicies, intensities):
    np_mz = timsfile.indexToMz(frame_id, np.array(indicies, dtype=np.float64))
    np_int = np.asarray(intensities, dtype=np.int32)
    low_indices = np.where(np_int < threshold)
    np_mz = np.delete(np_mz, low_indices)
    np_int = np.delete(np_int, low_indices)
    return np_mz, np_int

def slice_to_bbox(slices):
    for s in slices:
        # dy, dx = s[:2]
        dx, dy = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

if len(sys.argv) < 2:
    raise RuntimeError("need arguments: tdf_directory")

analysis_dir = sys.argv[1]
if sys.version_info.major == 2:
    analysis_dir = unicode(analysis_dir)

timsfile = timsdata.TimsData(analysis_dir)

# Get total frame count
q = timsfile.conn.execute("SELECT COUNT(*) FROM Frames")
row = q.fetchone()
frame_count = row[0]
print("Analysis has {0} frames.".format(frame_count))

# for frame_id in range(1, frame_count+1):
for frame_id in range(30000, 30001):
    print("Frame {:0>4} of {}".format(frame_id, frame_count))

    fig = plt.figure()
    dpi = fig.get_dpi()
    fig.set_size_inches(float(IMAGE_X_PIXELS)/float(dpi), float(IMAGE_Y_PIXELS)/float(dpi))
    axes = fig.add_subplot(111)

    (frame_data_id, num_scans) = timsfile.conn.execute(
          "SELECT TimsId, NumScans FROM Frames WHERE Id = {}".format(frame_id)).fetchone()
    scans = timsfile.readScans(frame_data_id, 0, num_scans)
    scan_begin = 0
    scan_end = num_scans

    frame = np.zeros((MAX_INDEX, MAX_SCANS))

    for i in range(scan_begin, scan_end):
        scan = scans[i]
        if len(scan[0]) > 0:
            mz, intensities = threshold_scan_transform(THRESHOLD, scan[0], scan[1])
            peaks = len(mz)
            if peaks > 0:
                frame[ix_(mz.astype(int)), i] = intensities

    # frame = gaussian_filter(frame, sigma=(20,10), order=0)
    axes.imshow(frame, interpolation='nearest', cmap='hot', aspect='auto')

    # Find the bounding boxes of the peaks
    # mask = frame > (frame.max()/10.0)
    # im, number_of_objects = ndimage.label(mask)
    # blobs = ndimage.find_objects(im)

    # Add the bounding boxes to the image plot
    # for blob in blobs:
    #     x1 = int(blob[1].start)
    #     width = int(blob[1].stop - blob[1].start)
    #     y1 = int(blob[0].start)
    #     height = int(blob[0].stop - blob[0].start)
    #     # print("x1={}, y1={}, w={}, h={}".format(x1, y1, width, height))
    #     p = patches.Rectangle((x1, y1), width, height, fc = 'none', ec = 'green', linewidth=2)
    #     axes.add_patch(p)

    # Save the figure as a PNG image
    plt.xlabel('Scan')
    plt.ylabel('Index')
    fig.suptitle('Frame {:0>4}'.format(frame_id), fontsize=20)
    fig.savefig('./frame-{:0>4}.png'.format(frame_id), pad_inches = 0.0, dpi='figure')
    plt.close(fig)
