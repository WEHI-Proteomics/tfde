import sys
# import matplotlib
# matplotlib.use("Agg")
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import ix_
import timsdata
from scipy import ndimage
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
MAX_SCANS = 400
MAX_INTENSITY = 25000
THRESHOLD = 100
IMAGE_X_PIXELS = 2000
IMAGE_Y_PIXELS = 2000
IMAGE_DPI = 100
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
        dy, dx = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

def remove_overlaps(bboxes):
    '''
    Return a set of BBoxes which contain the given BBoxes.
    When two BBoxes overlap, replace both with the minimal BBox that contains both.
    '''
    # list upper left and lower right corners of the Bboxes
    corners = []

    # list upper left corners of the Bboxes
    ulcorners = []

    # dict mapping corners to Bboxes.
    bbox_map = {}

    for bbox in bboxes:
        ul = (bbox.x1, bbox.y1)
        lr = (bbox.x2, bbox.y2)
        bbox_map[ul] = bbox
        bbox_map[lr] = bbox
        ulcorners.append(ul)
        corners.append(ul)
        corners.append(lr)        

    # Use a KDTree so we can find corners that are nearby efficiently.
    tree = spatial.KDTree(corners)
    new_corners = []
    for corner in ulcorners:
        bbox = bbox_map[corner]
        # Find all points which are within a taxicab distance of corner
        indices = tree.query_ball_point(
            corner, bbox_map[corner].taxicab_diagonal(), p = 1)
        for near_corner in tree.data[indices]:
            near_bbox = bbox_map[tuple(near_corner)]
            if bbox != near_bbox and bbox.overlaps(near_bbox):
                # Expand both bboxes.
                # Since we mutate the bbox, all references to this bbox in
                # bbox_map are updated simultaneously.
                bbox.x1 = near_bbox.x1 = min(bbox.x1, near_bbox.x1)
                bbox.y1 = near_bbox.y1 = min(bbox.y1, near_bbox.y1) 
                bbox.x2 = near_bbox.x2 = max(bbox.x2, near_bbox.x2)
                bbox.y2 = near_bbox.y2 = max(bbox.y2, near_bbox.y2) 
    return set(bbox_map.values())

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
for frame_id in range(20, 21):
    print("Frame {:0>4} of {}".format(frame_id, frame_count))

    fig = plt.figure(figsize=(IMAGE_X_PIXELS/float(IMAGE_DPI), IMAGE_Y_PIXELS/float(IMAGE_DPI)), dpi=IMAGE_DPI)
    axes = fig.add_subplot(111)
    plt.tight_layout()
    # axes = fig.add_subplot(111, projection='3d')
    # axes.view_init(elev=60, azim=100)

    # axes.set_xlabel('Index')
    # axes.set_xlim3d([0.0, MAX_INDEX])

    # axes.set_ylabel('Scan Number')
    # axes.set_ylim3d([0.0, MAX_SCANS])

    # axes.set_zlabel('Intensity')
    # axes.set_zlim3d([0.0, MAX_INTENSITY])

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
                # all bars for this scan are in the same plane (zs=i). See http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#bar-plots
                # axes.bar(left=mz, height=intensities, zs=i, zdir='y', width=2.0, linewidth=0, color='green')
                # fig.suptitle('Frame {:0>4}'.format(frame_id), fontsize=20)
                frame[ix_(mz.astype(int)), i] = intensities
    # Save the figure as a PNG image
    # fig.savefig('/tmp/frame-{:0>4}.png'.format(frame_id), bbox_inches = 'tight', pad_inches = 0, dpi=100)
    # plt.show()
    # plt.axis('off')
    plt.imshow(gaussian_filter(frame, sigma=(50,10), order=0), interpolation='nearest', extent=(0, MAX_INDEX, MAX_SCANS, 0), cmap='hot')
    plt.show()

    # plt.close(fig)
    # Find the bounding boxes of the peaks
    mask = frame > PEAK_THRESHOLD
    structure = [[1,1,1],   # look for adjacent and diagonal connections
                 [1,1,1],
                 [1,1,1]]
    label_frame, nb_labels = ndimage.label(mask, structure)
    slices = ndimage.find_objects(label_frame)
    bounding_boxes = slice_to_bbox(slices)
    non_overlap_bounding_boxes = remove_overlaps(bounding_boxes)
    print("detected {} peaks".format(len(non_overlap_bounding_boxes)))
    for box in non_overlap_bounding_boxes:
        print("{}".format(box))
