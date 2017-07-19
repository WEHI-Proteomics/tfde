import sys
import sqlite3
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import time

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return (mz / instrument_resolution) / 2.35482


parser = argparse.ArgumentParser(description='An intensity descent method for summing frames.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
subparsers = parser.add_subparsers(dest='cmd', help='help for subcommand')

# create the parser for the "command_1" command
parser_a = subparsers.add_parser('query', help='Display the attributes of the frames in the database.')

# create the parser for the "command_2" command
parser_b = subparsers.add_parser('process', help='Process the database.')
parser_b.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
parser_b.add_argument('-fl','--frame_lower', type=int, help='The lower frame number.', required=True)
parser_b.add_argument('-fu','--frame_upper', type=int, help='The upper frame number.', required=True)
parser_b.add_argument('-sl','--scan_lower', type=int, help='The lower scan number.', required=True)
parser_b.add_argument('-su','--scan_upper', type=int, help='The upper scan number.', required=True)
parser_b.add_argument('-fts','--frames_to_sum', type=int, default=5, help='Number of frames to sum.', required=False)
parser_b.add_argument('-nt','--noise_threshold', type=int, default=2, help='Minimum number of frames a point must appear in to be processed.', required=False)
parser_b.add_argument('-bf','--base_frame', type=int, default=1, help='Base frame number.', required=False)

args = parser.parse_args()
print args

source_conn = sqlite3.connect(args.source_database_name)

if args.cmd == 'query':
    frame_df = pd.read_sql_query("select frame_id,scan,mz,intensity from frames;", source_conn)
    scan_min = frame_df.scan.min()
    scan_max = frame_df.scan.max()
    mz_min = frame_df.mz.min()
    mz_max = frame_df.mz.max()
    frame_min = frame_df.frame_id.min()
    frame_max = frame_df.frame_id.max()
    intensity_min = frame_df.intensity.min()
    intensity_max = frame_df.intensity.max()
    print("frames {}-{}, scans {}-{}, m/z {}-{}, intensity {}-{}".format(frame_min, frame_max, scan_min, scan_max, mz_min, mz_max, intensity_min, intensity_max))
elif args.cmd == 'process':
    frame_lower = args.frame_lower
    frame_upper = args.frame_upper
    scan_lower = args.scan_lower
    scan_upper = args.scan_upper
    frames_to_sum = args.frames_to_sum

    dest_conn = sqlite3.connect(args.destination_database_name)
    dest_c = dest_conn.cursor()

    dest_c.execute('''CREATE TABLE IF NOT EXISTS frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)''')
    dest_c.execute('''CREATE INDEX IF NOT EXISTS idx_frames ON frames (frame_id)''')

    start_run = time.time()
    summedFrameId = args.base_frame
    for base_frame in range(frame_lower, frame_upper+1, frames_to_sum):
        print("Processing frames {}-{} to create frame {}".format(base_frame, base_frame+frames_to_sum-1, summedFrameId))
        frame_df = pd.read_sql_query("select frame_id,mz,scan,intensity from frames where frame_id>={} AND frame_id<={} order by frame_id, mz, scan asc;".format(base_frame, base_frame+frames_to_sum-1), source_conn)
        frame_v = frame_df.values
        frame_start = time.time()
        pointId = 1
        points = []
        for scan in range(scan_lower, scan_upper+1):
            # mz = []
            # intensity = []
            scan_start_time = time.time()
            points_v = frame_v[np.where(frame_v[:,2] == scan)]
            points_to_process = len(points_v)
            low_intensity = 10000
            high_intensity = 0
            while len(points_v) > 0:
                max_intensity_index = np.argmax(points_v[:,3])
                point_mz = points_v[max_intensity_index, 1]
                # print("m/z {}, intensity {}".format(point_mz, points_v[max_intensity_index, 3]))
                delta_mz = standard_deviation(point_mz) * 4.0
                # Find all the points in this point's std dev window
                nearby_point_indices = np.where((points_v[:,1] >= point_mz-delta_mz) & (points_v[:,1] <= point_mz+delta_mz))[0]
                nearby_points = points_v[nearby_point_indices]
                # How many distinct frames do the points come from?
                unique_frames = np.unique(nearby_points[:,0])
                if len(unique_frames) >= args.noise_threshold:
                    # find the total intensity and centroid m/z
                    centroid_intensity = nearby_points[:,3].sum()
                    centroid_mz = peakutils.centroid(nearby_points[:,1], nearby_points[:,3])
                    points.append((summedFrameId, pointId, centroid_mz, scan, int(round(centroid_intensity)), 0))
                    pointId += 1

                    # mz.append(centroid_mz)
                    # intensity.append(centroid_intensity)

                    # Plot a scan
                    # f = plt.figure()
                    # ax1 = f.add_subplot(111)
                    # plt.xlim(point_mz-delta_mz, point_mz+delta_mz)
                    # ax1.plot(points_v[:,1], points_v[:,3], 'o', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.0, markersize=4)
                    # ax1.plot(points_v[max_intensity_index,1], points_v[max_intensity_index,3], 'x', markerfacecolor='red', markeredgecolor='red', markeredgewidth=1.0, markersize=10)
                    # ax1.plot(points_v[nearby_point_indices,1], points_v[nearby_point_indices,3], '+', markerfacecolor='red', markeredgecolor='green', markeredgewidth=1.0, markersize=10)
                    # ax1.plot(mz, intensity, 'o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.0, markersize=6)
                    # plt.suptitle("Points along a scan")
                    # plt.title("database {}, base frame {}, scan {}, points={}".format(args.source_database_name, base_frame, scan, len(points_v)))
                    # plt.xlabel('m/z')
                    # plt.ylabel('intensity')
                    # plt.margins(0.02)
                    # plt.show()

                    # remove the points we've processed
                    points_v = np.delete(points_v, nearby_point_indices, 0)
                else:
                    # remove this point because it doesn't have enough neighbours
                    points_v = np.delete(points_v, max_intensity_index, 0)
            scan_stop_time = time.time()

        dest_c.executemany("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)", points)
        dest_conn.commit()
        frame_end = time.time()
        print("{} sec for frame {} (frames {}-{})".format(frame_end-frame_start, summedFrameId, base_frame, base_frame+frames_to_sum-1))
        summedFrameId += 1
    stop_run = time.time()
    print("{} seconds to process run".format(stop_run-start_run))
