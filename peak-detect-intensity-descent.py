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


parser = argparse.ArgumentParser(description='A tree descent method for peak detection.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
subparsers = parser.add_subparsers(dest='cmd', help='help for subcommand')

# create the parser for the "query" command
parser_a = subparsers.add_parser('query', help='Display the attributes of the frames in the database.')

# create the parser for the "process" command
parser_b = subparsers.add_parser('process', help='Process the database.')
parser_b.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
parser_b.add_argument('-fl','--frame_lower', type=int, help='The lower frame number.', required=True)
parser_b.add_argument('-fu','--frame_upper', type=int, help='The upper frame number.', required=True)
parser_b.add_argument('-sl','--scan_lower', type=int, help='The lower scan number.', required=True)
parser_b.add_argument('-su','--scan_upper', type=int, help='The upper scan number.', required=True)
parser_b.add_argument('-es','--empty_scans', type=int, default=2, help='Maximum number of empty scans to tolerate.', required=False)
parser_b.add_argument('-sd','--standard_deviations', type=int, default=4, help='Number of standard deviations to look either side of a point.', required=False)

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
    dest_conn = sqlite3.connect(args.destination_database_name)
    dest_c = dest_conn.cursor()

    dest_c.execute('''DROP TABLE IF EXISTS frames''')
    dest_c.execute('''CREATE TABLE frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)''')
    dest_c.execute('''DROP INDEX IF EXISTS idx_frames''')
    dest_c.execute('''CREATE INDEX idx_frames ON frames (frame_id)''')

    summedFrameId = 1
    peak_id = 1
    for frame in range(args.frame_lower, args.frame_upper+1):
        frame_df = pd.read_sql_query("select mz,scan,intensity,point_id,peak_id,frame_id from frames where frame_id={} order by mz, scan asc;"
            .format(frame), source_conn)
        start_frame = time.time()
        while frame_df.count()[0] > 0:
            peak_df = pd.DataFrame()
            max_p_df = frame_df[frame_df.intensity == frame_df.intensity.max()]
            peak_df = pd.concat([peak_df, max_p_df])
            mz = max_p_df.mz.values[0]
            scan = max_p_df.scan.values[0]
            std_dev_window = standard_deviation(mz) * args.standard_deviations
            # Look in the 'up' direction
            scan_offset = 1
            missed_scans = 0
            while missed_scans <= args.empty_scans:
                nearby_df = frame_df[(frame_df.scan == scan-scan_offset) & (frame_df.mz >= mz - std_dev_window) & 
                (frame_df.mz <= mz + std_dev_window)]
                if nearby_df.count()[0] == 0:
                    missed_scans += 1
                else:
                    peak_df = pd.concat([peak_df, nearby_df])
                    missed_scans = 0
                scan_offset += 1
            # Look in the 'down' direction
            scan_offset = 1
            missed_scans = 0
            while missed_scans <= args.empty_scans:
                nearby_df = frame_df[(frame_df.scan == scan+scan_offset) & (frame_df.mz >= mz - std_dev_window) & 
                (frame_df.mz <= mz + std_dev_window)]
                if nearby_df.count()[0] == 0:
                    missed_scans += 1
                else:
                    peak_df = pd.concat([peak_df, nearby_df])
                    missed_scans = 0
                scan_offset += 1
            
            # Remove the points in this peak
            frame_df = frame_df[-frame_df.index.isin(peak_df.index)]
            peak_df.peak_id = peak_id
            # Update database
            peak_df.to_sql(con=dest_conn, name='frames', index=False, if_exists='append')
            peak_id += 1
        stop_frame = time.time()
        print("{} seconds to process frame - {} peaks".format(stop_frame-start_frame, peak_id))
