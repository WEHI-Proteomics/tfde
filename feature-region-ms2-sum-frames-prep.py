from __future__ import print_function
import sys
import argparse
import time
import pymysql


parser = argparse.ArgumentParser(description='Prepare the database for MS2 frame summing in the region of the MS1 feature\'s drift and retention time.')
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
args = parser.parse_args()

# Connect to the database
dest_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database="{}".format(args.database_name))
dest_c = dest_conn.cursor()

# Set up the tables if they don't exist already
print("Setting up tables")
dest_c.execute("CREATE OR REPLACE TABLE summed_ms2_regions (feature_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, number_frames INTEGER, peak_id INTEGER)")  # number_frames = number of source frames the point was found in
dest_c.execute("CREATE OR REPLACE TABLE summed_ms2_regions_info (item TEXT, value TEXT)")

print("Setting up indexes")
dest_c.execute("CREATE OR REPLACE INDEX idx_features ON features (feature_id,charge_state)")
dest_c.execute("CREATE OR REPLACE INDEX idx_frames ON frames (frame_id,scan)")

dest_conn.commit()
dest_conn.close()
