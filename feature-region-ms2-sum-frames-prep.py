from __future__ import print_function
import sys
import argparse
import time
import sqlite3


parser = argparse.ArgumentParser(description='Prepare the database for MS2 frame summing in the region of the MS1 feature\'s drift and retention time.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-ddb','--destination_database_name', type=str, help='The name of the destination database.', required=True)
args = parser.parse_args()

# Connect to the databases
src_conn = sqlite3.connect(args.source_database_name)
src_c = src_conn.cursor()

dest_conn = sqlite3.connect(args.destination_database_name)
dest_c = dest_conn.cursor()

# Set up the tables if they don't exist already
print("Setting up tables")
dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions")
dest_c.execute("DROP TABLE IF EXISTS summed_ms2_regions_info")
dest_c.execute("CREATE TABLE summed_ms2_regions (feature_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, number_frames INTEGER, peak_id INTEGER)")  # number_frames = number of source frames the point was found in
dest_c.execute("CREATE TABLE summed_ms2_regions_info (item TEXT, value TEXT)")

print("Setting up indexes")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_frame_properties_2 ON frame_properties (collision_energy, frame_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_features_1 ON features (feature_id,charge_state,mz_lower,mz_upper)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_frames_2 ON frames (frame_id,scan)")

dest_conn.commit()
dest_conn.close()

src_conn.close()
