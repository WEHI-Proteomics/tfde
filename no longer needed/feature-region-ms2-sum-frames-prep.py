from __future__ import print_function
import sys
import argparse
import time
import sqlite3


parser = argparse.ArgumentParser(description='Prepare the database for MS2 frame summing in the region of the MS1 feature\'s drift and retention time.')
parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

# Connect to the databases
conv_conn = sqlite3.connect(args.converted_database_name)
conv_c = conv_conn.cursor()

src_conn = sqlite3.connect(args.source_database_name)
src_c = src_conn.cursor()

print("Setting up indexes")
conv_c.execute("CREATE INDEX IF NOT EXISTS idx_frame_properties_2 ON frame_properties (collision_energy, frame_id)")
conv_c.execute("CREATE INDEX IF NOT EXISTS idx_frames_2 ON frames (frame_id,scan)")

src_c.execute("CREATE INDEX IF NOT EXISTS idx_features_1 ON features (feature_id,charge_state,mz_lower,mz_upper)")

conv_conn.close()
src_conn.close()
