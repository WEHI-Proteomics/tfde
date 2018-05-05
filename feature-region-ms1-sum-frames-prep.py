from __future__ import print_function
import sys
import argparse
import time
import sqlite3

parser = argparse.ArgumentParser(description='Prepare the database for MS1 frame summing in the region of the MS1 feature\'s drift and retention time.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

# Connect to the database
src_conn = sqlite3.connect(args.source_database_name)
src_c = src_conn.cursor()

# Set up the tables if they don't exist already
print("Setting up tables and indexes")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_1 on features (feature_id, charge_state, mz_lower, mz_upper)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_2 on summed_frames (frame_id, peak_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_3 on clusters (feature_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_ms1_regions_4 on peaks (frame_id, cluster_id)")

src_conn.commit()
src_conn.close()
