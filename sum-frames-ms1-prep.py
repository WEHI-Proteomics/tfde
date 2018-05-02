import sys
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Prepare the database for MS1 frame summing.')
parser.add_argument('-sdb','--source_database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

source_conn = sqlite3.connect(args.source_database_name)
src_c = source_conn.cursor()

print("Setting up indexes...")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_frame_properties ON frame_properties (collision_energy, frame_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_frames ON frames (frame_id, mz, scan)")

source_conn.commit()
source_conn.close()
