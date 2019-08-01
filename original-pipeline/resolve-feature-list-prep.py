import sys
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Prepare the database for MS1 feature resolution.')
parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
args = parser.parse_args()

db_conn = sqlite3.connect(args.converted_database_name)

print("Setting up indexes...")
db_conn.cursor().execute("DROP INDEX IF EXISTS idx_raw_summed_join_1")
db_conn.cursor().execute("DROP INDEX IF EXISTS idx_frames_4")
db_conn.cursor().execute("CREATE INDEX idx_raw_summed_join_1 ON raw_summed_join (summed_frame_point)")
db_conn.cursor().execute("CREATE INDEX idx_raw_summed_join_2 ON raw_summed_join (summed_frame_id, summed_point_id)")
db_conn.cursor().execute("CREATE INDEX idx_frames_4 ON frames (raw_frame_point)")
db_conn.cursor().execute("CREATE INDEX idx_frames_5 ON frames (frame_id, point_id)")

db_conn.commit()
db_conn.close()
