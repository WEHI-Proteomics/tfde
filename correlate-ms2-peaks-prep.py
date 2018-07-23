import argparse
import sqlite3


parser = argparse.ArgumentParser(description='Prepare for the ms1 / ms2 peak correlation step.')
parser.add_argument('-cdb','--converted_database_name', type=str, help='The name of the converted database.', required=True)
args = parser.parse_args()

conv_db_conn = sqlite3.connect(args.converted_database_name)
conv_c = conv_db_conn.cursor()

print("Setting up indexes")
conv_c.execute("CREATE INDEX IF NOT EXISTS idx_raw_summed_join_1 ON raw_summed_join (summed_frame_id, summed_point_id)")
conv_c.execute("CREATE INDEX IF NOT EXISTS idx_frames_3 ON frames (frame_id, point_id)")

conv_db_conn.close()
