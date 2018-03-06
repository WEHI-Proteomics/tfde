import sys
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Prepare the database for MS1 peak detection.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

source_conn = sqlite3.connect(args.database_name)
src_c = source_conn.cursor()

print("Setting up tables...")

src_c.execute("DROP TABLE IF EXISTS peaks")
src_c.execute("DROP TABLE IF EXISTS peak_detect_info")

src_c.execute("CREATE TABLE peaks (frame_id INTEGER, peak_id INTEGER, centroid_mz REAL, centroid_scan REAL, intensity_sum INTEGER, scan_upper INTEGER, scan_lower INTEGER, std_dev_mz REAL, std_dev_scan REAL, rationale TEXT, intensity_max INTEGER, peak_max_mz REAL, peak_max_scan INTEGER, cluster_id INTEGER, PRIMARY KEY (frame_id, peak_id))")
src_c.execute("CREATE TABLE peak_detect_info (item TEXT, value TEXT)")

print("Setting up indexes...")

src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_frames ON summed_frames (frame_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_frames_2 ON summed_frames (frame_id,point_id)")

source_conn.commit()
source_conn.close()
