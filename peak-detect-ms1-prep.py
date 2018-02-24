import sys
import pymysql
import argparse

parser = argparse.ArgumentParser(description='Prepare the database for MS1 peak detection.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-hn','--hostname', default='mscypher-004', type=str, help='The hostname of the database.', required=False)
args = parser.parse_args()

source_conn = pymysql.connect(host="{}".format(args.hostname), user='root', passwd='password', database="{}".format(args.database_name))
src_c = source_conn.cursor()

print("Setting up tables and indexes")

src_c.execute("CREATE OR REPLACE TABLE peaks (frame_id INTEGER, peak_id INTEGER, centroid_mz REAL, centroid_scan REAL, intensity_sum INTEGER, scan_upper INTEGER, scan_lower INTEGER, std_dev_mz REAL, std_dev_scan REAL, rationale TEXT, intensity_max INTEGER, peak_max_mz REAL, peak_max_scan INTEGER, cluster_id INTEGER, PRIMARY KEY (frame_id, peak_id))")
src_c.execute("CREATE OR REPLACE TABLE peak_detect_info (item TEXT, value TEXT)")

# Indexes
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_frames ON summed_frames (frame_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_summed_frames_2 ON summed_frames (frame_id,point_id)")

print("Resetting peak IDs")
src_c.execute("update summed_frames set peak_id=0 where peak_id!=0")

source_conn.commit()
source_conn.close()
