import sys
import pymysql
import argparse

parser = argparse.ArgumentParser(description='Prepare the database for MS1 frame summing.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-hn','--hostname', default='mscypher-004', type=str, help='The hostname of the database.', required=False)
args = parser.parse_args()

source_conn = pymysql.connect(host="{}".format(args.hostname), user='root', passwd='password', database="{}".format(args.database_name))
src_c = source_conn.cursor()

print("Setting up tables and indexes")

src_c.execute("CREATE OR REPLACE TABLE summed_frames (frame_id INTEGER, point_id INTEGER, mz REAL, scan INTEGER, intensity INTEGER, peak_id INTEGER)")
src_c.execute("CREATE OR REPLACE TABLE summing_info (item TEXT, value TEXT)")
src_c.execute("CREATE OR REPLACE TABLE elution_profile (frame_id INTEGER, intensity INTEGER)")

source_conn.commit()
source_conn.close()
