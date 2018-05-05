from __future__ import print_function
import sys
import argparse
import time
import pymysql


parser = argparse.ArgumentParser(description='Prepare the database for MS1 frame summing in the region of the MS1 feature\'s drift and retention time.')
parser.add_argument('-db','--database_name', type=str, help='The name of the database.', required=True)
args = parser.parse_args()

# Connect to the database
dest_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database="{}".format(args.database_name))
dest_c = dest_conn.cursor()

print("Setting up tables and indexes")
dest_c.execute("CREATE OR REPLACE TABLE peak_correlation (feature_id INTEGER, base_peak_id INTEGER, ms2_peak_id INTEGER, correlation REAL, PRIMARY KEY (feature_id, base_peak_id, ms2_peak_id))")
dest_c.execute("CREATE OR REPLACE TABLE peak_correlation_info (item TEXT, value TEXT)")

dest_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 on summed_ms1_regions (feature_id, peak_id)")
dest_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_2 on feature_base_peaks (feature_id)")

dest_conn.commit()
dest_conn.close()
