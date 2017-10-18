import sys
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='A tree descent method for peak detection.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)

args = parser.parse_args()

source_conn = sqlite3.connect(args.database_name)

c = source_conn.cursor()
c.execute("PRAGMA temp_store = 2") # store temporary data in memory

# Set up the table for detected peaks
print("Setting up tables and indexes")
c.execute('''DROP TABLE IF EXISTS peaks''')
c.execute('''CREATE TABLE peaks (frame_id INTEGER, peak_id INTEGER, centroid_mz REAL, centroid_scan REAL, intensity_sum INTEGER, scan_upper INTEGER, scan_lower INTEGER, std_dev_mz REAL, std_dev_scan REAL, cluster_id INTEGER, 'rationale' TEXT, 'state' TEXT, intensity_max INTEGER, PRIMARY KEY (frame_id, peak_id))''')

# Indexes
c.execute('''DROP INDEX IF EXISTS idx_frame_peak''')
c.execute('''CREATE INDEX idx_frame_peak ON peaks (frame_id,peak_id)''')

c.execute('''DROP INDEX IF EXISTS idx_frame''')
c.execute('''CREATE INDEX idx_frame ON peaks (frame_id)''')

c.execute('''DROP INDEX IF EXISTS idx_frame_point''')
c.execute('''CREATE INDEX idx_frame_point ON frames (frame_id,point_id)''')

c.execute("update frames set peak_id=0")

c.execute('''DROP TABLE IF EXISTS peak_detect_info''')
c.execute('''CREATE TABLE peak_detect_info (item TEXT, value TEXT)''')

source_conn.commit()
source_conn.close()
