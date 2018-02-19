import sys
import pymysql
import argparse

parser = argparse.ArgumentParser(description='Prepare the database for MS1 cluster detection.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

source_conn = pymysql.connect(host='mscypher-004', user='root', passwd='password', database="{}".format(args.database_name))
src_c = source_conn.cursor()

print("Setting up tables and indexes")
src_c.execute("""CREATE OR REPLACE TABLE clusters (frame_id INTEGER, 
                                                cluster_id INTEGER, 
                                                charge_state INTEGER, 
                                                base_isotope_peak_id INTEGER, 
                                                base_peak_mz_centroid REAL, 
                                                base_peak_mz_std_dev REAL, 
                                                base_peak_scan_centroid REAL, 
                                                base_peak_scan_std_dev REAL, 
                                                base_peak_max_point_mz REAL, 
                                                base_peak_max_point_scan INTEGER, 
                                                monoisotopic_peak_id INTEGER, 
                                                sulphides INTEGER, 
                                                fit_error REAL, 
                                                rationale TEXT, 
                                                intensity_sum INTEGER, 
                                                feature_id INTEGER, 
                                                scan_lower INTEGER, 
                                                scan_upper INTEGER, 
                                                mz_lower REAL, 
                                                mz_upper REAL, 
                                                PRIMARY KEY(cluster_id,frame_id))""")
src_c.execute("CREATE OR REPLACE TABLE cluster_detect_info (item TEXT, value TEXT)")

# Indexes
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peaks ON peaks (frame_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peaks_2 ON peaks (frame_id,peak_id)")

print("Resetting cluster IDs")
src_c.execute("update peaks set cluster_id=0 where cluster_id!=0")

source_conn.commit()
source_conn.close()
