import sys
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='Prepare the database for MS1 cluster detection.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

source_conn = sqlite3.connect(args.database_name)
src_c = source_conn.cursor()

print("Setting up tables...")

src_c.execute("DROP TABLE IF EXISTS clusters")
src_c.execute("DROP TABLE IF EXISTS cluster_detect_info")

src_c.execute("""CREATE TABLE clusters (frame_id INTEGER, 
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
src_c.execute("CREATE TABLE cluster_detect_info (item TEXT, value TEXT)")

print("Setting up indexes...")

src_c.execute("CREATE INDEX IF NOT EXISTS idx_peaks ON peaks (frame_id)")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peaks_2 ON peaks (frame_id,peak_id)")

source_conn.commit()
source_conn.close()
