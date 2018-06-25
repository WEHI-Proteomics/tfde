import pandas as pd
import argparse
import numpy as np
import sqlite3
import time

# Number of points either side of the base peak's maximum intensity to check for correlation
BASE_PEAK_CORRELATION_SIDE_POINTS = 3

#
# nohup python -u ./otf-peak-detect/correlate-ms2-peaks.py -db /media/data-drive/Hela_20A_20R_500-features-1-100000-random-1000-sf-1000.sqlite -fl 1 -fu 100000 > correlate-ms2-peaks.log 2>&1 &
#

parser = argparse.ArgumentParser(description='Calculate correlation between MS1 and MS2 peaks for features.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=True)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=True)
args = parser.parse_args()

source_conn = sqlite3.connect(args.database_name)
src_c = source_conn.cursor()
src_c.execute("PRAGMA temp_store = 2")
src_c.execute("PRAGMA journal_mode = TRUNCATE")

# Store the arguments as metadata in the database for later reference
peak_correlation_info = []
for arg in vars(args):
    peak_correlation_info.append((arg, getattr(args, arg)))

print("Setting up tables")
src_c.execute("DROP TABLE IF EXISTS peak_correlation")
src_c.execute("DROP TABLE IF EXISTS peak_correlation_info")
src_c.execute("CREATE TABLE peak_correlation (feature_id INTEGER, base_peak_id INTEGER, ms2_peak_id INTEGER, correlation REAL, PRIMARY KEY (feature_id, base_peak_id, ms2_peak_id))")
src_c.execute("CREATE TABLE peak_correlation_info (item TEXT, value TEXT)")

print("Setting up indexes")
print("creating idx_peak_correlation_1")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 on summed_ms1_regions (feature_id, peak_id)")
print("creating idx_peak_correlation_2")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_2 on feature_base_peaks (feature_id)")
print("creating idx_peak_correlation_3")
src_c.execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_3 on summed_ms2_regions (feature_id)")

start_run = time.time()

print("Loading the MS1 base peaks for the feature range")
features_df = pd.read_sql_query("select feature_id,base_peak_id from feature_base_peaks where feature_id >= {} and feature_id <= {} order by feature_id ASC;".format(args.feature_id_lower, args.feature_id_upper), source_conn)

peak_correlation = []

print("Finding peak correlations for features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
for feature_ids_idx in range(0,len(features_df)):
    feature_id = features_df.loc[feature_ids_idx].feature_id.astype(int)
    base_peak_id = features_df.loc[feature_ids_idx].base_peak_id.astype(int)
    print("correlating ms2 peaks for feature {} in range {}-{}".format(feature_id, args.feature_id_lower, args.feature_id_upper))

    # load the feature's base peak points
    base_peak_df = pd.read_sql_query("select scan,intensity from summed_ms1_regions where feature_id={} and peak_id={}".format(feature_id,base_peak_id), source_conn)

    # load the ms2 peaks for this feature
    ms2_peaks_df = pd.read_sql_query("select peak_id,intensity from ms2_peaks where feature_id={}".format(feature_id), source_conn)

    # load the ms2 peak points for this feature
    ms2_peak_points_df = pd.read_sql_query("select peak_id,scan,intensity from summed_ms2_regions where feature_id={}".format(feature_id), source_conn)

    for ms2_peak_idx in range(len(ms2_peaks_df)):
        # get all the points for this ms2 peak
        ms2_peak_id = ms2_peaks_df.loc[ms2_peak_idx].peak_id.astype(int)
        ms2_peak_df = ms2_peak_points_df.loc[(ms2_peak_points_df.peak_id==ms2_peak_id),['scan','intensity']]
        # align the two peaks in the scan dimension
        combined_df = pd.merge(base_peak_df, ms2_peak_df, on='scan', how='outer', suffixes=('_base', '_ms2')).sort_values(by='scan')
        # fill in any NaN
        combined_df.intensity_ms2.fillna(0, inplace=True)
        combined_df.intensity_base.fillna(0, inplace=True)
        # and make sure they're all integers
        combined_df.intensity_ms2 = combined_df.intensity_ms2.astype(int)
        combined_df.intensity_base = combined_df.intensity_base.astype(int)
        # calculate the correlation between the two peaks
        correlation = np.corrcoef(combined_df.intensity_base, combined_df.intensity_ms2)[1,0]
        peak_correlation.append((feature_id, base_peak_id, ms2_peak_id, float(correlation)))

print("Writing out the peak correlations for features {}-{}".format(args.feature_id_lower, args.feature_id_upper))
# feature_id, base_peak_id, ms2_peak_id, float(correlation)
src_c.executemany("INSERT INTO peak_correlation VALUES (?, ?, ?, ?)", peak_correlation)

stop_run = time.time()
print("{} seconds to process features {} to {}".format(stop_run-start_run, args.feature_id_lower, args.feature_id_upper))

# write out the processing info
peak_correlation_info.append(("run processing time (sec)", stop_run-start_run))
peak_correlation_info.append(("processed", time.ctime()))

peak_correlation_info_entry = []
peak_correlation_info_entry.append(("features {}-{}".format(args.feature_id_lower, args.feature_id_upper), ' '.join(str(e) for e in peak_correlation_info)))

src_c.executemany("INSERT INTO peak_correlation_info VALUES (?, ?)", peak_correlation_info_entry)

source_conn.commit()
source_conn.close()
