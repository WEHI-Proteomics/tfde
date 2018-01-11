import pandas as pd
import argparse
import numpy as np
import sqlite3


# feature / base peak array indices
FEATURE_ID_IDX = 0
FEATURE_BASE_PEAK_ID_IDX = 1

# base peak points array indices
BASE_PEAK_POINT_ID_IDX = 0
BASE_PEAK_MZ_IDX = 1
BASE_PEAK_SCAN_IDX = 2
BASE_PEAK_INTENSITY_IDX = 3

# MS2 peak points array indices
MS2_PEAK_POINTS_PEAK_ID_IDX = 0
MS2_PEAK_POINTS_POINT_ID_IDX = 1
MS2_PEAK_POINTS_MZ_IDX = 2
MS2_PEAK_POINTS_SCAN_IDX = 3
MS2_PEAK_POINTS_INTENSITY_IDX = 4

# Number of points either side of the base peak's maximum intensity to check for correlation
BASE_PEAK_CORRELATION_SIDE_POINTS = 3

def calculate_correlation(base_peak_points, ms2_peak_points):
    # find the maximum point of the base peak
    base_peak_max_idx = np.argmax(base_peak_points[:,BASE_PEAK_INTENSITY_IDX])

    base_peak_lower_idx = base_peak_max_idx-BASE_PEAK_CORRELATION_SIDE_POINTS
    base_peak_upper_idx = base_peak_max_idx+BASE_PEAK_CORRELATION_SIDE_POINTS

    # find the scan numbers to reference
    corr_scan_lower = base_peak_points[base_peak_lower_idx,BASE_PEAK_SCAN_IDX]
    corr_scan_upper = base_peak_points[base_peak_upper_idx,BASE_PEAK_SCAN_IDX]

    base_peak_intensity_vector = base_peak_points[base_peak_lower_idx:base_peak_upper_idx+1,BASE_PEAK_INTENSITY_IDX]
    ms2_peak_intensity_vector = ms2_peak_points[np.where((ms2_peak_points[:,MS2_PEAK_POINTS_SCAN_IDX] >= corr_scan_lower) & (ms2_peak_points[:,MS2_PEAK_POINTS_SCAN_IDX] <= corr_scan_upper))[0], MS2_PEAK_POINTS_INTENSITY_IDX]
    # print("base {}, ms2 {}".format(len(base_peak_intensity_vector), len(ms2_peak_intensity_vector)))

    if len(ms2_peak_intensity_vector) < len(base_peak_intensity_vector):
        # pad the ms2 peak vector to be the same size
        pad_length = len(base_peak_intensity_vector) - len(ms2_peak_intensity_vector)
        ms2_peak_intensity_vector = np.pad(ms2_peak_intensity_vector,(0,pad_length),'constant')
    elif len(ms2_peak_intensity_vector) > len(base_peak_intensity_vector):
        # truncate the ms2 peak vector to be the same size
        ms2_peak_intensity_vector = ms2_peak_intensity_vector[:len(base_peak_intensity_vector)]

    # Calculate the correlation of the two vectors
    correlation = np.corrcoef(base_peak_intensity_vector, ms2_peak_intensity_vector)[1,0]
    if np.isnan(correlation):
        correlation = 0.0
    # print("base {}, ms2 {}, correlation {}".format(base_peak_intensity_vector, ms2_peak_intensity_vector, correlation))
    return correlation


parser = argparse.ArgumentParser(description='Calculate correlation between MS1 and MS2 peaks for features.')
parser.add_argument('-db','--database_name', type=str, help='The name of the source database.', required=True)
args = parser.parse_args()

source_conn = sqlite3.connect(database=args.database_name, timeout=60)
src_c = source_conn.cursor()

print("Setting up tables and indexes")
src_c.execute('''DROP TABLE IF EXISTS peak_correlation''')
src_c.execute('''CREATE TABLE peak_correlation (feature_id INTEGER, base_peak_id INTEGER, ms2_peak_id INTEGER, correlation REAL, PRIMARY KEY (feature_id, base_peak_id, ms2_peak_id))''')

print("Loading the MS1 base peaks")
features_df = pd.read_sql_query("select feature_id,base_peak_id from feature_base_peaks order by feature_id ASC;", source_conn)
features_v = features_df.values
print("found features {}-{}".format(int(np.min(features_v[:,FEATURE_ID_IDX])), int(np.max(features_v[:,FEATURE_ID_IDX]))))

peak_correlation = []

print("Finding peak correlations")
for feature in features_v:
    feature_id = int(feature[FEATURE_ID_IDX])
    base_peak_id = int(feature[FEATURE_BASE_PEAK_ID_IDX])

    feature_base_peak_points_df = pd.read_sql_query("select point_id,mz,scan,intensity from summed_ms1_regions where feature_id={} and peak_id={} order by scan ASC;".format(feature_id,base_peak_id), source_conn)
    feature_base_peak_points_v = feature_base_peak_points_df.values

    # Load the MS2 peak points
    ms2_peak_points_df = pd.read_sql_query("select peak_id,point_id,mz,scan,intensity from summed_ms2_regions where ms1_feature_id={} order by peak_id,scan ASC;".format(feature_id), source_conn)
    ms2_peak_points_v = ms2_peak_points_df.values

    if len(ms2_peak_points_v) > 0:

        ms2_peak_id_lower = int(np.min(ms2_peak_points_v[:,MS2_PEAK_POINTS_PEAK_ID_IDX]))
        ms2_peak_id_upper = int(np.max(ms2_peak_points_v[:,MS2_PEAK_POINTS_PEAK_ID_IDX]))
        for ms2_peak_id in range(ms2_peak_id_lower, ms2_peak_id_upper+1):
            # Find the points for this MS2 peak
            points_v = ms2_peak_points_v[np.where(ms2_peak_points_v[:,MS2_PEAK_POINTS_PEAK_ID_IDX] == ms2_peak_id)[0]]

            # Calculate the correlation between this MS2 peak and the MS1 base peak
            correlation = calculate_correlation(feature_base_peak_points_v, points_v)
            # print("feature {}, ms1 base peak ID {}, ms2 peak ID {}, correlation {}".format(feature_id, base_peak_id, ms2_peak_id, correlation))
            peak_correlation.append((feature_id, base_peak_id, ms2_peak_id, correlation))
    else:
        print("No MS2 peak points found for feature {}".format(feature_id))

print("Writing out the peak correlations")
src_c.executemany("INSERT INTO peak_correlation VALUES (?, ?, ?, ?)", peak_correlation)

source_conn.commit()
source_conn.close()
