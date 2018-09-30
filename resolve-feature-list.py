# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import time
import sqlite3
import operator
import os.path
import argparse
import os
import json
import peakutils

DELTA_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
PROTON_MASS = 1.007276  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

parser = argparse.ArgumentParser(description='Finalise the feature list and the deconvoluted ions for each feature.')
parser.add_argument('-fdb','--features_database', type=str, help='The name of the features database.', required=True)
parser.add_argument('-frdb','--feature_region_database', type=str, help='The name of the feature region database.', required=True)
parser.add_argument('-fl','--feature_id_lower', type=int, help='Lower feature ID to process.', required=True)
parser.add_argument('-fu','--feature_id_upper', type=int, help='Upper feature ID to process.', required=True)
args = parser.parse_args()

# Store the arguments as metadata in the database for later reference
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))

start_run = time.time()

def standard_deviation(mz):
    instrument_resolution = 40000.0
    return ((mz / instrument_resolution) / 2.35482)

def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

# From "A Model-Based Method for the Prediction of the Isotopic Distribution of Peptides", Dirk Valkenborg, 
# Ivy Jansen, and Tomasz Burzykowski, J Am Soc Mass Spectrom 2008, 19, 703–712

MAX_NUMBER_OF_SULPHUR_ATOMS = 3
MAX_NUMBER_OF_PREDICTED_RATIOS = 6

S0_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=object)
S0_r[1] = [-0.00142320578040, 0.53158267080224, 0.00572776591574, -0.00040226083326, -0.00007968737684]
S0_r[2] = [0.06258138406507, 0.24252967352808, 0.01729736525102, -0.00427641490976, 0.00038011211412]
S0_r[3] = [0.03092092306220, 0.22353930450345, -0.02630395501009, 0.00728183023772, -0.00073155573939]
S0_r[4] = [-0.02490747037406, 0.26363266501679, -0.07330346656184, 0.01876886839392, -0.00176688757979]
S0_r[5] = [-0.19423148776489, 0.45952477474223, -0.18163820209523, 0.04173579115885, -0.00355426505742]
S0_r[6] = [0.04574408690798, -0.05092121193598, 0.13874539944789, -0.04344815868749, 0.00449747222180]

S1_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=object)
S1_r[1] = [-0.01040584267474, 0.53121149663696, 0.00576913817747, -0.00039325152252, -0.00007954180489]
S1_r[2] = [0.37339166598255, -0.15814640001919, 0.24085046064819, -0.06068695741919, 0.00563606634601]
S1_r[3] = [0.06969331604484, 0.28154425636993, -0.08121643989151, 0.02372741957255, -0.00238998426027]
S1_r[4] = [0.04462649178239, 0.23204790123388, -0.06083969521863, 0.01564282892512, -0.00145145206815]
S1_r[5] = [-0.20727547407753, 0.53536509500863, -0.22521649838170, 0.05180965157326, -0.00439750995163]
S1_r[6] = [0.27169670700251, -0.37192045082925, 0.31939855191976, -0.08668833166842, 0.00822975581940]

S2_r = np.empty(MAX_NUMBER_OF_PREDICTED_RATIOS+1, dtype=object)
S2_r[1] = [-0.01937823810470, 0.53084210514216, 0.00580573751882, -0.00038281138203, -0.00007958217070]
S2_r[2] = [0.68496829280011, -0.54558176102022, 0.44926662609767, -0.11154849560657, 0.01023294598884]
S2_r[3] = [0.04215807391059, 0.40434195078925, -0.15884974959493, 0.04319968814535, -0.00413693825139]
S2_r[4] = [0.14015578207913, 0.14407679007180, -0.01310480312503, 0.00362292256563, -0.00034189078786]
S2_r[5] = [-0.02549241716294, 0.32153542852101, -0.11409513283836, 0.02617210469576, -0.00221816103608]
S2_r[6] = [-0.14490868030324, 0.33629928307361, -0.08223564735018, 0.01023410734015, -0.00027717589598]

model_params = np.empty(MAX_NUMBER_OF_SULPHUR_ATOMS, dtype=object)
model_params[0] = S0_r
model_params[1] = S1_r
model_params[2] = S2_r

# Find the ratio of H(peak_number)/H(peak_number-1) for peak_number=1..6
# peak_number = 0 refers to the monoisotopic peak
# number_of_sulphur = number of sulphur atoms in the molecule
def peak_ratio(monoisotopic_mass, peak_number, number_of_sulphur):
    ratio = 0.0
    if (((1 <= peak_number <= 3) & (((number_of_sulphur == 0) & (498 <= monoisotopic_mass <= 3915)) |
                                    ((number_of_sulphur == 1) & (530 <= monoisotopic_mass <= 3947)) |
                                    ((number_of_sulphur == 2) & (562 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 4) & (((number_of_sulphur == 0) & (907 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (939 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (971 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 5) & (((number_of_sulphur == 0) & (1219 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (1251 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (1283 <= monoisotopic_mass <= 3978)))) |
       ((peak_number == 6) & (((number_of_sulphur == 0) & (1559 <= monoisotopic_mass <= 3915)) |
                              ((number_of_sulphur == 1) & (1591 <= monoisotopic_mass <= 3947)) |
                              ((number_of_sulphur == 2) & (1623 <= monoisotopic_mass <= 3978))))):
        beta0 = model_params[number_of_sulphur][peak_number][0]
        beta1 = model_params[number_of_sulphur][peak_number][1]
        beta2 = model_params[number_of_sulphur][peak_number][2]
        beta3 = model_params[number_of_sulphur][peak_number][3]
        beta4 = model_params[number_of_sulphur][peak_number][4]
        scaled_m = monoisotopic_mass / 1000.0
        ratio = beta0 + (beta1*scaled_m) + beta2*(scaled_m**2) + beta3*(scaled_m**3) + beta4*(scaled_m**4)
    return ratio

feature_list = []
feature_isotopes_list = []

db_conn = sqlite3.connect(args.feature_region_database)
db_conn.cursor().execute("DROP TABLE IF EXISTS feature_isotopes")
db_conn.cursor().execute("CREATE TABLE feature_isotopes (feature_id INTEGER, feature_region_peak_id INTEGER, centroid_scan REAL, centroid_rt REAL, centroid_mz REAL, peak_summed_intensity INTEGER, PRIMARY KEY(feature_id, feature_region_peak_id))")
db_conn.cursor().execute("DROP TABLE IF EXISTS feature_list")
db_conn.cursor().execute("CREATE TABLE feature_list (feature_id INTEGER, charge_state INTEGER, monoisotopic_mass REAL, centroid_scan REAL, centroid_rt REAL, centroid_mz REAL, start_rt, end_rt, scan_lower, scan_upper, summed_intensity INTEGER, isotope_count INTEGER, PRIMARY KEY(feature_id))")
db_conn.close()

for feature_id in range(args.feature_id_lower, args.feature_id_upper+1):
    print("Processing feature {} ({}% complete)".format(feature_id, round(float(feature_id-args.feature_id_lower)/(args.feature_id_upper-args.feature_id_lower+1)*100,1)))

    db_conn = sqlite3.connect(args.features_database)
    feature_df = pd.read_sql_query("select * from features where feature_id = {}".format(feature_id), db_conn)
    charge_state = feature_df.loc[0].charge_state.astype(int)
    expected_spacing = DELTA_MZ / charge_state
    db_conn.close()

    # get the ms1 peaks
    db_conn = sqlite3.connect(args.feature_region_database)
    summed_ms1_region_df = pd.read_sql_query("select * from summed_ms1_regions where feature_id = {} order by peak_id".format(feature_id), db_conn)
    db_conn.close()

    if len(summed_ms1_region_df)>0:
        # determine which of the candidate peaks for the feature are true isotopic peaks
        mzs = summed_ms1_region_df.groupby('peak_id').apply(wavg, "mz", "intensity").reset_index(name='mz_centroid')
        intensities = summed_ms1_region_df.groupby('peak_id').intensity.sum().reset_index(name='summed_intensity')
        scans = summed_ms1_region_df.groupby('peak_id').apply(wavg, "scan", "intensity").reset_index(name='scan_centroid')

        cluster_df = pd.concat([mzs, scans.scan_centroid, intensities.summed_intensity], axis=1)
        cluster_df.sort_values(by='mz_centroid', inplace=True)

        cluster_df.reset_index(drop=True, inplace=True)
        base_peak_index = cluster_df.summed_intensity.idxmax()

        base_peak_mz = cluster_df.iloc[base_peak_index].mz_centroid
        std_dev = standard_deviation(base_peak_mz)

        spacing_from_base = abs(cluster_df.mz_centroid - base_peak_mz) % expected_spacing
        close_to_next_isotope = (abs(spacing_from_base - expected_spacing) < (std_dev * 4))
        close_to_this_isotope = spacing_from_base < (std_dev * 4)
        indexes_to_drop = ~(close_to_next_isotope | close_to_this_isotope)
        cluster_df.drop(cluster_df.index[indexes_to_drop], inplace=True)
        cluster_df.reset_index(drop=True, inplace=True)

        base_peak_index = cluster_df.summed_intensity.idxmax()
        base_peak_mz = cluster_df.iloc[base_peak_index].mz_centroid
        base_peak_id = cluster_df.iloc[base_peak_index].peak_id.astype(int)

        indexes_to_drop = abs(cluster_df.mz_centroid.diff() - expected_spacing) > 0.5
        cluster_df.drop(cluster_df.index[indexes_to_drop], inplace=True)
        cluster_df.reset_index(drop=True, inplace=True)

        if len(cluster_df) > 0:

            # find the combination of mono index and sulphurs that gives the smallest total height ratio error
            minimum_error = sys.float_info.max
            minimum_error_sulphur = None
            minimum_error_mono_index = None

            if (base_peak_index+1) >= len(cluster_df):
                base_peak_index = len(cluster_df)-1
                print("dodgy cluster for this feature")

            updated_min_error = False
            for test_mono_index in range(0,base_peak_index+1):  # consider moving it up to the base peak (but not beyond)
                test_monoisotopic_mass = (cluster_df.loc[test_mono_index].mz_centroid - PROTON_MASS) * charge_state
                for sulphur in range(0,MAX_NUMBER_OF_SULPHUR_ATOMS):
                    error = 0
                    number_of_peaks_to_test = min(MAX_NUMBER_OF_PREDICTED_RATIOS, len(cluster_df)-test_mono_index)
                    for peak_number in range(1,number_of_peaks_to_test):
                        predicted_ratio = peak_ratio(test_monoisotopic_mass, peak_number=peak_number, number_of_sulphur=sulphur)
                        if predicted_ratio > 0:
                            observed_ratio = cluster_df.loc[test_mono_index+peak_number].summed_intensity / cluster_df.loc[test_mono_index+peak_number-1].summed_intensity
                            error += (predicted_ratio - observed_ratio)**2 / predicted_ratio
                        if error < minimum_error:
                            minimum_error = error
                            minimum_error_sulphur = sulphur
                            minimum_error_mono_index = test_mono_index
                            updated_min_error = True

            if updated_min_error:
                error_as_string = "{:.2f}".format(minimum_error)
            else:
                error_as_string = "None"

            # update the cluster according to the mono index
            cluster_df = cluster_df.loc[minimum_error_mono_index:].copy()
            cluster_df.reset_index(drop=True, inplace=True)

            # de-isotope the cluster's peaks by folding-in the mz centroid of each peak
            cluster_df['mz_mod'] = cluster_df.mz_centroid - (cluster_df.index * expected_spacing)
            cluster_df['feature_id'] = feature_id

            # calculate the intensity-weighted centroid of the feature's folded-in peak
            deisotoped_mz_centroid = wavg(cluster_df, "mz_mod", "summed_intensity")
            isotopes_summed_intensity = cluster_df.summed_intensity.sum()

            # calculate the monoisotopic mass
            monoisotopic_mass = (deisotoped_mz_centroid - PROTON_MASS) * charge_state

            # trim the summed ms1 region to include only those peaks in the resolved cluster
            cluster_peaks_df = cluster_df[['feature_id','peak_id']].copy()
            cluster_peaks_df['feature_peak'] = cluster_peaks_df['feature_id'].map(str) + '|' + cluster_peaks_df['peak_id'].map(str)
            summed_ms1_region_df['feature_peak'] = summed_ms1_region_df['feature_id'].map(str) + '|' + summed_ms1_region_df['peak_id'].map(str)
            summed_ms1_region_df = summed_ms1_region_df[summed_ms1_region_df.feature_peak.isin(cluster_peaks_df.feature_peak)]

            # make the column names a bit more meaningful
            summed_ms1_region_df.rename(columns={"peak_id":"feature_peak_id","point_id":"feature_point_id"}, inplace=True)
            summed_ms1_region_df.drop(['mz', 'scan', 'intensity', 'number_frames', 'feature_peak'], axis=1, inplace=True)

            # add the summed_frame_point that contributed to each feature region point
            db_conn = sqlite3.connect(args.feature_region_database)
            ms1_feature_frame_join_df = pd.read_sql_query("select * from ms1_feature_frame_join where feature_id={}".format(feature_id), db_conn)
            ms1_feature_frame_join_df.rename(columns={"frame_id":"summed_frame_id"}, inplace=True)
            ms1_feature_frame_join_df.drop(['feature_id', 'feature_point_id', 'frame_point_id'], axis=1, inplace=True)
            db_conn.close()

            summed_ms1_region_df = pd.merge(summed_ms1_region_df, ms1_feature_frame_join_df, how='left', left_on=['feature_point'], right_on=['feature_point'])
            summed_ms1_region_df.drop(['feature_point'], axis=1, inplace=True)

            # add the raw_frame_point that contributed to each summed frame point
            db_conn = sqlite3.connect(args.features_database)
            raw_summed_join_df = pd.read_sql_query("select * from raw_summed_join where summed_frame_point in {}".format(tuple(summed_ms1_region_df.summed_frame_point.astype(str))), db_conn)
            raw_summed_join_df.drop(['summed_frame_id','summed_point_id'], axis=1, inplace=True)
            db_conn.close()

            summed_ms1_region_df = pd.merge(summed_ms1_region_df, raw_summed_join_df, how='left', left_on=['summed_frame_point'], right_on=['summed_frame_point'])
            summed_ms1_region_df.drop(['summed_frame_id','summed_frame_point','retention_time_secs'], axis=1, inplace=True)

            # get the raw frame point's intensity
            db_conn = sqlite3.connect(args.features_database)
            raw_frames_df = pd.read_sql_query("select * from frames where raw_frame_point in {}".format(tuple(summed_ms1_region_df.raw_frame_point.astype(str))), db_conn)
            db_conn.close()

            summed_ms1_region_df = pd.merge(summed_ms1_region_df, raw_frames_df, how='left', left_on=['raw_frame_point'], right_on=['raw_frame_point'])
            summed_ms1_region_df.drop(['peak_id','frame_id','raw_frame_point','point_id'], axis=1, inplace=True)

            # summed_ms1_region_df.info()
            # cluster_df.info()

            print("feature {}, cluster peaks {}, summed region points {}".format(feature_id, len(cluster_df), len(summed_ms1_region_df)))

            # for each feature peak, use the raw points to find the RT and drift intensity-weighted centroids
            for peak_idx in range(len(cluster_df)):
                peak_id = cluster_df.iloc[peak_idx].peak_id
                peak_summed_intensity = cluster_df.iloc[peak_idx].summed_intensity
                peak_points_df = summed_ms1_region_df.loc[summed_ms1_region_df.feature_peak_id==peak_id]
                centroid_scan = peakutils.centroid(peak_points_df.scan.astype(float), peak_points_df.intensity)
                centroid_rt = peakutils.centroid(peak_points_df.retention_time_secs.astype(float), peak_points_df.intensity)
                centroid_mz = peakutils.centroid(peak_points_df.mz.astype(float), peak_points_df.intensity)
                feature_isotopes_list.append((feature_id, peak_id, centroid_scan, centroid_rt, centroid_mz, peak_summed_intensity))

            # Collect the feature's attributes. Centroids are calculated using the raw points.
            feature_points_df = summed_ms1_region_df.loc[summed_ms1_region_df.feature_id==feature_id]
            feature_centroid_scan = peakutils.centroid(feature_points_df.scan.astype(float), feature_points_df.intensity)
            feature_centroid_rt = peakutils.centroid(feature_points_df.retention_time_secs.astype(float), feature_points_df.intensity)
            feature_centroid_mz = peakutils.centroid(feature_points_df.mz.astype(float), feature_points_df.intensity)
            start_rt = feature_points_df.retention_time_secs.min()
            end_rt = feature_points_df.retention_time_secs.max()
            scan_lower = feature_points_df.scan.min()
            scan_upper = feature_points_df.scan.max()
            feature_summed_intensity = feature_points_df.intensity.sum()
            isotope_count = len(cluster_df)

            # add the feature to the list
            feature_list.append((feature_id, charge_state, monoisotopic_mass, feature_centroid_scan, feature_centroid_rt, feature_centroid_mz, start_rt, end_rt, scan_lower, scan_upper, feature_summed_intensity, isotope_count))
        else:
            print("feature {}: there are no ms1 peaks remaining, so we're not including this feature.".format(feature_id))
    else:
        print("feature {}: there are no candidate ms1 peaks, so we're not including this feature.".format(feature_id))

# write out the deconvolved feature ms1 isotopes
db_conn = sqlite3.connect(args.feature_region_database)

print("writing out the deconvolved feature ms1 isotopes...")
db_conn.cursor().executemany("INSERT INTO feature_isotopes VALUES (?, ?, ?, ?, ?, ?)", feature_isotopes_list)

# ... and the feature list
print("writing out the feature list...")
db_conn.cursor().executemany("INSERT INTO feature_list VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", feature_list)

db_conn.commit()
db_conn.close()

stop_run = time.time()

info.append(("features resolved", len(feature_list)))
info.append(("run processing time (sec)", stop_run-start_run))
info.append(("processed", time.ctime()))
info.append(("processor", parser.prog))

print("{} info: {}".format(parser.prog, info))

info_entry = []
info_entry.append(("{}".format(os.path.basename(args.feature_region_database).split('.')[0]), json.dumps(info)))

info_entry_df = pd.DataFrame(info_entry, columns=['item', 'value'])
db_conn = sqlite3.connect(args.feature_region_database)
info_entry_df.to_sql(name='resolve_feature_list_info', con=db_conn, if_exists='replace', index=False)
db_conn.close()
