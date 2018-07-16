# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import time
import sqlite3
from pyteomics import mgf
import operator
import os.path
import argparse
import os
from multiprocessing import Pool

DELTA_MZ = 1.003355     # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
PROTON_MASS = 1.007276  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.

def run_process(process):
    os.system(process)

parser = argparse.ArgumentParser(description='Generate a text file containing the Hardklor commands.')
parser.add_argument('-fdb','--features_database', type=str, help='The name of the features database.', required=True)
parser.add_argument('-frdb','--feature_region_database', type=str, help='The name of the feature region database.', required=True)
parser.add_argument('-bfn','--base_mgf_filename', type=str, help='The base name of the MGF to give Hardklor.', required=True)
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
parser.add_argument('-mpc','--minimum_peak_correlation', type=float, default=0.6, help='Process ms2 peaks with at least this much correlation with the feature''s ms1 base peak.', required=False)
parser.add_argument('-fps','--frames_per_second', type=float, help='Effective frame rate.', required=True)
args = parser.parse_args()

mgf_directory = "{}/mgf".format(args.data_directory)
hk_directory = "{}/hk".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)
hardklor_commands_directory = "{}/hardklor-commands".format(args.data_directory)

# make sure the processing directories exist
try: 
    os.makedirs(mgf_directory)
except OSError:
    if not os.path.isdir(mgf_directory):
        raise

try: 
    os.makedirs(hk_directory)
except OSError:
    if not os.path.isdir(hk_directory):
        raise

try: 
    os.makedirs(search_headers_directory)
except OSError:
    if not os.path.isdir(search_headers_directory):
        raise

try: 
    os.makedirs(hardklor_commands_directory)
except OSError:
    if not os.path.isdir(hardklor_commands_directory):
        raise

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

print("Setting up indexes")
db_conn = sqlite3.connect(args.feature_region_database)
db_conn.cursor().execute("CREATE INDEX IF NOT EXISTS idx_peak_correlation_1 ON peak_correlation (feature_id)")

db_conn.cursor().execute("DROP TABLE IF EXISTS feature_isotopes")
db_conn.cursor().execute("DROP TABLE IF EXISTS feature_list")
db_conn.close()


db_conn = sqlite3.connect(args.feature_region_database)
feature_ids_df = pd.read_sql_query("select distinct(feature_id) from peak_correlation", db_conn)
db_conn.close()

if len(feature_ids_df) == 0:
    print("Error: no feature IDs found in peak_correlation for feature DB {}. Exiting.".format(args.feature_region_database))
    sys.exit(1)

hk_processes = []
feature_cluster_df = None

feature_list_columns = ['feature_id', 'charge_state', 'monoisotopic_mass', 'retention_time_secs', 'isotope_count', 'cluster_mz_centroid', 'cluster_summed_intensity', 'minimum_error', 'minimum_error_sulphur']
feature_list = []

for feature_ids_idx in range(0,len(feature_ids_df)):
    feature_id = feature_ids_df.loc[feature_ids_idx].feature_id.astype(int)
    print("Processing feature {}".format(feature_id))

    db_conn = sqlite3.connect(args.features_database)
    feature_df = pd.read_sql_query("select * from features where feature_id = {}".format(feature_id), db_conn)
    charge_state = feature_df.loc[0].charge_state.astype(int)
    expected_spacing = DELTA_MZ / charge_state
    db_conn.close()

    db_conn = sqlite3.connect(args.feature_region_database)
    peaks_df = pd.read_sql_query("select * from summed_ms1_regions where feature_id = {} order by peak_id".format(feature_id), db_conn)
    ms2_peaks_df = pd.read_sql_query("select * from ms2_peaks where (feature_id,peak_id) in (select feature_id,ms2_peak_id from peak_correlation where feature_id={} and correlation > {})".format(feature_id, args.minimum_peak_correlation), db_conn)
    db_conn.close()

    mzs = peaks_df.groupby('peak_id').apply(wavg, "mz", "intensity").reset_index(name='mz_centroid')
    intensities = peaks_df.groupby('peak_id').intensity.sum().reset_index(name='summed_intensity')

    cluster_df = pd.concat([mzs, intensities.summed_intensity], axis=1)
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

    indexes_to_drop = abs(cluster_df.mz_centroid.diff() - expected_spacing) > 0.5
    cluster_df.drop(cluster_df.index[indexes_to_drop], inplace=True)
    cluster_df.reset_index(drop=True, inplace=True)

    # find the combination of mono index and sulphurs that gives the smallest total height ratio error
    minimum_error = sys.float_info.max
    minimum_error_sulphur = None
    minimum_error_mono_index = None

    if (base_peak_index+1) >= len(cluster_df):
        base_peak_index = len(cluster_df)-1
        print("dodgy cluster for this feature")

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

    cluster_df['mz_mod'] = cluster_df.mz_centroid - ((cluster_df.peak_id-1)*expected_spacing)
    cluster_df['feature_id'] = feature_id

    # add to the feature clusters
    if feature_cluster_df is None:
        feature_cluster_df = cluster_df.copy()
    else:
        feature_cluster_df = feature_cluster_df.append(cluster_df, ignore_index=True)

    # calculate the centroid of the feature's cluster
    cluster_mz_centroid = wavg(cluster_df, "mz_mod", "summed_intensity")
    cluster_summed_intensity = cluster_df.summed_intensity.sum()

    # calculate the monoisotopic mass
    monoisotopic_mass = (cluster_mz_centroid - PROTON_MASS) * charge_state

    # ... and the retention time
    retention_time_secs = feature_df.loc[0].base_frame_id / args.frames_per_second

    isotope_count = len(cluster_df)
    feature_list.append((feature_id, charge_state, monoisotopic_mass, retention_time_secs, isotope_count, round(cluster_mz_centroid,6), cluster_summed_intensity, minimum_error, minimum_error_sulphur))

    pairs_df = ms2_peaks_df[['centroid_mz', 'intensity']].copy().sort_values(by=['intensity'], ascending=False)

    # Write out the spectrum
    spectra = []
    spectrum = {}
    spectrum["m/z array"] = pairs_df.centroid_mz.values
    spectrum["intensity array"] = pairs_df.intensity.values
    params = {}
    params["TITLE"] = "feature {}, file {}, correlation {}, model error {:.2f}, sulphurs {}".format(feature_id, args.base_mgf_filename, args.minimum_peak_correlation, minimum_error, minimum_error_sulphur)
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(cluster_mz_centroid,6), cluster_summed_intensity)
    params["CHARGE"] = "{}+".format(charge_state)
    params["RTINSECONDS"] = "{}".format(retention_time_secs)
    params["SCANS"] = "{}".format(feature_df.loc[0].base_frame_id.astype(int))
    spectrum["params"] = params
    spectra.append(spectrum)

    mgf_filename = "{}/feature-{}-correlation-{}.mgf".format(mgf_directory, feature_id, args.minimum_peak_correlation)
    hk_filename = "{}/feature-{}-correlation-{}.hk".format(hk_directory, feature_id, args.minimum_peak_correlation)
    header_filename = "{}/feature-{}-correlation-{}.txt".format(search_headers_directory, feature_id, args.minimum_peak_correlation)

    # write out the MGF file
    if os.path.isfile(mgf_filename):
        os.remove(mgf_filename)
    if os.path.isfile(hk_filename):
        os.remove(hk_filename)
    if os.path.isfile(header_filename):
        os.remove(header_filename)
    mgf.write(output=mgf_filename, spectra=spectra)

    # remove blank lines from the MGF file
    with open(mgf_filename, 'r') as file_handler:
        file_content = file_handler.readlines()
    file_content = [x for x in file_content if not x == '\n']
    with open(mgf_filename, 'w') as file_handler:
        file_handler.writelines(file_content)

    # write out the header with no fragment ions (with which to build the search MGF)
    spectra = []
    spectrum["m/z array"] = np.empty(0)
    spectrum["intensity array"] = np.empty(0)
    spectra.append(spectrum)
    mgf.write(output=header_filename, spectra=spectra)

    # append the Hardklor command to process it
    hk_processes.append("./hardklor/hardklor -cmd -instrument TOF -resolution 40000 -centroided 1 -ms_level 2 -algorithm Version2 -charge_algorithm Quick -charge_min 1 -charge_max {} -correlation {} -mz_window 5.25 -sensitivity 2 -depth 2 -max_features 12 -distribution_area 1 -xml 0 {} {}".format(charge_state, args.minimum_peak_correlation, mgf_filename, hk_filename))

# write out the deconvolved feature ms1 isotopes and the feature list
db_conn = sqlite3.connect(args.feature_region_database)
print("writing out the deconvolved feature ms1 isotopes...")
feature_cluster_df.to_sql(name='feature_isotopes', con=db_conn, if_exists='append', index=False)
print("writing out the feature list...")
feature_list_df = pd.DataFrame(feature_list, columns=feature_list_columns)
feature_list_df.to_sql(name='feature_list', con=db_conn, if_exists='append', index=False)
db_conn.close()

# Set up the processing pool for Hardklor
print("running Hardklor...")
pool = Pool()
pool.map(run_process, hk_processes)
