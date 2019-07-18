import pandas as pd
import numpy as np
import sys
from sys import getsizeof
from numba import njit
import argparse
import os.path
from pyteomics import mgf
import time

# for reading the de-dup pickle
from ms_deisotope import deconvolute_peaks, averagine, scoring
from ms_deisotope.deconvolution import peak_retention_strategy
from ms_peak_picker import simple_peak

# take the raw ms2 points for each precursor, deconvolute them, associate them with a feature, and write out the MGF

DELTA_MZ = 1.003355  # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.
DA_MIN = 100
DA_MAX = 4000

parser = argparse.ArgumentParser(description='Extract ms2 spectra from PASEF isolation windows.')
parser.add_argument('-cdbb','--converted_database_base', type=str, help='base path to the converted database.', required=True)
parser.add_argument('-mgf','--mgf_filename', type=str, help='File name of the MGF to be generated.', required=True)
parser.add_argument('-ddms1fn','--dedup_ms1_filename', type=str, default='./ms1_deduped_df.pkl', help='File containing de-duped ms1 features.', required=False)
parser.add_argument('-frpb','--feature_raw_points_base', type=str, help='Directory for the CSVs for the feature ms2 raw points.', required=True)
args = parser.parse_args()

def process_ms2(group_number, group_df):
    


CONVERTED_DATABASE_NAME = '{}/HeLa_20KInt.sqlite'.format(args.converted_database_base)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)


# process the ms2 raw points for each precursor
for idx,group_df in isolation_window_df.groupby('Precursor'):
    process_ms2(group_number=idx+1, group_df=group_df)







def collate_spectra_for_feature(feature_df, ms2_deconvoluted_df):
    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    pairs_df = ms2_deconvoluted_df[['singley_charged_monoisotope_mz', 'intensity']].copy().sort_values(by=['singley_charged_monoisotope_mz'], ascending=True)
    spectrum = {}
    spectrum["m/z array"] = np.round(pairs_df.singley_charged_monoisotope_mz.values,4)
    spectrum["intensity array"] = pairs_df.intensity.values
    params = {}
    params["TITLE"] = "RawFile: {} Charge: {} FeatureIntensity: {} Feature#: {} RtApex: {}".format(os.path.basename(CONVERTED_DATABASE_NAME).split('.')[0], feature_df.charge, feature_df.intensity, feature_df.feature_id, round(feature_df.rt_apex,2))
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(feature_df.monoisotopic_mz,6), feature_df.intensity)
    params["CHARGE"] = "{}+".format(feature_df.charge)
    params["RTINSECONDS"] = "{}".format(round(feature_df.rt_apex,2))
    spectrum["params"] = params
    return spectrum



# load the features we detected
ms1_deduped_df = pd.read_pickle(args.dedup_ms1_filename)
time_taken = []
for idx,feature_df in ms1_deduped_df.iterrows():
    # read the raw ms2 for this feature
    feature_raw_ms2_df = pd.read_csv('{}/feature-{}-ms2-raw-points.csv'.format(args.feature_raw_points_base, feature_df.feature_id))
    # deconvolute the raw points into peaks
    start_time = time.time()
    ms2_deconvoluted_df = deconvolute_ms2(mass_defect_window_bins, feature_raw_ms2_df)
    stop_time = time.time()
    ms2_deconvoluted_df.to_csv('./feature-{}-ms2-peaks-after-deconvolution-sfpd.csv'.format(feature_df.feature_id), index=False, header=True)
    time_taken.append(stop_time-start_time)
    # package the feature and its fragment ions for writing out to the MGF
    result = collate_spectra_for_feature(feature_df, ms2_deconvoluted_df)
    feature_results.append(result)

print("average deconvolution: {} seconds (N={})".format(round(np.average(time_taken),6), len(time_taken)))
# generate the MGF for all the features
print("generating the MGF: {}".format(args.mgf_filename))
if os.path.isfile(args.mgf_filename):
    os.remove(args.mgf_filename)
mgf.write(output=args.mgf_filename, spectra=feature_results)
