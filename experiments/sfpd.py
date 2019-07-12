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

# take the raw ms2 points for each feature, deconvolute them with the SFPD method, and write out the MGF

DELTA_MZ = 1.003355  # Mass difference between Carbon-12 and Carbon-13 isotopes, in Da. For calculating the spacing between isotopic peaks.
PROTON_MASS = 1.0073  # Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.
DA_MIN = 100
DA_MAX = 4000
charge_states_to_consider = [1,2]

parser = argparse.ArgumentParser(description='Extract ms1 features from PASEF isolation windows.')
parser.add_argument('-cdbb','--converted_database_base', type=str, help='base path to the converted database.', required=True)
parser.add_argument('-mgf','--mgf_filename', type=str, help='File name of the MGF to be generated.', required=True)
parser.add_argument('-ddms1fn','--dedup_ms1_filename', type=str, default='./ms1_deduped_df.pkl', help='File containing de-duped ms1 features.', required=False)
parser.add_argument('-frpb','--feature_raw_points_base', type=str, help='Directory for the CSVs for the feature ms2 raw points.', required=True)
args = parser.parse_args()

CONVERTED_DATABASE_NAME = '{}/HeLa_20KInt.sqlite'.format(args.converted_database_base)
if not os.path.isfile(CONVERTED_DATABASE_NAME):
    print("The converted database doesn't exist: {}".format(CONVERTED_DATABASE_NAME))
    sys.exit(1)

# from https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
def find_runs(x):
    # find runs of consecutive items in an array

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

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

def generate_mass_defect_windows():
    # generate a charge-aware mask of mass defect windows
    mass_defect_window_bins = {}
    for charge in charge_states_to_consider:
        bin_edges_l = []
        for nominal_mass in np.arange(start=DA_MIN, stop=DA_MAX, step=1/charge):
            proton_mass_adjustment = (charge - 1) * PROTON_MASS
            mass_centre = (nominal_mass + proton_mass_adjustment) * 1.00048  # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3184890/
            width = (0.19 + (0.0001 * nominal_mass)) / charge
            lower_mass = mass_centre - (width / 2)
            upper_mass = mass_centre + (width / 2)
            bin_edges_l.append(lower_mass)
            bin_edges_l.append(upper_mass)
        bins = np.asarray(bin_edges_l)
        mass_defect_window_bins['charge-{}'.format(charge)] = bins
    return mass_defect_window_bins

def deconvolute_ms2(mass_defect_window_bins, feature_raw_ms2_df):
    mz_a = feature_raw_ms2_df.mz.to_numpy()
    intensity_a = feature_raw_ms2_df.intensity.to_numpy()
    available_a = np.full((len(mz_a)), True)  # the raw points not yet assigned to a peak

    # cycle through the charge states, from higher to lower
    peaks_l = []
    for charge in charge_states_to_consider[::-1]:
        decharged_mass_a = ((feature_raw_ms2_df.mz + PROTON_MASS) * charge).to_numpy()
        bins = mass_defect_window_bins['charge-{}'.format(charge)]

        digitised_mass = np.digitize(decharged_mass_a[available_a == True], bins)  # an odd index means the point is inside a mass defect window

        # remove all the even indexes - the odd indexes are the mass defect windows
        mass_defect_window_indexes = digitised_mass[(digitised_mass % 2) == 1]

        # remove the duplicates
        unique_mass_defect_window_indexes = np.unique(mass_defect_window_indexes)

        # find where the windows are consecutive
        condition = (np.diff(unique_mass_defect_window_indexes) == 2)
        condition = np.hstack((condition[0],condition))  # now the flags correspond to whether each mass window containing signal is adjacent to the next one

        # the False values are stand-alone peaks we need to process as orphaned isotopes
        run_values, run_starts, run_lengths = find_runs(condition)
        chunk_sizes = run_lengths[run_values == True]
        chunk_starts = run_starts[run_values == True]

        expected_peak_spacing = DELTA_MZ / charge
        for series_idx in range(len(chunk_sizes)):
            number_of_windows = chunk_sizes[series_idx]
            start_idx = chunk_starts[series_idx]
            index_list = unique_mass_defect_window_indexes[start_idx:start_idx+number_of_windows].tolist()
            mz_l = []
            int_l = []
            for peak_idx,i in enumerate(index_list):
                lower_mass = round(bins[i-1],4)
                upper_mass = round(bins[i],4)
                # get the raw points allocated to this bin
                peak_indexes = np.where(digitised_mass == i)[0]
                mz_centroid = np.average(mz_a[peak_indexes], weights=intensity_a[peak_indexes])
                intensity = np.sum(intensity_a[peak_indexes])
                mz_l.append(mz_centroid)
                int_l.append(intensity)
                available_a[peak_indexes] = False
            # de-isotope the peaks
            peaks_mz_a = np.array(mz_l)
            peaks_int_a = np.array(int_l)
            for i in range(len(peaks_mz_a)):
                peaks_mz_a[i] = peaks_mz_a[i] - (i * expected_peak_spacing)
            deisotoped_mz = np.average(peaks_mz_a, weights=peaks_int_a)
            deisotoped_intensity = peaks_int_a.sum()
            monoisotopic_mass = (deisotoped_mz - PROTON_MASS) * charge
            # the MGF wants the m/z of the monoisotope (the de-isotoped m/z) as it would be if it was a single-charge ion
            singley_charged_monoisotope_mz = (deisotoped_mz * charge) - (PROTON_MASS * (charge - 1))
            peaks_l.append((series_idx, singley_charged_monoisotope_mz, deisotoped_intensity))

        if charge == 1:
            # process the peaks that are not allocated to an isotopic series - the orphans
            orphaned_chunk_sizes = run_lengths[run_values == False]
            orphaned_chunk_starts = run_starts[run_values == False]

            peak_idx = 1
            for i in range(len(orphaned_chunk_sizes)):
                chunk_size = orphaned_chunk_sizes[i]
                chunk_start = orphaned_chunk_starts[i]
                for chunk_idx in range(chunk_size):
                    index = unique_mass_defect_window_indexes[chunk_start+chunk_idx]
                    lower_mass = round(bins[index-1],4)
                    upper_mass = round(bins[index],4)
                    # get the raw points allocated to this bin
                    peak_indexes = np.where(digitised_mass == index)[0]
                    mz_centroid = np.average(mz_a[peak_indexes], weights=intensity_a[peak_indexes])
                    intensity = np.sum(intensity_a[peak_indexes])
                    monoisotopic_mass = (mz_centroid - PROTON_MASS) * charge
                    singley_charged_monoisotope_mz = (deisotoped_mz * charge) - (PROTON_MASS * (charge - 1))
                    peaks_l.append((0, singley_charged_monoisotope_mz, intensity))
                    peak_idx += 1

    peaks_df = pd.DataFrame(peaks_l, columns=['series','singley_charged_monoisotope_mz','intensity'])
    return peaks_df


feature_results = []
# create windows for each charge state
mass_defect_window_bins = generate_mass_defect_windows()
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
