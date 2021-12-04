import pandas as pd
from os.path import expanduser
import json
import alphatims.bruker
import os
import numpy as np
from ms_deisotope import deconvolute_peaks, averagine, scoring
from ms_deisotope.deconvolution import peak_retention_strategy
import ray
import multiprocessing as mp
import sys



# Mass of a proton in unified atomic mass units, or Da. For calculating the monoisotopic mass.
PROTON_MASS = 1.00727647

proportion_of_cores_to_use = 0.8

experiment_name = 'P3856_YHE211'
run_name = 'P3856_YHE211_1_Slot1-1_1_5104'

tfde_results_base_dir = '/media/data-4t-a/results-P3856_YHE211/2021-10-06-06-59-25/P3856_YHE211'
precursor_cuboids_name = '{}/precursor-cuboids-pasef/exp-{}-run-{}-precursor-cuboids-pasef.feather'.format(tfde_results_base_dir,experiment_name,run_name)

precursor_cuboids_df = pd.read_feather(precursor_cuboids_name)

tdid_experiment_name = 'P3856'
tdid_results_name = 'minvi-2000-2021-12-03-21-09-45'
features_dir = '/media/big-ssd/results-{}-3did/{}/features-3did'.format(tdid_experiment_name, tdid_results_name)
features_file = '{}/exp-{}-run-{}-features-3did-dedup.feather'.format(features_dir, tdid_experiment_name, run_name)
print('loading features from {}'.format(features_file))

features_df = pd.read_feather(features_file)

number_features_detected = len(features_df)

min_charge = features_df.charge.min()
max_charge = features_df.charge.max()
print('charge range: {} to {}'.format(min_charge,max_charge))

RAW_DATABASE_BASE_DIR = '/media/big-ssd/experiments/{}/raw-databases/denoised'.format(experiment_name)

# check the raw database exists
RAW_DATABASE_NAME = "{}/{}.d".format(RAW_DATABASE_BASE_DIR, run_name)
if not os.path.exists(RAW_DATABASE_NAME):
    print("The raw database is required but doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)



# create the TimsTOF object
RAW_HDF_FILE = '{}.hdf'.format(run_name)
RAW_HDF_PATH = '{}/{}'.format(RAW_DATABASE_BASE_DIR, RAW_HDF_FILE)
if not os.path.isfile(RAW_HDF_PATH):
    print('{} doesn\'t exist so loading the raw data from {}'.format(RAW_HDF_PATH, RAW_DATABASE_NAME))
    data = alphatims.bruker.TimsTOF(RAW_DATABASE_NAME)
    print('saving to {}'.format(RAW_HDF_PATH))
    _ = data.save_as_hdf(
        directory=RAW_DATABASE_BASE_DIR,
        file_name=RAW_HDF_FILE,
        overwrite=True
    )
else:
    print('loading raw data from {}'.format(RAW_HDF_PATH))
    data = alphatims.bruker.TimsTOF(RAW_HDF_PATH)

INSTRUMENT_RESOLUTION = 40000.0

# find 3sigma for a specified m/z
def calculate_peak_delta(mz):
    delta_m = mz / INSTRUMENT_RESOLUTION  # FWHM of the peak
    sigma = delta_m / 2.35482  # std dev is FWHM / 2.35482. See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    peak_delta = 3 * sigma  # 99.7% of values fall within +/- 3 sigma
    return peak_delta

# calculate the intensity-weighted centroid
# takes a numpy array of intensity, and another of mz
def intensity_weighted_centroid(_int_f, _x_f):
    return ((_int_f/_int_f.sum()) * _x_f).sum()

# peaks_a is a numpy array of [mz,intensity]
# returns a numpy array of [intensity_weighted_centroid,summed_intensity]
def intensity_descent(peaks_a, peak_delta=None):
    # intensity descent
    peaks_l = []
    while len(peaks_a) > 0:
        # find the most intense point
        max_intensity_index = np.argmax(peaks_a[:,1])
        peak_mz = peaks_a[max_intensity_index,0]
        if peak_delta == None:
            peak_delta = calculate_peak_delta(mz=peak_mz)
        peak_mz_lower = peak_mz - peak_delta
        peak_mz_upper = peak_mz + peak_delta

        # get all the raw points within this m/z region
        peak_indexes = np.where((peaks_a[:,0] >= peak_mz_lower) & (peaks_a[:,0] <= peak_mz_upper))[0]
        if len(peak_indexes) > 0:
            mz_cent = intensity_weighted_centroid(peaks_a[peak_indexes,1], peaks_a[peak_indexes,0])
            summed_intensity = peaks_a[peak_indexes,1].sum()
            peaks_l.append((mz_cent, summed_intensity))
            # remove the raw points assigned to this peak
            peaks_a = np.delete(peaks_a, peak_indexes, axis=0)
    return np.array(peaks_l)

# resolve the fragment ions for this feature
# returns a decharged peak list (neutral mass+proton mass, intensity)
def resolve_fragment_ions(feature_charge, ms2_points_df):
    # perform intensity descent to resolve peaks
    raw_points_a = ms2_points_df[['mz','intensity']].to_numpy()
    peaks_a = intensity_descent(peaks_a=raw_points_a, peak_delta=None)
    
    # deconvolution
    # for details on deconvolute_peaks see https://mobiusklein.github.io/ms_deisotope/docs/_build/html/deconvolution/deconvolution.html
    # returns a list of DeconvolutedPeak - see https://github.com/mobiusklein/ms_deisotope/blob/bce522a949579a5f54465eab24194eb5693f40ef/ms_deisotope/peak_set.py#L78
    peaks_l = list(map(tuple, peaks_a))
    maximum_neutral_mass = 1700*feature_charge  # give the deconvolution a reasonable upper limit to search within
    deconvoluted_peaks, _ = deconvolute_peaks(peaks_l, use_quick_charge=True, averagine=averagine.peptide, scorer=scoring.PenalizedMSDeconVFitter(minimum_score=20., penalty_factor=3.0), truncate_after=0.95, ignore_below=0.0, charge_range=(1,feature_charge), retention_strategy=peak_retention_strategy.TopNRetentionStrategy(n_peaks=100, base_peak_coefficient=1e-6, max_mass=maximum_neutral_mass))
    
    # package the spectra as a dataframe
    deconvoluted_peaks_l = []
    for peak in deconvoluted_peaks:
        d = {}
        d['singly_protonated_mass'] = round(peak.neutral_mass+PROTON_MASS, 4)
        d['neutral_mass'] = round(peak.neutral_mass, 4)
        d['intensity'] = peak.intensity
        deconvoluted_peaks_l.append(d)
    deconvoluted_peaks_df = pd.DataFrame(deconvoluted_peaks_l)
    
    # sort and normalise intensity
    deconvoluted_peaks_df.sort_values(by=['intensity'], ascending=False, inplace=True)
    deconvoluted_peaks_df.intensity = deconvoluted_peaks_df.intensity / deconvoluted_peaks_df.intensity.max() * 1000.0
    deconvoluted_peaks_df.intensity = deconvoluted_peaks_df.intensity.astype(np.uint)
    deconvoluted_peaks_df = deconvoluted_peaks_df[(deconvoluted_peaks_df.intensity > 0)]
    
    return deconvoluted_peaks_df.head(n=20)

@ray.remote
def process_cuboid_features(cuboid):
    features_subset_df = cuboid['features_subset_df']

    # add the precursor identifier
    features_subset_df['precursor_cuboid_id'] = cuboid['precursor_cuboid_id']

    # resolve the fragment ions for this feature's charge
    features_subset_df['fragment_ions_l'] = features_subset_df.apply(lambda row: json.dumps(resolve_fragment_ions(row.charge, cuboid['ms2_points_df']).to_dict(orient='records')), axis=1)

    return features_subset_df

# determine the number of workers based on the number of available cores and the proportion of the machine to be used
def number_of_workers():
    number_of_cores = mp.cpu_count()
    number_of_workers = int(proportion_of_cores_to_use * number_of_cores)
    return number_of_workers


############################

# set up Ray
print("setting up Ray")
if not ray.is_initialized():
    ray.init(num_cpus=number_of_workers())

print('preprocessing the cuboids')
cuboids_l = []
for cuboid in precursor_cuboids_df.itertuples():
    # determine the ms1 extent of the precursor cuboid
    mz_lower = cuboid.wide_mz_lower
    mz_upper = cuboid.wide_mz_upper
    rt_lower = cuboid.wide_ms1_rt_lower
    rt_upper = cuboid.wide_ms1_rt_upper
    scan_lower = cuboid.wide_scan_lower
    scan_upper = cuboid.wide_scan_upper

    # load the ms2 points for this cuboid's fragmentation event
    ms2_points_df = data[
        {
            "frame_indices": slice(int(cuboid.fe_ms2_frame_lower), int(cuboid.fe_ms2_frame_upper+1)),
            "scan_indices": slice(int(cuboid.fe_scan_lower), int(cuboid.fe_scan_upper+1)),
            "precursor_indices": slice(1, None)  # ms2 frames only
        }
    ][['mz_values','scan_indices','frame_indices','rt_values','intensity_values']]
    ms2_points_df.rename(columns={'mz_values':'mz', 'scan_indices':'scan', 'frame_indices':'frame_id', 'rt_values':'retention_time_secs', 'intensity_values':'intensity'}, inplace=True)

    # downcast the data types to minimise the memory used
    int_columns = ['frame_id','scan','intensity']
    ms2_points_df[int_columns] = ms2_points_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
    float_columns = ['retention_time_secs']
    ms2_points_df[float_columns] = ms2_points_df[float_columns].apply(pd.to_numeric, downcast="float")

    # get all the 3did features with an apex within the cuboid's bounds
    features_subset_df = features_df[(features_df.monoisotopic_mz >= mz_lower) & (features_df.monoisotopic_mz <= mz_upper) & (features_df.rt_apex >= rt_lower) & (features_df.rt_apex <= rt_upper) & (features_df.scan_apex >= scan_lower) & (features_df.scan_apex <= scan_upper)].copy()
    
    if len(features_subset_df) > 0:
        cuboids_l.append({'precursor_cuboid_id':cuboid.precursor_cuboid_id, 'features_subset_df':features_subset_df, 'ms2_points_df':ms2_points_df})

print('processing cuboid features')
features_with_fragments_l = ray.get([process_cuboid_features.remote(cuboid=cuboid) for cuboid in cuboids_l])

# join the list of dataframes into a single dataframe
features_within_fragments_df = pd.concat(features_with_fragments_l, axis=0, sort=False, ignore_index=True)

# add the run name
features_within_fragments_df['run_name'] = run_name

# save it back to the de-dup file because that's what the next step expects
print('writing {} features with fragments to {}'.format(len(features_within_fragments_df), features_file))
features_within_fragments_df.to_feather(features_file)

number_features_inside_isol_windows = len(features_within_fragments_df.feature_id.unique())
print('{} unique features inside isolation windows, {}% of features detected'.format(number_features_inside_isol_windows, round(number_features_inside_isol_windows/number_features_detected*100.0,1)))
