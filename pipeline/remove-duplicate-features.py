import sys
import os.path
import argparse
import time
import pickle
import configparser
from configparser import ExtendedInterpolation


###################################
parser = argparse.ArgumentParser(description='Remove duplicate features.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-pdm','--precursor_definition_method', type=str, choices=['pasef','3did'], help='The method used to define the precursor cuboids.', required=True)
parser.add_argument('-ini','--ini_file', type=str, default='./otf-peak-detect/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

# set up constants
if args.precursor_definition_method == '3did':
    DUP_MZ_TOLERANCE_PPM = cfg.getint('3did', 'DUP_MZ_TOLERANCE_PPM')
    DUP_SCAN_TOLERANCE = cfg.getint('3did', 'DUP_SCAN_TOLERANCE')
    DUP_RT_TOLERANCE = cfg.getint('3did', 'DUP_RT_TOLERANCE')
else:
    DUP_MZ_TOLERANCE_PPM = cfg.getint('ms1', 'DUP_MZ_TOLERANCE_PPM')
    DUP_SCAN_TOLERANCE = cfg.getint('ms1', 'DUP_SCAN_TOLERANCE')
    DUP_RT_TOLERANCE = cfg.getint('ms1', 'DUP_RT_TOLERANCE')

# input features
FEATURES_DIR = "{}/features-{}".format(EXPERIMENT_DIR, args.precursor_definition_method)
FEATURES_FILE = '{}/exp-{}-run-{}-features-{}.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name, args.precursor_definition_method)

# output features
FEATURES_DEDUP_FILE = '{}/exp-{}-run-{}-features-{}-dedup.pkl'.format(FEATURES_DIR, args.experiment_name, args.run_name, args.precursor_definition_method)

# check the features directory
if not os.path.exists(FEATURES_DIR):
    print("The features directory is required but doesn't exist: {}".format(FEATURES_DIR))
    sys.exit(1)

# check the features file
if not os.path.isfile(FEATURES_FILE):
    print("The features file is required but doesn't exist: {}".format(FEATURES_FILE))
    sys.exit(1)

# load the features
with open(FEATURES_FILE, 'rb') as handle:
    features_df = pickle.load(handle)['features_df']
    print('loaded {} features from {}'.format(len(features_df), FEATURES_FILE))

# de-dup the features
if (len(features_df) > 2):
    # set up dup definitions
    MZ_TOLERANCE_PERCENT = DUP_MZ_TOLERANCE_PPM * 10**-4
    features_df['dup_mz'] = features_df['monoisotopic_mz']  # shorthand to reduce verbosity
    features_df['dup_mz_ppm_tolerance'] = features_df.dup_mz * MZ_TOLERANCE_PERCENT / 100
    features_df['dup_mz_lower'] = features_df.dup_mz - features_df.dup_mz_ppm_tolerance
    features_df['dup_mz_upper'] = features_df.dup_mz + features_df.dup_mz_ppm_tolerance
    features_df['dup_scan_lower'] = features_df.scan_apex - DUP_SCAN_TOLERANCE
    features_df['dup_scan_upper'] = features_df.scan_apex + DUP_SCAN_TOLERANCE
    features_df['dup_rt_lower'] = features_df.rt_apex - DUP_RT_TOLERANCE
    features_df['dup_rt_upper'] = features_df.rt_apex + DUP_RT_TOLERANCE

    # remove these after we're finished
    columns_to_drop_l = []
    columns_to_drop_l.append('dup_mz')
    columns_to_drop_l.append('dup_mz_ppm_tolerance')
    columns_to_drop_l.append('dup_mz_lower')
    columns_to_drop_l.append('dup_mz_upper')
    columns_to_drop_l.append('dup_scan_lower')
    columns_to_drop_l.append('dup_scan_upper')
    columns_to_drop_l.append('dup_rt_lower')
    columns_to_drop_l.append('dup_rt_upper')

    # sort by decreasing deconvolution score
    features_df.sort_values(by=['deconvolution_score'], ascending=False, inplace=True)

    # see if any detections have a duplicate - if so, keep the first one
    keep_l = []
    features_processed = set()
    for feature in features_df.itertuples():
        if feature.feature_id not in features_processed:
            dup_df = features_df[(features_df.dup_mz > feature.dup_mz_lower) & (features_df.dup_mz < feature.dup_mz_upper) & (features_df.scan_apex > feature.dup_scan_lower) & (features_df.scan_apex < feature.dup_scan_upper) & (features_df.rt_apex > feature.dup_rt_lower) & (features_df.rt_apex < feature.dup_rt_upper)].copy()
            if len(dup_df) > 1:
                print('{} are duplicates'.format(dup_df.feature_id.tolist()))
            # group the dups by charge
            for group_name,group_df in dup_df.groupby(['charge'], as_index=False):
                keep_l.append(group_df.iloc[0].feature_id)
            # record the features that have been processed
            features_processed.update(set(dup_df.feature_id.tolist()))

    # remove any features that are not in the keep list
    dedup_df = features_df[features_df.feature_id.isin(keep_l)].copy()

    number_of_dups = len(features_df)-len(dedup_df)
    print('removed {} duplicates ({}% of the original detections)'.format(number_of_dups, round(number_of_dups/len(features_df)*100)))

    # remove the columns we added earlier
    dedup_df.drop(columns_to_drop_l, axis=1, inplace=True)
else:
    # nothing to de-dup
    print('nothing to de-dup')
    dedup_df = features_df

# write out all the features
print("writing {} de-duped features to {}".format(len(dedup_df), FEATURES_DEDUP_FILE))
content_d = {'features_df':dedup_df, 'metadata':info}
with open(FEATURES_DEDUP_FILE, 'wb') as handle:
    pickle.dump(content_d, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
