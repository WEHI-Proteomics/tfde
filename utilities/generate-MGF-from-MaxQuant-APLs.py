import pandas as pd
import sys
import glob, os
from pyteomics import mgf
import time
import argparse
import pickle
import json
import numpy as np
import configparser
from configparser import ExtendedInterpolation


def collate_spectra_for_feature(ms1_d, ms2_df):
    # append the monoisotopic and the ms2 fragments to the list for MGF creation
    pairs_df = ms2_df[['mz', 'intensity']].copy().sort_values(by=['mz'], ascending=True)
    spectrum = {}
    spectrum["m/z array"] = pairs_df.mz.values
    spectrum["intensity array"] = pairs_df.intensity.values
    params = {}
    params["TITLE"] = "RawFile: {} Index: {} Charge: {} FeatureIntensity: {} RtApex: {}".format(ms1_d['raw_file'], ms1_d['mq_index'], ms1_d['charge'], ms1_d['intensity'], round(ms1_d['rt_apex'],2))
    params["INSTRUMENT"] = "ESI-QUAD-TOF"
    params["PEPMASS"] = "{} {}".format(round(ms1_d['monoisotopic_mz'],6), ms1_d['intensity'])
    params["CHARGE"] = "{}+".format(ms1_d['charge'])
    params["RTINSECONDS"] = "{}".format(round(ms1_d['rt_apex'],2))
    params["SCANS"] = "{}".format(int(ms1_d['mq_index']))
    spectrum["params"] = params
    return spectrum



###################################
parser = argparse.ArgumentParser(description='Convert the APL files from MaxQuant to an MGF for each run.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-mqc','--maxquant_combined_dir', type=str, help='Path to the MaxQuant combined directory.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Limit the processing to this run.', required=False)
parser.add_argument('-ini','--ini_file', type=str, default='./tfde/pipeline/pasef-process-short-gradient.ini', help='Path to the config file.', required=False)
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

start_run = time.time()


BASE_MAXQUANT_DIR = args.maxquant_combined_dir
MAXQUANT_TXT_DIR = '{}/txt'.format(BASE_MAXQUANT_DIR)
ALLPEPTIDES_FILENAME = '{}/allPeptides.txt'.format(MAXQUANT_TXT_DIR)
APL_DIR = '{}/andromeda'.format(BASE_MAXQUANT_DIR)
MGF_DIR = '{}/{}/mgf-mq'.format(args.experiment_base_dir, args.experiment_name)
FEATURES_DIR = '{}/{}/features-mq'.format(args.experiment_base_dir, args.experiment_name)

# check the INI file exists
if not os.path.isfile(args.ini_file):
    print("The configuration file doesn't exist: {}".format(args.ini_file))
    sys.exit(1)

# load the INI file
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.read(args.ini_file)

CARBON_MASS_DIFFERENCE = cfg.getfloat('common','CARBON_MASS_DIFFERENCE')

# check the MaxQuant directory
if not os.path.exists(BASE_MAXQUANT_DIR):
    print("The base MaxQuant directory is required but doesn't exist: {}".format(BASE_MAXQUANT_DIR))
    sys.exit(1)

# check the output directory
if not os.path.exists(MGF_DIR):
    os.makedirs(MGF_DIR)

# check the features directory
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)

# load the allPeptides table
print('loading the allPeptides table from {}'.format(ALLPEPTIDES_FILENAME))
allpeptides_df = pd.read_csv(ALLPEPTIDES_FILENAME, sep='\t')
allpeptides_df.rename(columns={'Number of isotopic peaks':'isotope_count', 'm/z':'mz', 'Number of data points':'number_data_points', 'Intensity':'intensity', 'Ion mobility index':'scan', 'Ion mobility index length':'scan_length', 'Ion mobility index length (FWHM)':'scan_length_fwhm', 'Retention time':'rt', 'Retention length':'rt_length', 'Retention length (FWHM)':'rt_length_fwhm', 'Charge':'charge_state', 'Number of pasef MS/MS':'number_pasef_ms2_ids', 'Pasef MS/MS IDs':'pasef_msms_ids', 'MS/MS scan number':'msms_scan_number', 'Isotope correlation':'isotope_correlation'}, inplace=True)
allpeptides_df = allpeptides_df[allpeptides_df.intensity.notnull() & (allpeptides_df.number_pasef_ms2_ids > 0) & (allpeptides_df.msms_scan_number >= 0) & allpeptides_df.pasef_msms_ids.notnull()].copy()
allpeptides_df = allpeptides_df[(allpeptides_df.msms_scan_number >= 0)].copy()
allpeptides_df['rt_in_seconds'] = allpeptides_df.rt * 60.0
allpeptides_df['rt_length_secs'] = allpeptides_df.rt_length * 60.0
allpeptides_df['rt_length_fwhm_secs'] = allpeptides_df.rt_length_fwhm * 60.0
allpeptides_df.sort_values(by=['msms_scan_number'], ascending=True, inplace=True)
allpeptides_df.msms_scan_number = allpeptides_df.msms_scan_number.apply(lambda x: int(x))

if args.run_name is not None:
    allpeptides_df = allpeptides_df[(allpeptides_df['Raw file'] == args.run_name)]

print('loaded {} entries from allPeptides'.format(len(allpeptides_df)))

# build a list of indexes from the APL files
print('building the list of indexes from the APL files')
ms2_peaks = []
apl_indexes = []
for file in glob.glob("{}/*.apl".format(APL_DIR)):
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                if line.startswith("header="):
                    line_a = line.split(' ')
                    mq_index = int(line_a[3])
                    raw_file = line_a[1]
                if line[0].isdigit():
                    line_a = line.split('\t')
                    mz = float(line_a[0])
                    intensity = round(float(line_a[1]))
                    ms2_peaks.append((mz, intensity))
                if line.startswith("peaklist end"):
                    apl_indexes.append((mq_index, ms2_peaks.copy(), raw_file))
                    del ms2_peaks[:]
                    mq_index = 0
apl_indexes_df = pd.DataFrame(apl_indexes, columns=['mq_index','ms2_peaks','raw_file'])
apl_indexes_df.sort_values(by=['mq_index'], ascending=True, inplace=True)

# generate the MGFs
print('generating an MGF file for each run')
for group_name,group_df in allpeptides_df.groupby('Raw file'):
    mgf_spectra = []
    visualisation_l = []
    mgf_file_name = '{}/{}.mgf'.format(MGF_DIR, group_name)
    MGF_FILE = '{}/exp-{}-run-{}-features-mq.mgf'.format(MGF_DIR, args.experiment_name, group_name)
    file_apl_indexes_df = apl_indexes_df[(apl_indexes_df['raw_file'] == group_name)]
    for idx,row in group_df.iterrows():
        mq_index = row.msms_scan_number
        # determine the feature envelope
        expected_spacing_mz = CARBON_MASS_DIFFERENCE / row.charge_state
        mz_upper = row.mz + (row.isotope_count * expected_spacing_mz)
        envelope = np.array([(row.mz,0),(mz_upper,0)])
        # put everything together
        ms1_d = {'feature_id':mq_index,
                 'monoisotopic_mass':row.Mass, 
                 'charge':row.charge_state, 
                 'monoisotopic_mz':row.mz, 
                 'intensity':int(row.intensity), 
                 'scan_apex':row.scan, 
                 'scan_lower':row.scan-(row.scan_length/2),
                 'scan_upper':row.scan+(row.scan_length/2),
                 'rt_apex':row.rt_in_seconds,
                 'rt_lower':row.rt_in_seconds-(row.rt_length_secs/2),
                 'rt_upper':row.rt_in_seconds+(row.rt_length_secs/2),
                 'raw_file':row['Raw file'],
                 'envelope':json.dumps([tuple(e) for e in envelope]),
                 'isotope_count':row.isotope_count,
                 'mq_index':mq_index}
        df = file_apl_indexes_df[(file_apl_indexes_df.mq_index == mq_index)]
        ms2_peaks_df = pd.DataFrame(df.iloc[0].ms2_peaks, columns=['mz','intensity'])
        # construct the visualisation DF
        visualisation_l.append({**ms1_d, 'ms2_peaks':ms2_peaks_df.to_dict('records')})
        # construct the MGF
        feature_spectra = collate_spectra_for_feature(ms1_d, ms2_peaks_df)
        mgf_spectra.append(feature_spectra)
    # generate the MGF for all the features
    print("generating MGF: {}".format(MGF_FILE))
    if os.path.isfile(MGF_FILE):
        os.remove(MGF_FILE)
    f = mgf.write(output=MGF_FILE, spectra=mgf_spectra)

    # save the visualisation DF
    features_file = '{}/exp-{}-run-{}-features-mq.feather'.format(FEATURES_DIR, args.experiment_name, group_name)
    print("saving {} features to {}".format(len(visualisation_l), features_file))
    features_df = pd.DataFrame(visualisation_l)
    features_df['run_name'] = group_name
    features_df.reset_index(drop=True).to_feather(features_file)

    # write the metadata
    info.append(('total_running_time',round(time.time()-start_run,1)))
    info.append(('processor',parser.prog))
    info.append(('processed', time.ctime()))
    FEATURES_METADATA_FILE = '{}/exp-{}-run-{}-features-mq.json'.format(FEATURES_DIR, args.experiment_name, group_name)
    with open(FEATURES_METADATA_FILE, 'w') as handle:
        json.dump(info, handle)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
