import pandas as pd
import sys
import glob, os
from pyteomics import mgf
from os.path import expanduser
import time
import argparse


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
parser.add_argument('-mqc','--maxquant_combined_dir', type=str, help='Path to the MaxQuant combined directory.', required=True)
parser.add_argument('-mgf','--mgf_dir', type=str, help='Path to the MGF output directory.', required=True)
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
MGF_DIR = args.mgf_dir

# check the MaxQuant directory
if not os.path.exists(BASE_MAXQUANT_DIR):
    print("The base MaxQuant directory is required but doesn't exist: {}".format(BASE_MAXQUANT_DIR))
    sys.exit(1)

# check the output directory
if not os.path.exists(MGF_DIR):
    os.makedirs(MGF_DIR)

# load the allPeptides table
print('loading the allPeptides table from {}'.format(ALLPEPTIDES_FILENAME))
allpeptides_df = pd.read_csv(ALLPEPTIDES_FILENAME, sep='\t')
allpeptides_df.rename(columns={'Number of isotopic peaks':'isotope_count', 'm/z':'mz', 'Number of data points':'number_data_points', 'Intensity':'intensity', 'Ion mobility index':'scan', 'Ion mobility index length':'scan_length', 'Ion mobility index length (FWHM)':'scan_length_fwhm', 'Retention time':'rt', 'Retention length':'rt_length', 'Retention length (FWHM)':'rt_length_fwhm', 'Charge':'charge_state', 'Number of pasef MS/MS':'number_pasef_ms2_ids', 'Pasef MS/MS IDs':'pasef_msms_ids', 'MS/MS scan number':'msms_scan_number', 'Isotope correlation':'isotope_correlation'}, inplace=True)
# allpeptides_df = allpeptides_df[allpeptides_df.intensity.notnull() & (allpeptides_df.number_pasef_ms2_ids > 0) & (allpeptides_df.msms_scan_number >= 0) & allpeptides_df.pasef_msms_ids.notnull()].copy()
allpeptides_df = allpeptides_df[(allpeptides_df.msms_scan_number >= 0)].copy()
allpeptides_df['rt_in_seconds'] = allpeptides_df.rt * 60.0
allpeptides_df.sort_values(by=['msms_scan_number'], ascending=True, inplace=True)
allpeptides_df.msms_scan_number = allpeptides_df.msms_scan_number.apply(lambda x: int(x))
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
    df_file_name = '{}/{}.pkl'.format(MGF_DIR, group_name)
    file_apl_indexes_df = apl_indexes_df[(apl_indexes_df['raw_file'] == group_name)]
    for idx,row in group_df.iterrows():
        mq_index = row.msms_scan_number
        ms1_d = {'monoisotopic_mass':row.Mass, 
                 'charge':row.charge_state, 
                 'monoisotopic_mz':row.mz, 
                 'intensity':int(row.intensity), 
                 'scan_apex':row.scan, 
                 'rt_apex':row.rt_in_seconds,
                 'raw_file':row['Raw file'],
                 'mq_index':mq_index}
        df = file_apl_indexes_df[(file_apl_indexes_df.mq_index == mq_index)]
        ms2_peaks_df = pd.DataFrame(df.iloc[0].ms2_peaks, columns=['mz','intensity'])
        # construct the visualisation DF
        visualisation_l.append({**ms1_d, 'ms2_peaks':ms2_peaks_df.to_dict('records')})
        # construct the MGF
        feature_spectra = collate_spectra_for_feature(ms1_d, ms2_peaks_df)
        mgf_spectra.append(feature_spectra)
    # generate the MGF for all the features
    print("generating MGF: {}".format(mgf_file_name))
    if os.path.isfile(mgf_file_name):
        os.remove(mgf_file_name)
    f = mgf.write(output=mgf_file_name, spectra=mgf_spectra)
    # save the visualisation DF
    print("saving visualisation DF: {}".format(df_file_name))
    vis_df = pd.DataFrame(visualisation_l)
    vis_df.to_pickle(df_file_name)

stop_run = time.time()
print("total running time ({}): {} seconds".format(parser.prog, round(stop_run-start_run,1)))
