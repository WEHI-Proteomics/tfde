import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
import os
import shutil
import argparse
import sys
import alphatims.bruker


###################################
parser = argparse.ArgumentParser(description='Render a subset of raw data as a 2D spectra.')
parser.add_argument('-eb','--experiment_base_dir', type=str, default='./experiments', help='Path to the experiments directory.', required=False)
parser.add_argument('-en','--experiment_name', type=str, help='Name of the experiment.', required=True)
parser.add_argument('-rn','--run_name', type=str, help='Name of the run.', required=True)
parser.add_argument('-rl','--rt_lower', type=int, default='1650', help='Lower limit for retention time.', required=False)
parser.add_argument('-ru','--rt_upper', type=int, default='2200', help='Upper limit for retention time.', required=False)
parser.add_argument('-sl','--scan_lower', type=int, default='500', help='Lower limit for scan.', required=False)
parser.add_argument('-su','--scan_upper', type=int, default='800', help='Upper limit for scan.', required=False)
parser.add_argument('-ml','--mz_lower', type=int, default='695', help='Lower limit for m/z.', required=False)
parser.add_argument('-mu','--mz_upper', type=int, default='740', help='Upper limit for m/z.', required=False)
parser.add_argument('-od','--output_dir', type=str, help='Base directory for the tiles output.', required=True)
args = parser.parse_args()

# Print the arguments for the log
info = []
for arg in vars(args):
    info.append((arg, getattr(args, arg)))
print(info)

frame_counter = 0

# check the experiment directory exists
EXPERIMENT_DIR = "{}/{}".format(args.experiment_base_dir, args.experiment_name)
if not os.path.exists(EXPERIMENT_DIR):
    print("The experiment directory is required but doesn't exist: {}".format(EXPERIMENT_DIR))
    sys.exit(1)

# check the raw database
RAW_DATABASE_BASE_DIR = "{}/raw-databases".format(EXPERIMENT_DIR)
RAW_DATABASE_NAME = "{}/{}.d".format(RAW_DATABASE_BASE_DIR, args.run_name)
if not os.path.exists(RAW_DATABASE_NAME):
    print("The raw database is required but doesn't exist: {}".format(RAW_DATABASE_NAME))
    sys.exit(1)

# create the TimsTOF object
RAW_HDF_FILE = '{}.hdf'.format(args.run_name)
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

# set up the output directory
working_folder = args.output_dir
if os.path.exists(working_folder):
    shutil.rmtree(working_folder)
os.makedirs(working_folder)

print('loading the raw points')
# load the ms1 points for this cuboid
ms1_df = data[
    {
        "rt_values": slice(float(args.rt_lower), float(args.rt_upper)),
        "mz_values": slice(float(args.mz_lower), float(args.mz_upper)),
        "scan_indices": slice(int(args.scan_lower), int(args.scan_upper+1)),
        "precursor_indices": 0,  # ms1 frames only
    }
][['mz_values','scan_indices','frame_indices','rt_values','intensity_values']]
ms1_df.rename(columns={'mz_values':'mz', 'scan_indices':'scan', 'frame_indices':'frame_id', 'rt_values':'retention_time_secs', 'intensity_values':'intensity'}, inplace=True)
# downcast the data types to minimise the memory used
int_columns = ['frame_id','scan','intensity']
ms1_df[int_columns] = ms1_df[int_columns].apply(pd.to_numeric, downcast="unsigned")
float_columns = ['retention_time_secs']
ms1_df[float_columns] = ms1_df[float_columns].apply(pd.to_numeric, downcast="float")

print("loaded {} points".format(len(ms1_df)))

ms1_df['normalised_intensity'] = ms1_df.intensity / ms1_df.intensity.max()
ms1_df.sort_values(by=['frame_id','intensity'], ascending=True, inplace=True)

norm = colors.LogNorm(vmin=ms1_df.intensity.min(), vmax=ms1_df.intensity.max(), clip=True)

intensity_upper = 1.0

for frame_id,frame_df in ms1_df.groupby('frame_id'):
    if len(frame_df) > 0:
        retention_time_secs = frame_df.iloc[0].retention_time_secs

        print("rendering frame {}".format(frame_counter))

        f, ax = plt.subplots()
        f.set_facecolor('whitesmoke')
        plt.scatter(frame_df.mz, frame_df.normalised_intensity, s=5**2, c=norm(frame_df.intensity), cmap=plt.get_cmap('turbo'), alpha=1.0, edgecolors='face')
        plt.xlabel('m/z', fontsize=20)
        plt.ylabel('normalised intensity', fontsize=20)
        plt.tick_params(labelsize=18)
        ax.patch.set_facecolor((0.9, 0.9, 0.9, 0.8))

        plt.xlim((args.mz_lower,args.mz_upper))
        plt.ylim((0,intensity_upper))

        ax.set_yticks([])

        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        f.set_figheight(15)
        f.set_figwidth(15)

        plt.margins(0.06)
        # plt.suptitle('Peaks in the area predicted for sequence {}, charge {}'.format(sequence_name, sequence_charge))

        plt.savefig('{}/img-{:04d}.png'.format(working_folder, frame_counter), bbox_inches='tight', pad_inches=1.0, facecolor=f.get_facecolor())
        plt.close()

        frame_counter += 1
