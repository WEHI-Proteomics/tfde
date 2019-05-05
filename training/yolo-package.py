import argparse
import os

parser = argparse.ArgumentParser(description='Package a YOLO training set.')
parser.add_argument('-tb','--tile_base', type=str, help='Path to the base directory of the training set.', required=True)
args = parser.parse_args()

# get the base name of the package
basename = os.path.basename(args.tile_base)
package_filename = "{}-package.7z".format(basename)

# remove the package file if it already exists
if os.path.isfile(package_filename):
    os.remove(package_filename)

# training set
cmd = "7z a {} {}/train".format(package_filename, args.tile_base)
os.system(cmd)

# validation set
cmd = "7z a {} {}/validation".format(package_filename, args.tile_base)
os.system(cmd)

# test set
cmd = "7z a {} {}/test".format(package_filename, args.tile_base)
os.system(cmd)

# data files
cmd = "7z a {} {}/data-files".format(package_filename, args.tile_base)
os.system(cmd)

# dataset files
cmd = "7z a {} {}/peptides-obj.*".format(package_filename, args.tile_base)
os.system(cmd)
