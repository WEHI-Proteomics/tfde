# -*- coding: utf-8 -*-
import sys
import os.path
import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Prep for generating a text file containing the Hardklor commands.')
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
args = parser.parse_args()

mgf_directory = "{}/mgf".format(args.data_directory)
hk_directory = "{}/hk".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)

# clean up the output directories if they already exist
if os.path.exists(mgf_directory):
    shutil.rmtree(mgf_directory)
if os.path.exists(hk_directory):
    shutil.rmtree(hk_directory)
if os.path.exists(search_headers_directory):
    shutil.rmtree(search_headers_directory)

# create the output directories
os.makedirs(mgf_directory)
os.makedirs(hk_directory)
os.makedirs(search_headers_directory)
