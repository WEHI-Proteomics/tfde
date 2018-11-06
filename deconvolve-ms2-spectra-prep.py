# -*- coding: utf-8 -*-
import sys
import os.path
import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Prep for generating a text file containing the Hardklor commands.')
parser.add_argument('-dbd','--data_directory', type=str, help='The directory for the processing data.', required=True)
args = parser.parse_args()

# set up the directories for processing
raw_mgf_directory = "{}/raw-mgf".format(args.data_directory)
deconvolved_mgf_directory = "{}/deconvolved-mgf".format(args.data_directory)
search_mgf_directory = "{}/search-mgf".format(args.data_directory)
search_headers_directory = "{}/search-headers".format(args.data_directory)

# clean up the output directories if they already exist
if os.path.exists(raw_mgf_directory):
    shutil.rmtree(raw_mgf_directory)
if os.path.exists(deconvolved_mgf_directory):
    shutil.rmtree(deconvolved_mgf_directory)
if os.path.exists(search_mgf_directory):
    shutil.rmtree(search_mgf_directory)
if os.path.exists(search_headers_directory):
    shutil.rmtree(search_headers_directory)

# create the output directories
os.makedirs(raw_mgf_directory)
os.makedirs(deconvolved_mgf_directory)
os.makedirs(search_mgf_directory)
os.makedirs(search_headers_directory)
