import glob
import shutil
import os

base_dir = './190719_Hela_Ecoli'
target_mgf_dir = '{}/mgfs'.format(base_dir)
target_pkl_dir = '{}/pkls'.format(base_dir)

if os.path.exists(target_mgf_dir):
    shutil.rmtree(target_mgf_dir)
os.makedirs(target_mgf_dir)

if os.path.exists(target_pkl_dir):
    shutil.rmtree(target_pkl_dir)
os.makedirs(target_pkl_dir)

for file in glob.glob('{}/**/*.mgf'.format(base_dir), recursive=True):
    target_file = '{}/{}'.format(target_mgf_dir, os.path.basename(file))
    shutil.copy(file, target_file)

for file in glob.glob('{}/**/*.pkl'.format(base_dir), recursive=True):
    target_file = '{}/{}'.format(target_pkl_dir, os.path.basename(file))
    shutil.copy(file, target_file)
