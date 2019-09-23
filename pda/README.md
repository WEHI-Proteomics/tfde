#### Requirements
The code was developed on Python 3.6 using a Conda environment on Ubuntu 18.04.

- wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
- bash Anaconda-latest-Linux-x86_64.sh

- conda create --name py36 python=3.6
- conda activate py36


#### Running the Infusini Method pipeline
To process a whole experiment end-to-end, including mass recalibration and re-searching, perform the steps below. In this example, the experiment name is 190719_Hela_Ecoli.

1. python ./otf-peak-detect/pda/bulk-convert-raw-databases.py -raw ./experiments/190719_Hela_Ecoli/raw-databases/ -en 190719_Hela_Ecoli
2. python ./otf-peak-detect/pda/bulk-pasef-process.py -en 190719_Hela_Ecoli
3. python ./otf-peak-detect/pda/bulk-comet-search.py -en 190719_Hela_Ecoli
4. python ./otf-peak-detect/pda/bulk-percolator-id.py -en 190719_Hela_Ecoli
5. python ./otf-peak-detect/pda/adjust-feature-mass.py -en 190719_Hela_Ecoli -rm cluster
6. python ./otf-peak-detect/pda/bulk-comet-search.py -en 190719_Hela_Ecoli -recal
7. python ./otf-peak-detect/pda/bulk-percolator-id.py -en 190719_Hela_Ecoli -recal
