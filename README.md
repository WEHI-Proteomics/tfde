# TFD/E
This is a repository for algorithms exploring untargeted and targeted detection, extraction, and characterisation of tryptic peptide features in raw MS1 data produced by the [timsTOF](https://www.bruker.com/en/products-and-solutions/mass-spectrometry/timstof/timstof.html) mass spectrometer for LC-MS/MS proteomics experiments.

## Peptide feature detectors
There are three approaches to peptide feature detection for the timsTOF in this repository.

#### Targeted Feature Detection and Extraction Pipeline
A DDA analysis pipeline where the first phase processes one or more runs at a time and detects peptide features using the instrument isolation windows as a starting point (targeted feature detection). It builds a library of peptides identified in at least one run. The second phase uses the peptide library to build machine learning models that predict the 3D coordinates for each peptide in the library. It then extracts them and decoys to control the FDR (targeted extraction). Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/pipeline).

#### 3D Intensity Descent (3DID)
3DID is a *de novo* MS1 feature detector that uses the characteristic structure of peptides in 4D to detect and segment features for identification. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/3did).

#### Also in this repository...
- [Jupyter notebooks](https://github.com/WEHI-Proteomics/tfde/tree/master/notebooks/papers) for generating the figures for the papers and some other visualisations.

## Installation

#### Data requirements
The code has been tested with the runs deposited to the ProteomeXchange Consortium via the PRIDE partner repository with the dataset identifier PXD030706 and 10.6019/PXD030706. Other timsTOF raw data should work.

#### Hardware requirements
The code has been tested on a PC with a 12-core Intel i7 6850K processor and 64 GB of memory running Ubuntu 20.04. It will run faster with more cores and more memory that will allow it to increase the parallelism. The pipeline will automatically detect the hardware environment and utilise resources up the specified `proportion_of_cores_to_use` value.

#### Conda
1. Follow the installation instructions [here](https://www.anaconda.com/products/distribution).
2. Create a Python 3.8 environment with `conda create -n [name] python=3.8`
3. Activate the environment with `conda activate [name]`

#### TensorFlow
Follow the installation instructions [here](https://www.tensorflow.org/install).

#### TFD/E
1. Clone the repository with `git clone git@github.com:WEHI-Proteomics/tfde.git`.
2. Install the required packages with `pip install -r ./tfde/requirements.txt`.

## Usage
1. Create a directory for the group of experiments. For example, `/media/big-ssd/experiments`. This is called the experiment base directory. All the intermediate artefacts and results produced by the pipeline will be stored in subdirectories created automatically under this directory.  
2. Under the experiment base directory, create a directory for each experiment. For example, `P3856_YHE010` for the human-only data.  
3. The pipeline expects the raw `.d` directories to be in a directory called `raw-databases` under the experiment directoy. Either copy the `.d` directories here, or save storage by creating symlinks to them. For example, the .d directories have been downloaded to `/media/timstof-output`, the symlinks can be created like this:  
    1. `cd /media/big-ssd/experiments/P3856_YHE010/raw-databases`
    2. `ln -s /media/timstof-output/* .`
4. Edit the `./tfde/pipeline/bulk-run.sh` bash script to process the groups of technical replicates of the experiment. These are the runs that will be used to build the peptide library and from which the library peptides will be extracted. Be sure to specify the experiment base directory with the `-eb` flag, which has the value `/media/big-ssd/experiments` by default.  
5. Execute the pipeline with `./tfde/pipeline/bulk-run.sh`. Progress information is printed to stdout. Analysis will take a number of hours, depending on the complexity of the samples, the number of runs in the experiment, the length of the LC gradient, and the computing resources of the machine. It's convenient to use a command like this for long-running processes: `nohup ./tfde/pipeline/bulk-run.sh > tfde.log 2>&1 &`.
6. The results are stored in a SQLite database called `results.sqlite` in the `summarised-results` directory. This database includes the peptides identified and extracted, the runs from which they were identified and extracted, and the proteins inferred. Examples of how to extract data from the results schema are in the `notebooks` directory.
