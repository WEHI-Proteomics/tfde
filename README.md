# otf-peak-detect
This is the repository for Daryl's PhD project about on-the-fly timsTOF peak detection.

There are a few approaches to feature detection here.

### All-Ion
A DIA pipeline using intensity descent to find features in ms1. [This diagram](https://github.com/WEHI-Proteomics/otf-peak-detect/blob/master/pipeline%20schematic.png) explains the steps in the pipeline. Code is [here](https://github.com/WEHI-Proteomics/otf-peak-detect/tree/master/original-pipeline).

### Infusini Method
A DDA pipeline using PASEF's isolation windows to detect features in ms1, and using the fragments of the precursors selected by PASEF to associate ms2 spectra with those features. Code is [here](https://github.com/WEHI-Proteomics/otf-peak-detect/tree/master/pda).

### YOLO feature detection
Training a YOLO object detection ANN to detect features in ms1 frames. Intended to be used on the instrument at run-time to select precursors to fragment. Code is [here](https://github.com/WEHI-Proteomics/otf-peak-detect/tree/master/yolo).

Also in the repository
- [Jupyter notebooks](https://github.com/WEHI-Proteomics/otf-peak-detect/tree/master/notebooks) for prototypying ideas.
- [Data plotting code](https://github.com/WEHI-Proteomics/otf-peak-detect/tree/master/plotting) from the time before I got the hang of Jupyter notebooks.
- [Experiments](https://github.com/WEHI-Proteomics/otf-peak-detect/tree/master/experiments) that didn't go very far.
