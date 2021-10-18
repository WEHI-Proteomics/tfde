# tfde
This is a repository for algorithms about detection and characterisation of peptide features in timsTOF data.

There are a couple of approaches to feature detection here.

#### All-Ion
A DIA pipeline using intensity descent to find features in ms1. [This diagram](https://github.com/WEHI-Proteomics/tfde/blob/master/documentation/pipeline%20schematic.png) explains the steps in the pipeline. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/original-pipeline).

#### YOLO feature detection
Training a YOLO object detection ANN to detect features in ms1 frames. Intended to be used on the instrument at run-time to select precursors to fragment. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/yolo).

#### Also in this repository...
- [Jupyter notebooks](https://github.com/WEHI-Proteomics/tfde/tree/master/notebooks) for prototypying ideas.
- [Data plotting code](https://github.com/WEHI-Proteomics/tfde/tree/master/plotting) from the time before I got the hang of Jupyter notebooks.
- [Experiments](https://github.com/WEHI-Proteomics/tfde/tree/master/experiments), some of which didn't go very far.
