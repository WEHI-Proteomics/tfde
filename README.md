# TFD/E
This is a repository for algorithms exploring targeted detection, extraction, and characterisation of tryptic peptide features in [timsTOF](https://www.bruker.com/en/products-and-solutions/mass-spectrometry/timstof/timstof.html) data.

There are three approaches to feature detection here.

#### Targeted Feature Detection and Extraction Pipeline
A DDA pipeline that detects peptide features using the instrument isolation windows as a starting point. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/pipeline).

#### 3D Intensity Descent (3DID)
Using the characteristic structure of peptides to detect and segment features for identification. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/3did).

#### Feature detection with a YOLO-based CNN detector
Training a YOLO object detection ANN to detect features in ms1 frames. Intended to be used on the instrument at run-time to select precursors to fragment. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/yolo).

#### Also in this repository...
- [Jupyter notebooks](https://github.com/WEHI-Proteomics/tfde/tree/master/notebooks/papers) for generating the figures for the papers.
- [Jupyter notebooks](https://github.com/WEHI-Proteomics/tfde/tree/master/notebooks/prototyping) for prototypying ideas.
- [Experiments](https://github.com/WEHI-Proteomics/tfde/tree/master/experiments), some of which helped develop ideas, while others didn't go very far.
