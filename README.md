# tfde
This is a repository for algorithms for targeted detection, extraction, and characterisation of peptide features in timsTOF data.

There are a couple of approaches to feature detection here.

#### PASEF-based
A DDA pipeline that detects peptide features using the instrument isolation windows as a starting point. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/pipeline).

#### YOLO feature detection
Training a YOLO object detection ANN to detect features in ms1 frames. Intended to be used on the instrument at run-time to select precursors to fragment. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/yolo).

#### 3D intensity descent
Using the characteristic structure of peptides to detect and segment features for identification. Code is [here](https://github.com/WEHI-Proteomics/tfde/tree/master/3did).

#### Also in this repository...
- [Jupyter notebooks](https://github.com/WEHI-Proteomics/tfde/tree/master/notebooks) for prototypying ideas.
- [Experiments](https://github.com/WEHI-Proteomics/tfde/tree/master/experiments), some of which helped develop ideas, while others didn't go very far.
