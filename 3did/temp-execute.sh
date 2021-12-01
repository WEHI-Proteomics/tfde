#!/bin/bash
set -e

python -u ./tfde/pipeline/remove-duplicate-features.py -eb /media/big-ssd/experiments -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -pdm 3did
python -u ./tfde/pipeline/render-features-as-mgf.py -eb /media/big-ssd/experiments -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -pdm 3did
python -u ./tfde/pipeline/search-mgf-against-sequence-db.py -eb /media/big-ssd/experiments -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did
python -u ./tfde/pipeline/identify-searched-features.py -eb /media/big-ssd/experiments -en P3856_YHE211 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did
python -u ./tfde/pipeline/recalibrate-feature-mass.py -eb /media/big-ssd/experiments -en P3856_YHE211 -pdm 3did
python -u ./tfde/pipeline/render-features-as-mgf.py -eb /media/big-ssd/experiments -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -pdm 3did -recal
python -u ./tfde/pipeline/search-mgf-against-sequence-db.py -eb /media/big-ssd/experiments -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did -recal
python -u ./tfde/pipeline/identify-searched-features.py -eb /media/big-ssd/experiments -en P3856_YHE211 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did -recal
python -u ./tfde/pipeline/build-sequence-library.py -eb /media/big-ssd/experiments -en P3856_YHE211
