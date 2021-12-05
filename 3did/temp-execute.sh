#!/bin/bash
set -e  # exit on error

# EXPERIMENT_BASE_DIR="/media/big-ssd/experiments"
EXPERIMENT_BASE_DIR="/media/big-ssd/results-P3856_YHE211-3did/minvi-1000-2021-12-05-01-06-49"

echo "render-features-as-mgf (first pass)"
python -u ./tfde/pipeline/render-features-as-mgf.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -pdm 3did

echo ""
echo "search-mgf-against-sequence (first pass)"
python -u ./tfde/pipeline/search-mgf-against-sequence-db.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did

echo ""
echo "identify-searched-features (first pass)"
python -u ./tfde/pipeline/identify-searched-features.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did

echo ""
echo "recalibrate-feature-mass"
python -u ./tfde/pipeline/recalibrate-feature-mass.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211 -pdm 3did

echo ""
echo "render-features-as-mgf (recal)"
python -u ./tfde/pipeline/render-features-as-mgf.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -pdm 3did -recal

echo ""
echo "search-mgf-against-sequence-db (recal)"
python -u ./tfde/pipeline/search-mgf-against-sequence-db.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211 -rn P3856_YHE211_1_Slot1-1_1_5104 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did -recal

echo ""
echo "identify-searched-features (recal)"
python -u ./tfde/pipeline/identify-searched-features.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211 -ff ./tfde/fasta/Human_Yeast_Ecoli.fasta -pdm 3did -recal

echo ""
echo "build-sequence-library"
python -u ./tfde/pipeline/build-sequence-library.py -eb $EXPERIMENT_BASE_DIR -en P3856_YHE211
