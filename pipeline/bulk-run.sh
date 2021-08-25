#!/bin/bash

# P3830
#
# cs true, fmdw false
echo "P3830, cs true, fmdw false"
doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.6 cs=true fmdw=false en=P3830 rn=P3830_YeastUPS2_01_Slot1-1_1_5082 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
echo

# cs false, fmdw false
echo "P3830, cs false, fmdw false"
doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.6 cs=false fmdw=false en=P3830 rn=P3830_YeastUPS2_01_Slot1-1_1_5082 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
echo

# cs true, fmdw true
echo "P3830, cs true, fmdw true"
doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.6 cs=true fmdw=true en=P3830 rn=P3830_YeastUPS1_01_Slot1-1_1_5066 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
echo

# cs false, fmdw true
echo "P3830, cs false, fmdw true"
doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.6 cs=false fmdw=true en=P3830 rn=P3830_YeastUPS1_01_Slot1-1_1_5066 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
echo

# P3856
#
# cs true, fmdw false
# echo "P3856, cs true, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=true fmdw=false en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta"
# echo

# cs false, fmdw false
# echo "P3856, cs false, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=false fmdw=false en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta"
# echo

# cs true, fmdw true
# echo "P3856, cs true, fmdw true"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta"
# echo

# cs false, fmdw true
# echo "P3856, cs false, fmdw true"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=false fmdw=true en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta"
# echo
