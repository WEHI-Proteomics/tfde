#!/bin/bash

# P3830
#
# echo "P3830, YeastUPS2, cs true, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=true fmdw=false en=P3830 rn=P3830_YeastUPS2_01_Slot1-1_1_5082 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
# echo

# echo "P3830, YeastUPS2, cs false, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=false fmdw=false en=P3830 rn=P3830_YeastUPS2_01_Slot1-1_1_5082 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
# echo

# echo "P3830, YeastUPS1, cs true, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=true fmdw=false en=P3830 rn=P3830_YeastUPS1_01_Slot1-1_1_5066 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
# echo

# echo "P3830, YeastUPS1, cs false, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3830
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=false fmdw=false en=P3830 rn=P3830_YeastUPS1_01_Slot1-1_1_5066 rl=0 ru=3000 ff="~/otf-peak-detect/fasta/ups1-ups2-yeast.fasta"
# echo

# P3856
#
# echo "P3856, cs true, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=true fmdw=false en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta"
# echo

# echo "P3856, cs false, fmdw false"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=false fmdw=false en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta"
# echo

echo "P3856_YHE211"
doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856_YHE211
doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3856_YHE211 rn=P3856_YHE211_1_Slot1-1_1_5104,P3856_YHE211_2_Slot1-1_1_5105,P3856_YHE211_3_Slot1-1_1_5106,P3856_YHE211_4_Slot1-1_1_5107,P3856_YHE211_5_Slot1-1_1_5108,P3856_YHE211_6_Slot1-1_1_5109,P3856_YHE211_7_Slot1-1_1_5110,P3856_YHE211_8_Slot1-1_1_5111,P3856_YHE211_9_Slot1-1_1_5112,P3856_YHE211_10_Slot1-1_1_5113 rl=0 ru=3000 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta" ini='/home/daryl/otf-peak-detect/pipeline/pasef-process-short-gradient.ini'
echo

# echo "P3856_YHE114"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856_YHE114
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3856_YHE114 rn=P3856_YHE114_1_Slot1-1_1_5115,P3856_YHE114_2_Slot1-1_1_5116,P3856_YHE114_3_Slot1-1_1_5117,P3856_YHE114_4_Slot1-1_1_5118,P3856_YHE114_5_Slot1-1_1_5119,P3856_YHE114_6_Slot1-1_1_5120,P3856_YHE114_7_Slot1-1_1_5121,P3856_YHE114_8_Slot1-1_1_5122,P3856_YHE114_9_Slot1-1_1_5123,P3856_YHE114_10_Slot1-1_1_5124 rl=0 ru=3000 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta" ini='/home/daryl/otf-peak-detect/pipeline/pasef-process-short-gradient.ini'
# echo

# echo "P3856, cs false, fmdw true"
# doit -f ./otf-peak-detect/pipeline/execute-run.py clean en=P3856
# doit -f ./otf-peak-detect/pipeline/execute-run.py pc=0.8 cs=false fmdw=true en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/otf-peak-detect/fasta/Human_Yeast_Ecoli.fasta"
# echo
