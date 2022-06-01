#!/bin/bash
set -e

# P3830
#
# echo "P3830, YeastUPS2, cs true, fmdw false"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3830
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=false en=P3830 rn=P3830_YeastUPS2_01_Slot1-1_1_5082 rl=0 ru=3000 ff="~/tfde/fasta/ups1-ups2-yeast.fasta"
# echo

# echo "P3830, YeastUPS2, cs false, fmdw false"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3830
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=false fmdw=false en=P3830 rn=P3830_YeastUPS2_01_Slot1-1_1_5082 rl=0 ru=3000 ff="~/tfde/fasta/ups1-ups2-yeast.fasta"
# echo

# echo "P3830, YeastUPS1, cs true, fmdw false"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3830
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=false en=P3830 rn=P3830_YeastUPS1_01_Slot1-1_1_5066 rl=0 ru=3000 ff="~/tfde/fasta/ups1-ups2-yeast.fasta"
# echo

# echo "P3830, YeastUPS1, cs false, fmdw false"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3830
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=false fmdw=false en=P3830 rn=P3830_YeastUPS1_01_Slot1-1_1_5066 rl=0 ru=3000 ff="~/tfde/fasta/ups1-ups2-yeast.fasta"
# echo

# P3856
#
# echo "P3856, cs true, fmdw false"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3856
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=false en=P3856 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta"
# echo

echo "P3856, cs false, fmdw false"
doit -f ./tfde/pipeline/execute-run.py clean en=P3856_YHE211
doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=false fmdw=false en=P3856_YHE211 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta"
echo

echo "P3856, cs false, fmdw true"
doit -f ./tfde/pipeline/execute-run.py clean en=P3856_YHE211
doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=false fmdw=true en=P3856_YHE211 rn=P3856_YHE211_1_Slot1-1_1_5104 rl=1650 ru=2200 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta"
echo


###############################################################
# technical replicates

# echo "P3856_YHE211"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3856_YHE211
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3856_YHE211 rn=P3856_YHE211_1_Slot1-1_1_5104,P3856_YHE211_2_Slot1-1_1_5105,P3856_YHE211_3_Slot1-1_1_5106,P3856_YHE211_4_Slot1-1_1_5107,P3856_YHE211_5_Slot1-1_1_5108,P3856_YHE211_6_Slot1-1_1_5109,P3856_YHE211_7_Slot1-1_1_5110,P3856_YHE211_8_Slot1-1_1_5111,P3856_YHE211_9_Slot1-1_1_5112,P3856_YHE211_10_Slot1-1_1_5113 rl=0 ru=3000 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta" ini='/home/daryl/tfde/pipeline/pasef-process-short-gradient.ini'
# echo

# echo "P3856_YHE114"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3856_YHE114
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3856_YHE114 rn=P3856_YHE114_1_Slot1-1_1_5115,P3856_YHE114_2_Slot1-1_1_5116,P3856_YHE114_3_Slot1-1_1_5117,P3856_YHE114_4_Slot1-1_1_5118,P3856_YHE114_5_Slot1-1_1_5119,P3856_YHE114_6_Slot1-1_1_5120,P3856_YHE114_7_Slot1-1_1_5121,P3856_YHE114_8_Slot1-1_1_5122,P3856_YHE114_9_Slot1-1_1_5123,P3856_YHE114_10_Slot1-1_1_5124 rl=0 ru=3000 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta" ini='/home/daryl/tfde/pipeline/pasef-process-short-gradient.ini'
# echo

# echo "P3856_YHE010"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3856_YHE010
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3856_YHE010 rn=P3856_YHE010_10_Slot1-1_1_5102,P3856_YHE010_2_Slot1-1_1_5094,P3856_YHE010_4_Slot1-1_1_5096,P3856_YHE010_6_Slot1-1_1_5098,P3856_YHE010_8_Slot1-1_1_5100,P3856_YHE010_1_Slot1-1_1_5093,P3856_YHE010_3_Slot1-1_1_5095,P3856_YHE010_5_Slot1-1_1_5097,P3856_YHE010_7_Slot1-1_1_5099,P3856_YHE010_9_Slot1-1_1_5101 rl=0 ru=3000 ff="/home/daryl/tfde/fasta/Human_Yeast_Ecoli.fasta" ini='/home/daryl/tfde/pipeline/pasef-process-short-gradient.ini'
# echo

# echo "P3830_YUPS1"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3830_YUPS1
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3830_YUPS1 rn=P3830_YeastUPS1_01_Slot1-1_1_5066,P3830_YeastUPS1_02_Slot1-1_1_5067,P3830_YeastUPS1_03_Slot1-1_1_5068,P3830_YeastUPS1_04_Slot1-1_1_5069,P3830_YeastUPS1_05_Slot1-1_1_5070,P3830_YeastUPS1_06_Slot1-1_1_5076,P3830_YeastUPS1_07_Slot1-1_1_5077,P3830_YeastUPS1_08_Slot1-1_1_5078,P3830_YeastUPS1_09_Slot1-1_1_5079,P3830_YeastUPS1_10_Slot1-1_1_5080 rl=0 ru=3000 ff="/home/daryl/tfde/fasta/ups1-ups2-yeast.fasta" ini='/home/daryl/tfde/pipeline/pasef-process-short-gradient.ini'
# echo

# echo "P3830_YUPS2"
# doit -f ./tfde/pipeline/execute-run.py clean en=P3830_YUPS2
# doit -f ./tfde/pipeline/execute-run.py pc=0.8 cs=true fmdw=true en=P3830_YUPS2 rn=P3830_YeastUPS2_01_Slot1-1_1_5082,P3830_YeastUPS2_02_Slot1-1_1_5083,P3830_YeastUPS2_03_Slot1-1_1_5084,P3830_YeastUPS2_04_Slot1-1_1_5085,P3830_YeastUPS2_05_Slot1-1_1_5086,P3830_YeastUPS2_06_Slot1-1_1_5087,P3830_YeastUPS2_07_Slot1-1_1_5088,P3830_YeastUPS2_08_Slot1-1_1_5089,P3830_YeastUPS2_09_Slot1-1_1_5090,P3830_YeastUPS2_10_Slot1-1_1_5091 rl=0 ru=3000 ff="/home/daryl/tfde/fasta/ups1-ups2-yeast.fasta" ini='/home/daryl/tfde/pipeline/pasef-process-short-gradient.ini'
# echo
