##!/bin/bash

module use /stornext/System/data/modulefiles/sysbio/
module load anaconda2/4.2.0

python $@
