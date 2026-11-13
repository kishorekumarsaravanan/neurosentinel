#!/usr/bin/env bash
# Downloads the CHB-MIT recordings used by NeuroSentinel into data/raw/<patient>/
# Usage: bash scripts/download_data.sh
set -e
BASE=https://physionet.org/files/chbmit/1.0.0
cd "$(dirname "$0")/.."
FILES="
chb01/chb01_01 chb01/chb01_02 chb01/chb01_03 chb01/chb01_04 chb01/chb01_05 chb01/chb01_06
chb01/chb01_15 chb01/chb01_16 chb01/chb01_18 chb01/chb01_21 chb01/chb01_26
chb03/chb03_01 chb03/chb03_02 chb03/chb03_03 chb03/chb03_04 chb03/chb03_05 chb03/chb03_06
chb03/chb03_07 chb03/chb03_34 chb03/chb03_35 chb03/chb03_36
"
for p in chb01 chb03; do mkdir -p data/raw/$p; curl -sL -o data/raw/$p/$p-summary.txt $BASE/$p/$p-summary.txt; done
echo $FILES | tr ' ' '\n' | xargs -P 4 -I{} sh -c 'curl -sL --retry 3 -C - -o data/raw/{}.edf '"$BASE"'/{}.edf && echo "done {}"'
echo ALL_DONE
