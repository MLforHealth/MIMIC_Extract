#!/bin/bash
#
# Build curated dataset of .csv / .npy / .hd5 files
# for patient time-series data extracted from PSQL DB
#
# Takes optional argument POP_SIZE
#

mkdir -p $MIMIC_EXTRACT_OUTPUT_DIR;

if [[ -z $POP_SIZE ]]; then
    # means extract all available data
    POP_SIZE=0;
fi

python -u $MIMIC_EXTRACT_CODE_DIR/mimic_direct_extract.py \
    --out_path $MIMIC_EXTRACT_OUTPUT_DIR/ \
    --resource_path $MIMIC_EXTRACT_CODE_DIR/resources/ \
    --extract_pop 2 \
    --extract_outcomes 2 \
    --extract_codes 0 \
    --extract_numerics 2 \
    --extract_notes 0\
    --exit_after_loading 0 \
    --plot_hist 0 \
    --pop_size $POP_SIZE \
    --psql_password $PGPASSWORD \
    --psql_host $HOST \
    --min_percent 0 \
