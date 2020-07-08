#!/bin/bash

export MIMIC_CODE_DIR=../../mimic-code
export MIMIC_EXTRACT_CODE_DIR=../

export MIMIC_DATA_DIR=$MIMIC_EXTRACT_CODE_DIR/data/

export MIMIC_EXTRACT_OUTPUT_DIR=$MIMIC_DATA_DIR/curated/
mkdir -p $MIMIC_EXTRACT_OUTPUT_DIR

export DBUSER=bnestor
export DBNAME=mimic
export SCHEMA=mimiciii
export HOST=mimic
export DBSTRING="dbname=$DBNAME options=--search_path=$SCHEMA"

