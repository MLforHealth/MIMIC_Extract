#!/bin/bash

export MIMIC_CODE_DIR=$(realpath ../../mimic-code)
export MIMIC_EXTRACT_CODE_DIR=$(realpath ../)

export MIMIC_DATA_DIR=$MIMIC_EXTRACT_CODE_DIR/data/

export MIMIC_EXTRACT_OUTPUT_DIR=$MIMIC_DATA_DIR/curated/
mkdir -p $MIMIC_EXTRACT_OUTPUT_DIR

export DBUSER=bnestor
export DBNAME=mimic
export SCHEMA=mimiciii
export HOST=mimic
export DBSTRING="dbname=$DBNAME options=--search_path=$SCHEMA"
alias psql="psql -h $HOST -U $DBUSER "

export PGHOST=$HOST
export PGUSER=$DBUSER

export PGPASSWORD=$1
