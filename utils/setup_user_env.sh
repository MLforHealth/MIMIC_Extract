#!/bin/bash

export MIMIC_CODE_DIR=$(realpath ../../mimic-code)
export MIMIC_EXTRACT_CODE_DIR=$(realpath ../)

export MIMIC_DATA_DIR=$MIMIC_EXTRACT_CODE_DIR/data/

export MIMIC_EXTRACT_OUTPUT_DIR=$MIMIC_DATA_DIR/curated/
mkdir -p $MIMIC_EXTRACT_OUTPUT_DIR

export DBUSER=mimic
export DBNAME=mimic
export SCHEMA=mimiciii
export HOST=localhost
export PORT=5432
export PGPASSWORD=mimic

export DBSTRING="host=$HOST port=$PORT user=$DBUSER password=$DBPASSWORD dbname=$DBNAME options=--search_path=$SCHEMA"

export PGHOST=$HOST
export PGPORT=$PORT
export PGUSER=$DBUSER

