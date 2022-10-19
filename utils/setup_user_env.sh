#!/bin/bash

export MIMIC_CODE_DIR=$(realpath ../../mimic-code)

export MIMIC_EXTRACT_CODE_DIR=$(realpath ..)
export MIMIC_EXTRACT_OUTPUT_DIR=$MIMIC_EXTRACT_CODE_DIR/data/curated

export DBUSER=mimic
export DBNAME=mimic
export DBPASSWORD=mimic
export SCHEMA=mimiciii
export HOST=SOCKET
export PORT=5432

# Allow users to override any of the above in a local configuration file
if [ -f "setup_user_env_local.sh" ]
then
    . setup_user_env_local.sh
fi

mkdir -p $MIMIC_EXTRACT_OUTPUT_DIR

if [ $HOST = SOCKET ]
then
    export DBSTRING="port=$PORT user=$DBUSER password=$DBPASSWORD dbname=$DBNAME options=--search_path=$SCHEMA"
else
    export DBSTRING="host=$HOST port=$PORT user=$DBUSER password=$DBPASSWORD dbname=$DBNAME options=--search_path=$SCHEMA"
fi
