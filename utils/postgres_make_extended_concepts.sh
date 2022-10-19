# This file makes tables for the concepts in this subfolder.
# Be sure to run postgres-functions.sql first, as the concepts rely on those function definitions.
# Note that this may take a large amount of time and hard drive space.
#
# Exporting DBCONNEXTRA before calling this script will add this to the
# connection string.  For example, running:
# DBCONNEXTRA="user=mimic password=mimic" bash postgres_make_extended_concepts.sh
# will add these settings to all of the psql calls.  (Note that "dbname"
# and "search_path" do not need to be set.)

# string replacements are necessary for some queries
REGEX_DATETIME_DIFF="s/DATETIME_DIFF\((.+?),\s?(.+?),\s?(DAY|MINUTE|SECOND|HOUR|YEAR)\)/DATETIME_DIFF(\1, \2, '\3')/g"
REGEX_SCHEMA='s/`physionet-data.(mimiciii_clinical|mimiciii_derived|mimiciii_notes).(.+?)`/\2/g'
CONNSTR="dbname=mimic $DBCONNEXTRA"

# this is set as the search_path variable for psql
# a search path of "public,mimiciii" will search both public and mimiciii
# schemas for data, but will create tables on the public schema
PSQL_PREAMBLE='SET search_path TO public,mimiciii'

echo ''
echo '==='
echo 'Beginning to create tables for MIMIC database.'
echo 'Any notices of the form "NOTICE: TABLE "XXXXXX" does not exist" can be ignored.'
echo 'The scripts drop views before creating them, and these notices indicate nothing existed prior to creating the view.'
echo '==='
echo ''

echo 'Directory 5 of 9: fluid_balance'
{ echo "${PSQL_PREAMBLE}; DROP TABLE IF EXISTS colloid_bolus; CREATE TABLE colloid_bolus AS "; cat $MIMIC_CODE_DIR/concepts/fluid_balance/colloid_bolus.sql; } | sed -r -e "${REGEX_DATETIME_DIFF}" | sed -r -e "${REGEX_SCHEMA}" | psql "${CONNSTR}"
{ echo "${PSQL_PREAMBLE}; DROP TABLE IF EXISTS crystalloid_bolus; CREATE TABLE crystalloid_bolus AS "; cat $MIMIC_CODE_DIR/concepts/fluid_balance/crystalloid_bolus.sql; } | sed -r -e "${REGEX_DATETIME_DIFF}" | sed -r -e "${REGEX_SCHEMA}" | psql "${CONNSTR}"

echo 'Finished creating tables.'
