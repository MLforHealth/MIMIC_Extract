# **MIMIC-Extract**:A Data Extraction, Preprocessing, and Representation Pipeline for MIMIC-III

# About
This repo contains code for **MIMIC-Extract**. It has been divided into the following folders:
* Data: Locally contains the data to be extracted.
* Notebooks: Jupyter Notebooks demonstrating test cases and usage of output data in risk and intervention prediction tasks.
* Resources: Consist of Rohit_itemid.txt which describes the correlation of MIMIC-III item ids with those of MIMIC II as used by Rohit; itemid_to_variable_map.csv which is the main file used in data extraction - consists of groupings of item ids as well as which item ids are ready to extract; variable_ranges.csv which describes the normal variable ranges for the levels assisting in extraction of proper data. It also contains expected schema of output tables.
* Utils: scripts and detailed instructions for running **MIMIC-Extract** data pipeline.
* `mimic_direct_extract.py`: extraction script. 

# Paper
If you use this code in your research, please cite the following publication:

```
Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Michael C. Hughes, Tristan Naumann, 
and Marzyeh Ghassemi. MIMIC-Extract: A Data Extraction, Preprocessing, and Representation 
Pipeline for MIMIC-III. arXiv:1907.08322. 
```

# Pre-processed Output
If you simply wish to use the output of this pipeline in your own research, a preprocessed version with
default parameters is available via gcp,
[here](https://console.cloud.google.com/storage/browser/mimic_extract).

To access this, you will need to be credentialed for MIMIC-III GCP access through physionet. Instructions for
that are available [on physionet](https://mimic.physionet.org/gettingstarted/cloud/).

This output is released on an as-is basis, with no guarantees, but if you find any issues with it please let
us know via Github issues.

# Step-by-step instructions
The first several steps are the same here as above. These instructions are tested with mimic-code at version
762943eab64deb30bdb2abcf7db43602ccb25908

## Step 0: Required software and prereqs

Your local system should have the following executables on the PATH:

* conda
* psql (PostgreSQL 9.4 or higher)
* git
* MIMIC-iii psql relational database (Refer to [MIT-LCP Repo](https://github.com/MIT-LCP/mimic-code))

All instructions below should be executed from a terminal, with current directory set to utils/

## Step 1: Create conda environment

Next, make a new conda environment from [mimic_extract_env_py36.yml](../mimic_extract_env_py36.yml) and
activate that environment.

```
conda env create --force -f ../mimic_extract_env_py36.yml
```

This step will _report failure on the pip installation stage_. This is not the end of the world. Instead,
simply activate the environment (which should work despite the former "failure"):

```
conda activate mimic_data_extraction
```

And then install any failed packages with pip (e.g., `pip install [package]`). This may include, in
particular, packages: `datapackage`, `spacy`, and `scispacy`.
You will also then need to install the english language model for spacy, via:
`python -m spacy download en_core_web_sm`

#### Expected Outcome

The desired enviroment will be created and activated.

#### Expected Resources

Will typically take less than 5 minutes.
Requires a good internet connection.

## Step 3: Build Views for Feature Extraction

Materialized views in the MIMIC PostgreSQL database will be generated.
This includes all concept tables in [MIT-LCP Repo](https://github.com/MIT-LCP/mimic-code) and tables for
extracting non-mechanical ventilation, and injections of crystalloid bolus and colloid bolus.

Note that you need to have schema edit permission on your postgres user to make concepts in this way. First,
you must clone this github repository to a directory, which here we assume is stored in the environment
variable `$MIMIC_CODE_DIR`. After cloning, follow these instructions:

```
cd $MIMIC_CODE_DIR/concepts
psql -d mimic -f postgres-functions.sql
bash postgres_make_concepts.sh
```

Next, you'll need to build 3 additional materialized views necessary for this pipeline. To do this (again with
schema edit permission), navigate to `utils` and run `bash postgres_make_extended_concepts.sh` followed by
`psql -d mimic -f niv-durations.sql`.  (You can add extra `psql`
connection parameters; see the start of
`postgres_make_extended_concepts.sh` for details.)

## Step 4: Set Cohort Selection and Extraction Criteria

Next, navigate to the root directory of _this repository_, activate your conda environment and run
`python mimic_direct_extract.py ...` with your args as desired.

#### Expected Outcome

The default setting will create an hdf5 file inside MIMIC_EXTRACT_OUTPUT_DIR with four tables:
* **patients**: static demographics, static outcomes
  * One row per (subj_id,hadm_id,icustay_id)

* **vitals_labs**: time-varying vitals and labs (hourly mean, count and standard deviation)
  * One row per (subj_id,hadm_id,icustay_id,hours_in)

* **vitals_labs_mean**: time-varying vitals and labs (hourly mean only)
  * One row per (subj_id,hadm_id,icustay_id,hours_in)

* **interventions**: hourly binary indicators for administered interventions
  * One row per (subj_id,hadm_id,icustay_id,hours_in)


#### Expected Resources

Will probably take 5-10 hours.
Will require a good machine with at least 50GB RAM.

#### Setting the population size

By default, this step builds a dataset with all eligible patients. Sometimes, we wish to run with only a small subset of patients (debugging, etc.).

To do this, just set the POP_SIZE environmental variable. For example, to build a curated dataset with only the first 1000 patients, we could do:


# Common Errors / FAQ:
  1. When running `mimic_direct_extract.py`, I encounter an error of the form: 
     ```
        psycopg2.OperationalError: could not connect to server: No such file or directory
        Is the server running locally and accepting
        connections on Unix domain socket "/tmp/.s.PGSQL.5432"?
     ```
     or
     ```
        psycopg2.OperationalError: could not connect to server: No such file or directory
        Is the server running locally and accepting
        connections on Unix domain socket "/var/run/postgresql/..."?
     ```
     For this issue, see [this stackoverflow
     post](https://stackoverflow.com/questions/5500332/cant-connect-the-postgresql-with-psycopg2) and use our
     `--psql_host` argument, which you can pass either directly when calling `mimic_direct_extract.py` or use
     via the Makefile instructions by setting the `HOST` environment variable.
  2. `relation "code_status" does not exist`
     In this error, the table `code_status` hasn't been built successfully, and you'll need to rebuild your
     MIMIC-III concepts. Instructions for this can be found in Step 3 of either instruction set. Also see
     below for our issues specific to building concepts.

## Common Errors with Building Concepts
  1. When I built concepts, the system complained it didn't have permissions to edit schema mimiciii. This
     error indicates that your default psql user doesn't have authority to build concepts. You need to login
     as a higher authority postgres user to and have it run the commands. This is common in setups where
     multiple users have read-only access to postgres at once. If you do this, you may need to take extra
     steps to expose the resulting concepts tables to other users.
  2. I built concepts, but now the code can't see them. This can be for a few reasons - firslty, you may not
     have permissions to read the new tables, and secondly, they may be in the wrong namespace. Our code
     expects them to be fully visible and within the mimiciii namespace. To adjust these properties, login as
     the owning postgres user and adjust the permissions and namespaces of those views manually. A few
     commands that are relevant are:
    * `ALTER TABLE code_status SET SCHEMA mimiciii;`
    * `GRANT SELECT ON mimiciii.code_status TO [USER];`
    Note that you'll need to run these on _every_ concepts table accessed by the script.
