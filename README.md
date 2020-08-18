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

# Step-by-step Instructions

Step-by-step instructions using the system Makefile can be found below, but if these don't work for your
system, there are less abstracted, more direct instructoins below.

* [Step 0: Required software and prereqs](#step-0-required-software-and-prereqs)
* [Step 1: Setup env vars for local system](#step-1-setup-env-vars-for-current-local-system)
* [Step 2: Create conda environment](#step-2-create-conda-environment)
* [Step 3: Build Views for Feature Extraction](#step-3-build-views-for-feature-extraction)
* [Step 4: Set Cohort Selection and Extraction Criteria](#step-4-set-cohort-selection-and-extraction-criteria)
* [Step 5: Build Curated Dataset from PostgreSQL](#step-5-build-curated-dataset-from-psql)


## Step 0: Required software and prereqs

Your local system should have the following executables on the PATH:

* conda
* psql (PostgreSQL 9.4 or higher)
* git
* MIMIC-iii psql relational database (Refer to [MIT-LCP Repo](https://github.com/MIT-LCP/mimic-code))

All instructions below should be executed from a terminal, with current directory set to utils/

```
cd utils/
```

## Step 1: Setup env vars for current local system

Edit [setup_user_env.sh](./utils/setup_user_env.sh) so all paths point to valid locations on local file system and export those variables.

```
source ./setup_user_env.sh {your psql password}
```

## Step 2: Create conda environment

Next, make a new conda environment from [mimic_extract_env_py36.yml](../mimic_extract_env_py36.yml) and
activate that environment.

```
conda env create --force -f ../mimic_extract_env_py36.yml
conda activate mimic_data_extraction
```

#### Expected Outcome

The desired enviroment will be created and activated.

#### Expected Resources

Will typically take less than 5 minutes.
Requires a good internet connection.

## Step 3: Build Views for Feature Extraction

Materialized views in the MIMIC PostgreSQL database will be generated.
This includes all concept tables in [MIT-LCP Repo](https://github.com/MIT-LCP/mimic-code) and tables for
extracting non-mechanical ventilation, and injections of crystalloid bolus and colloid bolus. Note that you
need to have schema edit permission on your postgres user to make concepts in this way.

```
cd $MIMIC_CODE_DIR/concepts
psql -d mimic -f postgres-functions.sql
bash postgres_make_concepts.sh
```

## Step 4: Set Cohort Selection and Extraction Criteria

```
cd $MIMIC_EXTRACT_CODE_DIR
cd utils
```
Parameters for the extraction code are specified in `build_curated_from_psql.sh`.
Cohort selection criteria regarding minimum admission age is set through `min_age`; minimum and maximum 
length of ICU stay in hours are set through `min_duration` and `max_duration`.
Only vitals and labs that contain over `min_percent` percent non-missingness are extracted and extracted vitals and labs are
clinically aggregated unless `group_by_level2` is explicitly set. Outlier correction is applied unless `var_limit` is set to 0.

## Step 5: Build Curated Dataset from PSQL

```
make build_curated_from_psql
```

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

To do this, just set the `POP_SIZE` environmental variable. For example, to build a curated dataset with only the first 1000 patients, we could do:

```
POP_SIZE=100 make build_curated_from_psql
```

# Alternate Step-by-step instructions (no Makefile)
The first several steps are the same here as above.

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
conda activate mimic_data_extraction
```

Note that after this installation step, you may need to manually install some packages via conda/pip because
the environment has a mix of pip/conda dependencies. These include `datapackage`, `spacy`, and `scispacy`. You
will also then need to install the english language model for spacy, via:
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
`psql -d mimic -f niv-durations.sql`.

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
