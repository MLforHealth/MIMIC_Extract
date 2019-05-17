# README for Building Curated Mimic Dataset from scratch

* Author: Mike Hughes (mike@michaelchughes.com), Shirly Wang (shirlywang@cs.toronto.edu)
* Date: 2019-04-21

# Table of Contents

* [Step 0: Required software and prereqs](#step-0-required-software-and-prereqs)
* [Step 1: Setup env vars for local system](#step-1-setup-env-vars-for-current-local-system)
* [Step 2: Create conda environment](#step-2-create-conda-environment)
* [Step 3: Build Views for Feature Extraction](#step-3-build-views-for-feature-extraction)
* [Step 4: Set Cohort Selection and Extraction Criteria](#step-4-set-cohort-selection-and-extraction-criteria)
* [Step 5: Build Curated Dataset from PostgreSQL](#step-5-build-curated-dataset-from-psql)


# Step-by-step Instructions

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

Edit [setup_user_env.sh](./setup_user_env.sh) so all paths point to valid locations on local file system. 

## Step 2: Create conda environment

Next, make a new conda environment from (../mimic_extract_env_py36.yml) and activate that environment.

```
conda env create -f ../mimic_extract_env_py36.yml
conda activate mimic_extract_py36
```

#### Expected Outcome

The desired enviroment will be created and activated.

#### Expected Resources

Will typically take less than 5 minutes.
Requires a good internet connection.

## Step 3: Build Views for Feature Extraction

Materialized views in the MIMIC PostgreSQL database will be generated. This includes all concept tables in [MIT-LCP Repo](https://github.com/MIT-LCP/mimic-code) and tables for extracting non-mechanical ventilation, and injections of crystalloid bolus and colloid bolus.

```
make build_concepts
```

## Step 4: Set Cohort Selection and Extraction Criteria

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
* **vitals_labs**: time-varying vitals and labs (hourly mean, count and standard deviation)
* **vitals_labs_mean**: time-varying vitals and labs (hourly mean only)
* **interventions**: hourly binary indicators for administered interventions

#### Expected Resources

Will probably take 5-10 hours.
Will require a good machine with at least 50GB RAM.

#### Setting the population size

By default, this step builds a dataset with all eligible patients. Sometimes, we wish to run with only a small subset of patients (debugging, etc.).

To do this, just set the POP_SIZE environmental variable. For example, to build a curated dataset with only the first 1000 patients, we could do:

```
POP_SIZE=1000 make build_curated_from_psql
```



