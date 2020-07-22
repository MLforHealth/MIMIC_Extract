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

Next, make a new conda environment from [mimic_extract_env.yml](../mimic_extract_env.yml) and activate that environment.

```
conda env create --force -f ../mimic_extract_env.yml
conda activate mimic_data_extraction
```

#### Expected Outcome

The desired enviroment will be created and activated.

#### Expected Resources

Will typically take less than 5 minutes.
Requires a good internet connection.

## Step 3: Build Views for Feature Extraction

Materialized views in the MIMIC PostgreSQL database will be generated. This includes all concept tables in [MIT-LCP Repo](https://github.com/MIT-LCP/mimic-code) and tables for extracting non-mechanical ventilation, and injections of crystalloid bolus and colloid bolus.

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

To do this, just set the POP_SIZE environmental variable. For example, to build a curated dataset with only the first 1000 patients, we could do:

```
POP_SIZE=100 make build_curated_from_psql
```
