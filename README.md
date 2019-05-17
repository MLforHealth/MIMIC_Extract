# **MIMIC-Extract**:A Data Extraction, Preprocessing, and Representation Pipeline for MIMIC-III

# About
This repo contains code for **MIMIC-Extract**. It has been divided into the following folders:
* Data: Locally contains the data to be extracted.
* Notebooks: Jupyter Notebooks demonstrating test cases and usage of output data in risk and intervention prediction tasks.
* Resources: Consist of Rohit_itemid.txt which describes the correlation of MIMIC-III item ids with those of MIMIC II as used by Rohit; itemid_to_variable_map.csv which is the main file used in data extraction - consists of groupings of item ids as well as which item ids are ready to extract; variable_ranges.csv which describes the normal variable ranges for the levels assisting in extraction of proper data. It also contains expected schema of output tables.
* Utils: scripts and detailed instructions for running **MIMIC-Extract** data pipeline.
* `mimic_direct_extract.py`: extraction script. 
* `mimic_extract_env_py36.yml`: environment file
# Instructions
To implement **MIMIC_Extract**, follow the instructions in `/utils`.
# Paper

