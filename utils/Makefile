SHELL:=/bin/bash

PROJECT_ENV_NAME:=mimic_extract_py27_env

CONDA_EXECUTABLE:=$(shell which conda)
CONDA_has_conda_env:=$(shell conda env list | grep ${PROJECT_ENV_NAME})

PSQL_EXECUTABLE:=$(shell which psql)

MIMIC_CODE_DIR:=${shell grep "MIMIC_CODE_DIR" setup_user_env.sh | cut -d'=' -f2}

.PHONY: help
help:                                                 ## Show help messages for each command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/:.*##/:    ##/'

#=== Commands

.PHONY: install_env
install_env: setup_user_env.sh has_conda_exe ./requirements.txt        ## Create new env for this project and install all req'd packages
	{ \
	source setup_user_env.sh; \
	conda create --name ${PROJECT_ENV_NAME} python=2.7; \
	source activate ${PROJECT_ENV_NAME}; \
	conda install --yes --file ./requirements.txt; \
	conda install -c conda-forge datapackage; \
	which wget || conda install wget --yes; \
	}

.PHONY: build_curated_from_psql           

build_curated_from_psql: setup_user_env.sh has_psql_exe has_conda_env   ## Build curated flat files from PSQL db, using this repo
	{ \
	source setup_user_env.sh; \
	source activate ${PROJECT_ENV_NAME}; \
	bash build_curated_from_psql.sh; \
    }

#=== Required mimic-code/ repo
# Rules below will download the repo if it doesn't exist
# Expected location MIMIC_CODE_DIR is defined in setup_user_env.sh

.PHONY: clone_mimic_code_repo ${MIMIC_CODE_DIR}/buildmimic/postgres/Makefile

clone_mimic_code_repo: ${MIMIC_CODE_DIR}/buildmimic/postgres/Makefile

${MIMIC_CODE_DIR}/buildmimic/postgres/Makefile: setup_user_env.sh
	{ \
	source setup_user_env.sh; \
	[ -e $@ ] || git clone https://github.com/MIT-LCP/mimic-code/ ${MIMIC_CODE_DIR}/; \
	}

#=== Build concepts
.PHONY: build_concepts
build_concepts: build_concepts_mimic_code build_extra_concepts

.PHONY: build_concepts_mimic_code
build_concepts_mimic_code: setup_user_env.sh clone_mimic_code_repo
	{ \
	source setup_user_env.sh; \
	cd ${MIMIC_CODE_DIR}/concepts; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./make-concepts.sql; \
	cd ../../mimic_extract/utils; \
	}

.PHONY: build_extra_concepts
build_extra_concepts: setup_user_env.sh niv-durations.sql crystalloid-bolus.sql colloid-bolus.sql
	{ \
	source setup_user_env.sh; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./niv-durations.sql; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./crystalloid-bolus.sql; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./colloid-bolus.sql; \
	}

#=== Env Checks

.PHONY: has_conda_env
has_conda_env: has_conda_exe 
ifndef CONDA_has_conda_env
	$(error "Error: Conda env '${PROJECT_ENV_NAME}' does not exist. Do this: 'make install_env'")
endif
	

.PHONY: has_conda_exe
has_conda_exe: setup_user_env.sh
ifndef CONDA_EXECUTABLE
	$(error "Error: 'conda' is undefined. Please install/add to current path.")
endif

.PHONY: has_psql_exe
has_psql_exe: setup_user_env.sh
ifndef PSQL_EXECUTABLE
	$(error "Error: 'psql' is undefined. Please install/add to current path.")
endif
