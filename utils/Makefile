SHELL:=/bin/bash

PSQL_EXECUTABLE:=$(shell which psql)

MIMIC_CODE_DIR:=${shell grep "MIMIC_CODE_DIR" setup_user_env.sh | cut -d'=' -f2}

#=== Commands

.PHONY: build_curated_from_psql           

build_curated_from_psql: setup_user_env.sh has_psql_exe   ## Build curated flat files from PSQL db, using this repo
	{ \
	source ./setup_user_env.sh; \
	bash build_curated_from_psql.sh; \
	}

#=== Required mimic-code/ repo
# Rules below will download the repo if it doesn't exist
# Expected location MIMIC_CODE_DIR is defined in setup_user_env.sh

.PHONY: clone_mimic_code_repo ${MIMIC_CODE_DIR}/buildmimic/postgres/Makefile

clone_mimic_code_repo: ${MIMIC_CODE_DIR}/buildmimic/postgres/Makefile

${MIMIC_CODE_DIR}/buildmimic/postgres/Makefile: setup_user_env.sh
	{ \
	source ./setup_user_env.sh; \
	[ -e $@ ] || git clone https://github.com/MIT-LCP/mimic-code/ ${MIMIC_CODE_DIR}/; \
	}

#=== Build concepts
.PHONY: build_concepts
build_concepts: build_concepts_mimic_code build_extra_concepts

.PHONY: build_concepts_mimic_code
build_concepts_mimic_code: setup_user_env.sh clone_mimic_code_repo
	{ \
	source ./setup_user_env.sh; \
	cd ${MIMIC_CODE_DIR}/concepts; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./make-concepts.sql; \
	cd ../../MIMIC_Extract/utils; \
	}

.PHONY: build_extra_concepts
build_extra_concepts: setup_user_env.sh niv-durations.sql crystalloid-bolus.sql colloid-bolus.sql
	{ \
	source ./setup_user_env.sh; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./niv-durations.sql; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./crystalloid-bolus.sql; \
	psql -U ${DBUSER} "${DBSTRING}" -h ${HOST} -f ./colloid-bolus.sql; \
	}

#=== Env Checks

.PHONY: has_psql_exe
has_psql_exe: setup_user_env.sh
ifndef PSQL_EXECUTABLE
	$(error "Error: 'psql' is undefined. Please install/add to current path.")
endif
