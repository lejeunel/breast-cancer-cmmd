DOCKER_EXEC := docker
PYTHON_EXEC := python
DOCKER_IMAGE := lejeunel379/breastclf
VERSION := $(shell git describe --always --dirty --long)
DOCKER_TAGGED_IMAGE := $(DOCKER_IMAGE):$(VERSION)

DOCKER_GPU := $(if $(BREASTCLF_USE_CUDA),--gpus all,--gpus 0)
PYTHON_GPU := $(if $(BREASTCLF_USE_CUDA),--cuda,)
BREASTCLF_DATA_DIR ?= ./data
BREASTCLF_RUN_DIR ?= ./runs


DOCKER_RUN := $(DOCKER_EXEC) \
	run \
	--ipc=host \
	$(DOCKER_GPU) \
	-it \
	-u `id -u $(USER)`:`id -g $(USER)` \
	-e MPLCONFIGDIR=$(BREASTCLF_DATA_DIR) \
	--mount type=bind,source=$(BREASTCLF_RUN_DIR),target=/runs \
	--mount type=bind,source=$(BREASTCLF_DATA_DIR),target=/data \
	--mount type=bind,source=./assets,target=/assets \
	$(DOCKER_TAGGED_IMAGE)


default: all


## for development: Build docker image locally
build-image:
	@echo ">>> building image"
	$(DOCKER_EXEC) build . \
		-t $(DOCKER_TAGGED_IMAGE)

## for development: push docker image to docker-hub
push-image:
	@echo ">>> pushing image"
	$(DOCKER_EXEC) push $(DOCKER_TAGGED_IMAGE)

## download raw DICOM data
raw-data:
	@echo ">>> downloading raw data"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd fetch-raw-data -w 32 /assets/meta.csv /data/dicom

## apply annotations to patients
annotated-meta: raw-data
	@echo ">>> applying annotations to meta-data"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd merge-meta-and-annotations /assets/meta.csv /assets/annotations.csv /data/meta-annotated.csv

## parse DICOM files
parse-dicom: annotated-meta
	@echo ">>> parsing DICOM files"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd build-per-image-meta /data/meta-annotated.csv /data/dicom /data/meta-images.csv

## convert DICOM to PNG
png-images: parse-dicom
	@echo ">>> converting DICOM files to PNG"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd dicom-to-png /data/meta-images.csv /data/dicom /data/png

## generate train/val/test splits
ml-splitted-dataset: png-images
	@echo ">>> Generating train/val/test splits"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml split /data/meta-images.csv /data/meta-images-split.csv 0.2 0.2

## train and test our best model
best-model: ml-splitted-dataset
	@echo ">>> Training and testing best model"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml run-experiments $(PYTHON_GPU) --best-only

## train and test all models
all: ml-splitted-dataset
	@echo ">>> Training and testing all models"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml run-experiments $(PYTHON_GPU)

## deletes all data, checkpoints, and results
clean:
	rm -rf $(BREASTCLF_RUN_DIR)/*
	rm -rf $(BREASTCLF_DATA_DIR)/*

.DEFAULT_GOAL := show-help

help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) == Darwin && echo '--no-init --raw-control-chars')
