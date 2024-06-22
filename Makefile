## ----------------------------------------------------------------------
## Defines targets and dependencies to run the proposed pipeline.
## Our targets are run from inside a provided docker image.
## For persistence, we create and mount local directories.
## These can be set with the following environment variables:
##     - BREASTCLF_RUN_DIR: Path where model checkpoints and results are stored.
##     - BREASTCLF_DATA_DIR: Path where input data (DICOM, PNG, ...) are stored.
## We also allow to train our models using GPU acceleration with the CUDA
## library.
## To do so, set BREAST_USE_CUDA=1, e.g:
##     BREASTCLF_USE_CUDA=1 make best-model
## ----------------------------------------------------------------------

help:     ## Show this help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

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

dirs: ## create local directories

	@echo ">>> Creating output directories"
	mkdir -p $(BREASTCLF_DATA_DIR)/dicom
	mkdir -p $(BREASTCLF_DATA_DIR)/png
	mkdir -p $(BREASTCLF_RUN_DIR)

build-image: ## for development: Build docker image locally
	@echo ">>> building image"
	$(DOCKER_EXEC) build . \
		-t $(DOCKER_TAGGED_IMAGE)

push-image: ## for development: push docker image to docker-hub
	@echo ">>> pusing image"
	$(DOCKER_EXEC) push $(DOCKER_TAGGED_IMAGE)

raw-data: dirs ## download raw DICOM data
	@echo ">>> downloading raw data"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd fetch-raw-data -w 32 /assets/meta.csv /data/dicom

annotated-patient-meta: raw-data ## apply annotations to patients
	@echo ">>> applying annotations to meta-data"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd merge-meta-and-annotations /assets/meta.csv /assets/annotations.csv /data/meta-annotated.csv

parse-and-merge-dicom: annotated-patient-meta ## parse DICOM files
	@echo ">>> parsing DICOM files"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd build-per-image-meta /data/meta-annotated.csv /data/dicom /data/meta-images.csv

png-images: parse-and-merge-dicom ## convert DICOM to PNG
	@echo ">>> converting DICOM files to PNG"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd dicom-to-png /data/meta-images.csv /data/dicom /data/png

ml-splitted-dataset: png-images ## generate train/val/test splits
	@echo ">>> Generating train/val/test splits"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml split /data/meta-images.csv /data/meta-images-split.csv 0.2 0.2

best-model: ml-splitted-dataset ## train and test our best model
	@echo ">>> Training and testing best model"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml run-experiments.py $(PYTHON_GPU) --best-only

all: ml-splitted-dataset ## train and test all models
	@echo ">>> Training and testing all models"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml run-experiments $(PYTHON_GPU)


clean: ## deletes all data, checkpoints, and results
	rm -rf $(BREASTCLF_RUN_DIR)/*
	rm -rf $(BREASTCLF_DATA_DIR)/*
