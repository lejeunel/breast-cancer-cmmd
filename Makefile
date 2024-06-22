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

dirs:

	@echo ">>> Creating output directories"
	mkdir -p $(BREASTCLF_DATA_DIR)/dicom
	mkdir -p $(BREASTCLF_DATA_DIR)/png
	mkdir -p $(BREASTCLF_RUN_DIR)

build-image:
	@echo ">>> building image"
	$(DOCKER_EXEC) build . \
		-t $(DOCKER_TAGGED_IMAGE)

push-image:
	@echo ">>> pusing image"
	$(DOCKER_EXEC) push $(DOCKER_TAGGED_IMAGE)

raw-data: dirs
	@echo ">>> downloading raw data"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd fetch-raw-data -w 32 /assets/meta.csv /data/dicom

annotated-patient-meta: raw-data
	@echo ">>> applying annotations to meta-data"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd merge-meta-and-annotations /assets/meta.csv /assets/annotations.csv /data/meta-annotated.csv

parse-and-merge-dicom: annotated-patient-meta
	@echo ">>> parsing DICOM files"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd build-per-image-meta /data/meta-annotated.csv /data/dicom /data/meta-images.csv

png-images: parse-and-merge-dicom
	@echo ">>> converting DICOM files to PNG"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py cmmd dicom-to-png /data/meta-images.csv /data/dicom /data/png

ml-splitted-dataset: png-images
	@echo ">>> Generating train/val/test splits"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml split /data/meta-images.csv /data/meta-images-split.csv 0.2 0.2

best-model: ml-splitted-dataset
	@echo ">>> Training and testing best model"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml run-experiments.py $(PYTHON_GPU) --best-only

all: ml-splitted-dataset
	@echo ">>> Training and testing all models"
	@$(DOCKER_RUN) $(PYTHON_EXEC) breastclf/main.py ml run-experiments $(PYTHON_GPU)


clean:
	rm -rf $(BREASTCLF_RUN_DIR)/*
	rm -rf $(BREASTCLF_DATA_DIR)/*
