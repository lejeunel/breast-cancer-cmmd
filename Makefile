DOCKER_EXEC := docker
DOCKER_IMAGE := lejeunel379/hmi
VERSION := $(shell git describe --always --dirty --long)
DOCKER_TAGGED_IMAGE := $(DOCKER_IMAGE):$(VERSION)
POETRY_RUN := poetry run python

BREASTCLF_DATA_DIR ?= ./data
BREASTCLF_RUN_DIR ?= ./runs

DOCKER_RUN := $(DOCKER_EXEC) \
	run -it \
	--mount type=bind,source=$(BREASTCLF_RUN_DIR),target=/runs \
	--mount type=bind,source=$(BREASTCLF_DATA_DIR),target=/data \
	--mount type=bind,source=./assets,target=/assets \
	$(DOCKER_TAGGED_IMAGE)

default:
	echo "See readme"

build-image:
	$(DOCKER_EXEC) build . \
		-t $(DOCKER_TAGGED_IMAGE)

push-image:
	$(DOCKER_EXEC) push $(DOCKER_TAGGED_IMAGE)

raw-data:
	mkdir -p $(BREASTCLF_DATA_DIR)/dicom
	$(DOCKER_RUN) $(POETRY_RUN) breastclf/main.py cmmd fetch-raw-data -w 32 /assets/meta.csv /data/dicom

annotated-patient-meta: raw-data
	$(DOCKER_RUN) $(POETRY_RUN) breastclf/main.py cmmd merge-meta-and-annotations /assets/meta.csv /assets/annotations.csv /data/meta-annotated.csv

per-image-meta: annotated-patient-meta
	$(DOCKER_RUN) $(POETRY_RUN) breastclf/main.py cmmd build-per-image-meta /data/meta-annotated.csv /data/dicom /data/meta-images.csv

png-images: per-image-meta
	mkdir -p $(BREASTCLF_DATA_DIR)/png
	$(DOCKER_RUN) $(POETRY_RUN) breastclf/main.py cmmd dicom-to-png /data/meta-images.csv /data/dicom /data/png

ml-splitted-dataset: png-images
	$(DOCKER_RUN) $(POETRY_RUN) breastclf/main.py ml split /data/meta-images.csv /data/meta-images-split.csv 0.2 0.2


clean:
	sudo rm -rf $(BREASTCLF_RUN_DIR)/*
	sudo rm -rf $(BREASTCLF_DATA_DIR)/*
