DOCKER_EXEC := docker
DOCKER_IMAGE := lejeunel379/hmi
VENV := hera-mi-test
PYTHON_VERSION := 3.12.0
VERSION := $(shell git describe --always --dirty --long)
DOCKER_TAGGED_IMAGE := $(DOCKER_IMAGE):$(VERSION)
POETRY_RUN := poetry run python


ifdef HMTEST_DATA_DIR
DATA_DIR := $(HMTEST_DATA_DIR)
else
DATA_DIR := ./data
endif

ifdef HMTEST_RUN_DIR
RUN_DIR := $(HMTEST_RUN_DIR)
else
RUN_DIR := ./runs
endif


DOCKER_RUN := $(DOCKER_EXEC) \
	run -it \
	--mount type=bind,source=$(RUN_DIR),target=/runs \
	--mount type=bind,source=$(DATA_DIR),target=/data \
	$(DOCKER_TAGGED_IMAGE)

default:
	echo "See readme"

init:
	pyenv install ${PYTHON_VERSION}
	pyenv virtualenv ${PYTHON_VERSION} ${VENV}
	poetry install

build-image:
	$(DOCKER_EXEC) build . \
		-f ./Dockerfile \
		-t $(DOCKER_TAGGED_IMAGE)

fetch-raw-data:
	mkdir -p $(DATA_DIR)/dicom
	$(DOCKER_RUN) $(POETRY_RUN) hmtest/main.py fetch-raw-data -w 32 assets/meta.csv /data/dicom

preprocess-data: fetch-raw-data
	mkdir -p $(DATA_DIR)/png
	$(DOCKER_RUN) $(POETRY_RUN) hmtest/main.py fetch-raw-data -w 32 assets/meta.csv /data/dicom
