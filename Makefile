DOCKER_IMAGE := lejeunel379/hmi
VENV := hera-mi-test
PYTHON_VERSION := 3.12.0
VERSION := $(shell git describe --always --dirty --long)
DOCKER_TAG := $(DOCKER_IMAGE):$(VERSION)
DOCKER_EXEC := docker

ifdef HMTEST_ASSETS_DIR
ASSETS_DIR := $(HMTEST_ASSETS_DIR)
else
ASSETS_DIR := ./assets
endif

default:
	echo "See readme"

init:
	pyenv install ${PYTHON_VERSION}
	pyenv virtualenv ${PYTHON_VERSION} ${VENV}
	poetry install

build-image:
	$(DOCKER_EXEC) build . \
		-f ./Dockerfile \
		-t $(DOCKER_TAG)

push-image:
	$(DOCKER_EXEC) push $(DOCKER_TAG)

data:
	mkdir -p $(ASSETS_DIR)
