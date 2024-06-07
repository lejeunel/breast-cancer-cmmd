DOCKER_IMAGE := lejeunel379/hmi
VENV := hera-mi-test
PYTHON_VERSION := 3.12.0
VERSION := $(shell git describe --always --dirty --long)

default:
    echo "See readme"

init:
	echo $VERSION
	pyenv install ${PYTHON_VERSION}
	pyenv virtualenv ${PYTHON_VERSION} ${VENV}
    poetry install
	pyenv activate ${VENV}

build-image:
    docker build .
        -f ./Dockerfile
        -t $(DOCKER_IMAGE):$(VERSION)

push-image:
    docker push $(DOCKER_IMAGE):$(VERSION)
