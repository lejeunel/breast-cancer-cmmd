FROM python:3.12.0-bookworm as python-base

# https://python-poetry.org/docs#ci-recommendations
ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VENV=/opt/poetry-venv \
    POETRY_CACHE_DIR=/opt/.cache

# Create stage for Poetry installation
FROM python-base as poetry-base

# Creating a virtual environment just for poetry and install it with pip
RUN python3 -m venv $POETRY_VENV \
	&& $POETRY_VENV/bin/pip install -U pip setuptools \
	&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Create a new stage from the base python image
FROM python-base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        xdg-utils

# This fixes a bug in xdg-desktop-menu
RUN mkdir /usr/share/desktop-directories/

# Copy Poetry to app image
COPY --from=poetry-base ${POETRY_VENV} ${POETRY_VENV}

# Add Poetry to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

# Copy Dependencies
COPY poetry.lock pyproject.toml README.org hmtest ./
COPY hmtest/* ./hmtest/

# [OPTIONAL] Validate the project is properly configured
RUN poetry check

RUN wget -q -O nbia-retriever.deb https://cbiit-download.nci.nih.gov/nbia/releases/ForTCIA/NBIADataRetriever_4.4/nbia-data-retriever-4.4.2.deb \
  && dpkg -i nbia-retriever.deb \
  && rm nbia-retriever.deb

# Install Dependencies
RUN poetry install --no-interaction --no-cache

# Copy Application
COPY . /app
