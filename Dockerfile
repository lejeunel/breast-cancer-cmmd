# The builder image, used to build the virtual environment
FROM python:3.12.0-bookworm as builder

RUN pip install poetry==1.8.2

# https://python-poetry.org/docs#ci-recommendations
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENV_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy Dependencies
COPY poetry.lock pyproject.toml ./

# Install Dependencies
RUN poetry install --no-interaction --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.12-slim-bookworm as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY breast-cancer-cmmd ./breast-cancer-cmmd
