FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_VERSION=1.8.5
ENV POETRY_VIRTUALENVS_CREATE=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_HOME=/opt/poetry
ENV DAGSTER_HOME=/opt/dagster
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /tmp

SHELL [ "/bin/bash", "-euxo", "pipefail", "-c" ]
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgl1 tesseract-ocr poppler-utils

ADD https://install.python-poetry.org install-poetry.py

RUN python install-poetry.py

WORKDIR /app

ENTRYPOINT [ "/app/docker-entrypoint.sh" ]
