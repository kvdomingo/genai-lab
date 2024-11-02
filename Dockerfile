FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_VERSION=1.8.4
ENV POETRY_HOME=/opt/poetry
ENV DAGSTER_HOME=/opt/dagster
ENV PATH=${PATH}:${POETRY_HOME}/bin

SHELL [ "/bin/bash", "-euxo", "pipefail", "-c" ]

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -sSL https://install.python-poetry.org | python - && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true

WORKDIR /app

ENTRYPOINT [ "/bin/bash", "-c" ]
