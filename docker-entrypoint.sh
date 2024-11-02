#!/usr/bin/env bash

set -euxo pipefail

poetry install --no-root --with dev
exec poetry run dagster dev -h 0.0.0.0 -p 3030
