#!/bin/bash

set -euxo pipefail

poetry install --sync --with dev

exec poetry run dagster dev -h 0.0.0.0 -p 3030
