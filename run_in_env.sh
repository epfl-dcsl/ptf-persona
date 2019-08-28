#!/usr/bin/env bash

set -u
venv_location="$1"
if [ ! -f "$venv_location" ]; then
    echo "virtual environment location not found at $venv_location"
    exit 1
fi
set +u
source "$venv_location"
shift

trap "exit" INT TERM
trap "kill 0" EXIT
exec "$@"