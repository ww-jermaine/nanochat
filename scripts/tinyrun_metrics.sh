#!/usr/bin/env bash

# Wrapper to run the tinyrun_metrics pipeline from the repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
exec ./tinyrun_metrics.sh

