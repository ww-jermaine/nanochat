#!/bin/bash

# Wrapper to run the tinyrun pipeline from the repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
exec ./tinyrun.sh

