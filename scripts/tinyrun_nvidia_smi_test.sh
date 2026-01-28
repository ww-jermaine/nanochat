#!/usr/bin/env bash

# Wrapper to run the tinyrun_nvidia_smi_test pipeline from the repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
exec ./tinyrun_nvidia_smi_test.sh

