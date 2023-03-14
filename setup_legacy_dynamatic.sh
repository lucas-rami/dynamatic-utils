#!/bin/bash

# ===- setup_legacy_dynamatic.sh - Legacy Dynamatic setup ------*- Bash -*-=== #
# 
# This script clones legacy Dynamatic into the directory expected by the 
# LEGACY_DYNAMATIC_PATH environment variable in the repository's .env file. 
# 
# ===----------------------------------------------------------------------=== #

# Legacy Dynamatic expects to be placed in a directory tree that ends with
# .../dhls/etc, we have to oblige 
CLONE_DIR="legacy-dynamatic/dhls/etc"

# Create directory to clone into
mkdir -p $CLONE_DIR
cd $CLONE_DIR

# Clone the repository
if [[ -d "dynamatic" ]]; then
    echo "[INFO] It looks like Dynamatic is already installed at \
${CLONE_DIR}/dynamatic, the script will try to use that installation instead \
of cloning again."
else
    git clone https://github.com/lana555/dynamatic
fi

echo "[INFO] All done!"