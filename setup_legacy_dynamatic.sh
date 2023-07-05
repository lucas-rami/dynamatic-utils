#!/bin/bash

# ===- setup_legacy_dynamatic.sh - Legacy Dynamatic setup ------*- Bash -*-=== #
# 
# This script clones legacy Dynamatic into the directory expected by the 
# LEGACY_DYNAMATIC_PATH environment variable in the repository's .env file. It
# also symlinks all of legacy Dynamatic's binaries in the right place for the
# frontend to find them. 
# 
# ===----------------------------------------------------------------------=== #


# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source utils.sh

# Check that required environment variables exist
check_env_variables \
    LEGACY_DYNAMATIC_PATH \
    LEGACY_DYNAMATIC_ROOT


# Path to directory where to clone legacy Dynamatic
CLONE_PATH="$(dirname $LEGACY_DYNAMATIC_ROOT)"

# Path to directory where to symlink all legacy Dynamatic binaries from
SYMLINK_BIN_PATH="$LEGACY_DYNAMATIC_PATH/dhls/bin"

# Create symbolic link from SYMLINK_BIN_PATH to an executable file built by
# legacy Dynamatic. The symbolic link's name is the same as the executable file.
# The path to the executable file must be passed as the first argument to this
# function and be absolute.
create_symlink() {
    local src=$1
    local dst="$SYMLINK_BIN_PATH/$(basename $1)"
    echo "$dst -> $src"
    ln -f --symbolic $src $dst
}

# Clone the repository if necessary
if [[ -d "$LEGACY_DYNAMATIC_ROOT" ]]; then
    echo "[INFO] It looks like Dynamatic is already installed at \
$LEGACY_DYNAMATIC_ROOT, the script will try to use that installation instead \
of cloning again."
else
    mkdir -p $CLONE_PATH
    git clone https://github.com/lana555/dynamatic $LEGACY_DYNAMATIC_ROOT
fi

echo "[INFO] Creating symlinks..."
mkdir -p "legacy-dynamatic/dhls/bin"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/compile"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/create_project"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/dcfg"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/compile"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/design_compiler"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/dynamatic"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/lsq_generate"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/update-dynamatic"
create_symlink "$LEGACY_DYNAMATIC_ROOT/bin/write_hdl"
create_symlink "$LEGACY_DYNAMATIC_ROOT/Buffers/bin/buffers"
create_symlink "$LEGACY_DYNAMATIC_ROOT/dot2vhdl/bin/dot2vhdl"
create_symlink "$LEGACY_DYNAMATIC_ROOT/Frontend/bin/analyze"
create_symlink "$LEGACY_DYNAMATIC_ROOT/Frontend/bin/dhls"
create_symlink "$LEGACY_DYNAMATIC_ROOT/Frontend/bin/elaborate"

echo "[INFO] All done!"
