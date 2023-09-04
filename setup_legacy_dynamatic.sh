#!/bin/bash

# ===- setup_legacy_dynamatic.sh - Legacy Dynamatic setup ------*- Bash -*-=== #
# 
# This script clones and configures legacy Dynamatic for use with the runner
# script. Legacy Dynamatic is cloned inside the directory indicated by the
# LEGACY_DYNAMATIC_PATH environment variable (from the repository's .env file). 
# Similarly, llvm-6.0 and corresponding clang/polly sources are cloned to 
# LEGACY_DYNAMATIC_LLVM_PATH. Then, the script automatically builds
# llvm-6.0/clang/polly, the elastic pass, the dot2vhdl tool, the buffers tool,
# and the hlsverifier tool. It also creates symlinks where legacy Dynamatic's
# frontend expects them in case one wishes to use the latter (note that the
# frontend is not built by this script, as it is not used by the repository's
# scripts).
# 
# ===----------------------------------------------------------------------=== #


# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source utils.sh

# Check that required environment variables exist
check_env_variables \
    BUFFERS_BIN \
    DOT2VHDL_BIN \
    HLS_VERIFIER_BIN \
    LEGACY_DYNAMATIC_PATH \
    LEGACY_DYNAMATIC_LLVM_PATH \
    LEGACY_DYNAMATIC_ROOT

# Path to directory where to symlink all legacy Dynamatic binaries from
SYMLINK_BIN_PATH="$LEGACY_DYNAMATIC_PATH/dhls/bin"

# Create symbolic link from SYMLINK_BIN_PATH to an executable file built by
# legacy Dynamatic. The symbolic link's name is the same as the executable file.
# The path to the executable file must be passed as the first argument to this
# function and be absolute.
create_symlink() {
    local src=$1
    local dst="$SYMLINK_BIN_PATH/$(basename $1)"
    ln -f --symbolic $src $dst
}

# Clones a repository if necessary.
#   $1: path where to clonse the repository
#   $2: display name of the repository
#   $3: URL to remote repository
clone_repo() {
    local clone_path="$1"
    local name="$2"
    local url="$3"
    
    if [[ -d "$clone_path" ]]; then
        echo_info "It looks like $name is already installed at $clone_path, the script will try to use that installation instead of cloning again."
    else
        echo_info "Cloning $name in $clone_path"
        mkdir -p "$(dirname $clone_path)"
        git clone $url --branch release_60 --depth 1 "$clone_path" > /dev/null
    fi
    echo ""
}

# Builds a tool from legacy Dynamatic if necessary.
#   $1: path to binary to build 
build_tool() {
    local bin_path="$1"
    local dir_path="$(dirname $bin_path)"
    local bin_name="$(basename $bin_path)"

    echo ""
    echo_section "Building $bin_name"
    
    # Go into the tool folder and build the binary if necessary
    if [[ -f "$bin_path" ]]; then
        echo_info "$bin_name binary found, the script will not try to rebuilt it."
    else
        mkdir -p "$dir_path"
        cd "$(dirname $dir_path)"
        make clean
        make
        return $?
    fi
    return 0
}

echo_section "Cloning legacy Dynamatic"

# Clone legacy Dynamatic if necessary
if [[ -d "$LEGACY_DYNAMATIC_ROOT" ]]; then
    echo_info "It looks like legacy Dynamatic is already installed at \
$LEGACY_DYNAMATIC_ROOT, the script will try to use that installation instead \
of cloning again." 
else
    echo_info "Cloning legacy Dynamatic in $LEGACY_DYNAMATIC_ROOT"
    mkdir -p "$(dirname $LEGACY_DYNAMATIC_ROOT)"
    git clone https://github.com/lana555/dynamatic $LEGACY_DYNAMATIC_ROOT \
        > /dev/null

    echo_info "Creating symlinks in bin/ directory"
    mkdir -p "$SYMLINK_BIN_PATH"
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
fi
echo ""

echo_section "Building elastic pass"
echo ""
echo_subsection "Cloning LLVM/clang/polly"

clone_repo "$LEGACY_DYNAMATIC_LLVM_PATH"             "llvm-6.0" "http://github.com/llvm-mirror/llvm"
clone_repo "$LEGACY_DYNAMATIC_LLVM_PATH/tools/clang" "clang"    "http://github.com/llvm-mirror/clang"
clone_repo "$LEGACY_DYNAMATIC_LLVM_PATH/tools/polly" "polly"    "http://github.com/llvm-mirror/polly"

echo_subsection "Building LLVM and tools"

# Create a build directory inside the llvm-6.0 directory and go build in there
mkdir -p "$LEGACY_DYNAMATIC_LLVM_PATH/_build"
cd "$LEGACY_DYNAMATIC_LLVM_PATH/_build"
if [[ -f "CMakeCache.txt" ]]; then
    echo_info "CMake configuration for LLVM 6.0 found, the script will not try to reconfigure."
else
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_INSTALL_PREFIX=$LEGACY_DYNAMATIC_LLVM_PATH
fi

# Run ninja if the binaries are not available
if [[ -f "$LLVM_CLANG_BIN" && -f "$LLVM_OPT_BIN" ]]; then
    echo_info "clang and opt binaries found, the script will not try to rebuild them."
else
    ninja -j 4
    ninja install
fi

echo ""
echo_subsection "Building elastic-circuits"

# Create a build directory inside the elastic-circuits folder and go build in
# there
ELASTIC_CIRCUITS="$LEGACY_DYNAMATIC_ROOT/elastic-circuits"
mkdir -p "$ELASTIC_CIRCUITS/_build"
cd "$ELASTIC_CIRCUITS/_build"
if [[ -f "CMakeCache.txt" ]]; then
    echo_info "CMake configuration for elastic-circuits found, the script will not try to reconfigure."
else
    cmake .. -G Ninja -DLLVM_ROOT="$LEGACY_DYNAMATIC_LLVM_PATH"
    ninja
fi

echo ""
echo_subsection "Building log_FrequencyCounter.c manually"

# Manually CC the frequency counter source if necessary
if [[ -f "FrequencyCounterPass/log_FrequencyCounter.o" ]]; then
    echo_info "log_FrequencyCounter.o found in build folder, the script will not try to rebuild it."
else
    cc -c ../FrequencyCounterPass/log_FrequencyCounter.c
    mv log_FrequencyCounter.o FrequencyCounterPass
    echo_info "Built and moved frequency counter"
fi

# Build the tools that we care about
build_tool "$DOT2VHDL_BIN"
if [[ $? -ne 0 ]]; then
    echo -e "\n[ERROR] Failed to build dot2vhdl tool"
    exit 1
fi

build_tool "$BUFFERS_BIN"
if [[ $? -ne 0 ]]; then
    echo -e \
"\n[ERROR] Failed to build buffers tool: if you are using a recent compiler, you may need
to #include <cstdint> in:
- $LEGACY_DYNAMATIC_ROOT/Buffers/src/DFnetlist_MG.cpp, and
- $LEGACY_DYNAMATIC_ROOT/Buffers/src/DFnetlist_buffers.cpp
"
    exit 1
fi

build_tool "$HLS_VERIFIER_BIN"
if [[ $? -ne 0 ]]; then
    echo -e \
"\n[ERROR] Failed to build HLS verifier tool: if you are using a recent compiler, you may
need to #include <sstream> in:
- $LEGACY_DYNAMATIC_ROOT/hls_verifier/HLSVerifier/CInjector.cpp
"
    exit 1
fi

echo ""
echo_info "All done!"
