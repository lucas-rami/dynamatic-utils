#!/bin/bash

# ===- bridge_legacy.sh - Bridge new and legacy Dynamatic ------*- Bash -*-=== #
# 
# This script bridges the new Dynamatic flow with the legacy Dynamatic flow. The
# flow takes as input the handshake-level IR of benchmarks compiled with the new
# flow, applies a pre-processing step to it, and then converts it to a DOT file
# compatible with the legacy flow. This DOT is then passed to the dot2vhdl tool
# from legacy Dynamatic to obtain a VHDL circuit design for the benchmark.
# Results of intermediate conversion steps are also kept for completeness. All
# results are placed in a folder named "out" created next to this file.
# 
# ===----------------------------------------------------------------------=== #

# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source ../utils.sh

# Check that required environment variables exist
check_env_variables \
    DYNAMATIC_OPT_BIN \
    DOT2VHDL_BIN

bridge () {
    local bench_dir=$1
    local name="$(basename $bench_dir)"
    local out="$bench_dir/bridge-legacy"
    local f_src="$bench_dir/dynamatic/handshake.mlir"
    mkdir -p "$out"

    # Generated files
    local f_handshake="$out/handshake.mlir"
    local f_dot="$out/$name.dot"
    local f_png="$out/$name.png"

    # handshake dialect -> legacy-compatible handshake
    "$DYNAMATIC_OPT_BIN" "$f_src" --allow-unregistered-dialect \
        --handshake-prepare-for-legacy --handshake-infer-basic-blocks \
        > "$f_handshake"
    exit_on_fail "Failed handshake -> legacy-compatible handshake" \
        "Lowered to legacy-compatible handshake"

    # Create DOT graph
    "$DYNAMATIC_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
        --handshake-insert-buffers="buffer-size=2 strategy=cycles" \
        --handshake-infer-basic-blocks \
        --export-dot="legacy pretty-print=false" > /dev/null
    if [ $? -ne 0 ]; then
        # DOT gets generated in script directory, remove it 
        rm "$name.dot" 
    
        echo "[ERROR] Failed to create DOT graph"
        return 1
    fi
    echo "[INFO] Created DOT graph"

    # DOT gets generated in script directory, move it to the right place
    mv "$name.dot" "$f_dot"

    # Convert DOT graph to PNG
    dot -Tpng "$f_dot" > "$f_png"
    exit_on_fail "Failed to convert DOT to PNG" "Converted DOT to PNG"

    # Convert DOT graph to VHDL
    "$DOT2VHDL_BIN" "$out/$name" > /dev/null
    echo_status "Failed to convert DOT to VHDL" "Converted DOT to VHDL"
    return 0
}

# Iterate over all benchmarks that were compiled by at least one flow and bridge
# those that have been compiled by the new Dynamatic flow
for benchmark in out/*; do
    name="$(basename $benchmark)"
    out="out"

    # Check whether the IR exists at the handshake level
    if [[ -f "$out/$name/dynamatic/handshake.mlir" ]]; then
        echo_section "Bridging $name"
        bridge "$out/$name"
        exit_on_fail "Failed to bridge $name" "Done bridging $name"
        echo ""
    fi
done

echo "[INFO] All done!"
