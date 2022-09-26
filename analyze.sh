#!/bin/bash
# === Usage ===
# TODO

# Check that required environment variables are defined
if [[ -z "$POLYGEIST_PATH" ]]; then
    echo "Environment variable \"POLYGEIST_PATH\" is not defined. Abort."
    exit
fi

# Define some paths
SCRIPT_DIR=$PWD
BENCHMARKS_DIR="$SCRIPT_DIR/benchmarks"

analyze () {
    local name=$1

    # LLVM
    local opt="$POLYGEIST_PATH/build/bin/opt"
    local llvm_pass="$SCRIPT_DIR/build/LLVMIRStats/libLLVMIrStats.so"
    local llvm_ir="$BENCHMARKS_DIR/$1/llvm/final.ll"

    "$opt" -load "$llvm_pass" -ir-stats -enable-new-pm=0 "$llvm_ir" > /dev/null

    # MLIR

}

# Process all benchmarks
for name in $BENCHMARKS_DIR/*/; do
    bname="$(basename $name)"
    echo "Processing $name"
    analyze "$bname"
done

