#!/bin/bash
# === Usage ===
# TODO

# NOTE This should use Polymer, but I forgot to include the flag to disable
# checking for ABI breaking changes and I don't want to rebuild it
# Check that required environment variables are defined
if [[ -z "$POLYGEIST_PATH" ]]; then
    echo "Environment variable \"POLYGEIST_PATH\" is not defined. Abort."
    exit
fi

# Convert potential relative path to absolute
POLYGEIST_DIR=`realpath "$POLYGEIST_PATH"`
echo "Using local frontend installation at \"$POLYGEIST_DIR\""
echo ""

# Parse arguments
COMPILE_ALL=0
for arg in "$@"; 
do
    case "$arg" in 
        "--all")
            COMPILE_ALL=1
            ;;
        *)
            ;;
    esac
done

# Define some paths
SCRIPT_DIR=$PWD
BENCHMARKS_DIR="$SCRIPT_DIR/benchmarks"

analyze_llvm () {
    local name=$1
    local opt="$POLYGEIST_PATH/build/bin/opt"
    local llvm_pass="$SCRIPT_DIR/build/LLVMIRStats/libLLVMIrStats.so"
    local llvm_dir="$BENCHMARKS_DIR/$1/llvm"
    local llvm_ir="$llvm_dir/final.ll"
    "$opt" -load "$llvm_pass" -ir-stats -enable-new-pm=0 "$llvm_ir" > \
        /dev/null 2> "$llvm_dir/stats.txt"
}

analyze_mlir () {
    local name=$1
    local opt="$SCRIPT_DIR/build/bin/tools-opt"
    local mlir_dir="$BENCHMARKS_DIR/$1/mlir"
    local mlir="$mlir_dir/std.mlir"
    local mlir_opt="$mlir_dir/std_opt.mlir"
    "$opt" "$mlir" --ir-stats > /dev/null 2> "$mlir_dir/stats.txt"
    "$opt" "$mlir_opt" --ir-stats > /dev/null 2> "$mlir_dir/stats_opt.txt"
}

analyze() {
    local name=$1
    echo "Processing $name"
    analyze_llvm "$name"
    analyze_mlir "$name"
}

# Process benchmarks
if [ $COMPILE_ALL -eq 1 ]; then
    for name in $BENCHMARKS_DIR/*/; do
        analyze "$(basename $name)"
    done
else
    for name in "$@"; do
        if [[ $name != --* ]]; then
            analyze "$name"
        fi
    done
fi
echo "Done!"
