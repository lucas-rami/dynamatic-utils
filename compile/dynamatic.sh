#!/bin/bash

# ===- dynamatic.sh - Compile w/ new Dynamatic -----------------*- Bash -*-=== #
# 
# This script compiles all benchmarks from legacy Dynamatic using the new
# Dynamatic flow, eventually producing a DOT file representing the circuit for
# each benchmark. Results of intermediate conversion steps are also kept for
# completeness. All results are placed in a folder named "out" created next to
# this file.
# 
# ===----------------------------------------------------------------------=== #

# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source ../utils.sh

# Check that required environment variables exist
check_env_variables \
    BENCHMARKS_PATH \
    POLYGEIST_PATH \
    POLYGEIST_CLANG_BIN \
    MLIR_OPT_BIN \
    DYNAMATIC_OPT_BIN

compile () {
    local bench_dir=$1
    local name="$(basename $bench_dir)"
    local out="$bench_dir/dynamatic"
    mkdir -p "$out"

    # Generated files
    local f_scf="$out/scf.mlir"
    local f_affine="$out/affine.mlir"
    local f_affine_mem="$out/affine_mem.mlir"
    local f_std="$out/std.mlir"
    local f_handshake="$out/handshake.mlir"
    local f_netlist="$out/netlist.mlir"
    local f_dot="$out/$name.dot"
    local f_png="$out/$name.png"

    # source code -> affine dialect 
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    "$POLYGEIST_CLANG_BIN" "$bench_dir/$name.c" -I "$include" \
        --function="$name" -S -O3 --memref-fullrank --raise-scf-to-affine \
        > "$f_affine" 2>/dev/null
    exit_on_fail "Failed source -> affine conversion" "Lowered to affine"
    
    # memory analysis 
    "$DYNAMATIC_OPT_BIN" "$f_affine" --allow-unregistered-dialect \
        --name-memory-ops --analyze-memory-accesses > "$f_affine_mem"
    exit_on_fail "Failed memory analysis" "Passed memory analysis"

    # affine dialect -> scf dialect
    "$DYNAMATIC_OPT_BIN" "$f_affine_mem" --allow-unregistered-dialect \
        --lower-affine-to-scf > "$f_scf"
    exit_on_fail "Failed affine -> scf conversion" "Lowered to scf"

    # scf dialect -> standard dialect
    "$MLIR_OPT_BIN" "$f_scf" --allow-unregistered-dialect --convert-scf-to-cf \
        --canonicalize --cse --sccp --symbol-dce --control-flow-sink \
        --loop-invariant-code-motion --canonicalize > "$f_std"
    exit_on_fail "Failed scf -> std conversion" "Lowered to std"

    # standard dialect -> handshake dialect
    "$DYNAMATIC_OPT_BIN" "$f_std" --allow-unregistered-dialect \
        --flatten-memref --flatten-memref-calls --push-constants \
        --lower-std-to-handshake-fpga18="id-basic-blocks" > "$f_handshake"
    exit_on_fail "Failed std -> handshake conversion" "Lowered to handshake"

    # handshake dialect -> netlist
    "$DYNAMATIC_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-materialize-forks-sinks --lower-handshake-to-netlist \
        > "$f_netlist"
    exit_on_fail "Failed handshake -> netlist conversion" "Lowered to netlist"

    # Create DOT graph
    "$DYNAMATIC_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
        --export-dot > /dev/null 2>&1
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
    return 0
}

process_benchmark () {
    local name=$1
    local out="out"

    echo_section "Compiling $name"

    # Copy benchmark from Dynamatic folder to local folder
    copy_src "$BENCHMARKS_PATH/$name/src" "$out/$name" "$name" "cpp"
    exit_on_fail "Failed to copy source files"

    # Compile with new Dynamatic
    compile "$out/$name"
    echo_status "Failed to compile $name" "Done compiling $name"
    return $?
}

for name in $BENCHMARKS_PATH/*/; do
    bname="$(basename $name)"
    process_benchmark "$bname"
    echo ""
done

echo "[INFO] All done!"
