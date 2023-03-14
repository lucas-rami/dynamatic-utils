#!/bin/bash

# ===- circt.sh - Compile w/ CIRCT -----------------------------*- Bash -*-=== #
# 
# This script compiles all benchmarks from legacy Dynamatic using the CIRCT
# flow, eventually producing a DOT file representing the circuit for each
# benchmark. Results of intermediate conversion steps are also kept for
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
    CIRCT_OPT_BIN

compile () {
    local bench_dir=$1
    local name="$(basename $bench_dir)"
    local out="$bench_dir/circt" 
    mkdir -p "$out"
    
    # Generated files
    local f_affine="$out/affine.mlir"
    local f_std="$out/std.mlir"
    local f_handshake="$out/handshake.mlir"
    local f_hw="$out/hw.mlir"
    local f_dot="$out/$name.dot"
    local f_png="$out/$name.png"

    # source code -> affine dialect 
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    "$POLYGEIST_CLANG_BIN" "$bench_dir/$name.c" \
        -I "$include" --function="$name" -S -O3 --raise-scf-to-affine \
        --memref-fullrank \
        > "$f_affine" 2>/dev/null
    exit_on_fail "Failed source -> affine conversion" "Lowered to affine"
    
    # affine dialect -> standard dialect
    local to_std_passes="-convert-scf-to-cf -canonicalize -cse -sccp \
        -symbol-dce -control-flow-sink -loop-invariant-code-motion \
        -canonicalize"
    "$MLIR_OPT_BIN" "$f_affine" -lower-affine $to_std_passes > "$f_std"
    exit_on_fail "Failed affine -> std conversion" "Lowered to std"

    # standard dialect -> handshake dialect
    "$CIRCT_OPT_BIN" "$f_std" --allow-unregistered-dialect \
        --flatten-memref --flatten-memref-calls \
        --lower-std-to-handshake="disable-task-pipelining source-constants" \
        > "$f_handshake"
    exit_on_fail "Failed std -> handshake conversion" "Lowered to handshake"

    # Create DOT graph
    "$CIRCT_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-print-dot > /dev/null 2>&1 
    if [ $? -ne 0 ]; then
        # DOT gets generated in script directory, remove it 
        rm "$name.dot" 
        
        echo "[ERROR] Failed to create DOT graph"
        return 1
    fi

    # DOT gets generated in script directory, move it to the right place
    mv "$name.dot" "$f_dot"

    # Convert DOT graph to PNG
    dot -Tpng "$f_dot" > "$f_png"
    echo_status "Failed to convert DOT to PNG" "Converted DOT to PNG"
    return 0
}

process_benchmark () {
    local name=$1
    local out="out"

    echo_section "Compiling $name"

    # Copy benchmark from Dynamatic folder to local folder
    copy_src "$BENCHMARKS_PATH/$name/src" "$out/$name" "$name" "cpp"
    exit_on_fail "Failed to copy source files" ""

    # Compile with CIRCT
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
