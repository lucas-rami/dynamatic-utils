#!/bin/bash

# ===- legacy_dynamatic.sh - Compile w/ legacy Dynamatic -------*- Bash -*-=== #
# 
# This script compiles all benchmarks from legacy Dynamatic using the legacy
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
    LEGACY_DYNAMATIC_PATH \
    LEGACY_DYNAMATIC_ROOT \
    LLVM_CLANG_BIN \
    LLVM_OPT_BIN \
    TESTSUITE_DYNAMATIC_PATH

# Path to build folders containing legacy Dynamatic object files
ELASTIC_BUILD_PATH="$LEGACY_DYNAMATIC_ROOT/elastic-circuits/_build"
FREQUENCY_COUNTER_PATH="$ELASTIC_BUILD_PATH/FrequencyCounterPass"
FREQUENCY_DATA_PATH="$ELASTIC_BUILD_PATH/FrequencyDataGatherPass"

compile () {
    local bench_dir="$1"
    local name="$(basename $bench_dir)"
    local out="$bench_dir/legacy-dynamatic"
    mkdir -p "$out"

    # Generated files
    local f_ir="$out/ir.ll"
    local f_ir_opt="$out/ir_opt.ll"
    local f_ir_opt_obj="$out/ir_opt.o"
    local f_dot="$out/$name.dot"
    local f_dot_bb="$out/${name}_bb.dot"
    local f_png="$out/$name.png"
    local f_png_bb="$out/${name}_bb.png"

    # source code -> LLVM IR
	"$LLVM_CLANG_BIN" -Xclang -disable-O0-optnone -emit-llvm -S \
        -I "$bench_dir" \
        -c "$bench_dir/$name.c" \
        -o $out/ir.ll
    exit_on_fail "Failed to compile to LLVM IR" "Compiled to LLVM IR"
    
    # LLVM IR -> standard optimized LLVM IR
	"$LLVM_OPT_BIN" -mem2reg -loop-rotate -constprop -simplifycfg -die \
        -instcombine -lowerswitch $f_ir -S -o "$f_ir_opt"
    exit_on_fail "Failed to apply standard optimization" "Applied standard optimization"

    # Run frequency analysis
	"$LLVM_CLANG_BIN" -fPIC -Xclang -load -Xclang \
        "$FREQUENCY_COUNTER_PATH/libFrequencyCounterPass.so" -c "$f_ir_opt" \
        -o "$f_ir_opt_obj"
	"$LLVM_CLANG_BIN" -fPIC "$f_ir_opt_obj" \
        "$FREQUENCY_COUNTER_PATH/log_FrequencyCounter.o"
	./a.out
	rm a.out
	"$LLVM_CLANG_BIN" -Xclang -load -Xclang \
        "$FREQUENCY_DATA_PATH/libFrequencyDataGatherPass.so" $f_ir_opt"" -S
	rm *.s

    # standard optimized LLVM IR -> elastic circuit
    "$LLVM_OPT_BIN" \
        -load "$ELASTIC_BUILD_PATH/MemElemInfo/libLLVMMemElemInfo.so" \
        -load "$ELASTIC_BUILD_PATH/ElasticPass/libElasticPass.so" \
        -load "$ELASTIC_BUILD_PATH/OptimizeBitwidth/libLLVMOptimizeBitWidth.so" \
        -load "$ELASTIC_BUILD_PATH/MyCFGPass/libMyCFGPass.so" \
        -polly-process-unprofitable -mycfgpass -S "$f_ir_opt" \
        "-cfg-outdir=$out" > /dev/null 2>&1

    # Remove temporary build files
    rm *_freq.txt mapping.txt out.txt

    # Can't check the return value here, as this often crashes right after
    # generating the files we want
    if [[ ! -f "$out/${name}_graph.dot" || ! -f "$out/${name}_bbgraph.dot" ]]; then
        return 1
    fi
    echo "[INFO] Applied elastic pass"

    # Rename DOT files
    mv "$out/${name}_graph.dot" "$f_dot"
    mv "$out/${name}_bbgraph.dot" "$f_dot_bb"

    # Convert to PNG
    dot -Tpng "$f_dot" > "$f_png"
    echo_status "Failed to convert DOT to PNG" "Converted DOT to PNG"

    dot -Tpng "$f_dot_bb" > "$f_png_bb"
    echo_status "Failed to convert DOT (BB) to PNG" "Converted DOT (BB) to PNG"

    return 0
}

process_benchmark () {
    local name=$1
    local out="out"

    echo_section "Compiling $name"

    # Copy benchmark from Dynamatic folder to local folder
    copy_src "$TESTSUITE_DYNAMATIC_PATH/$name/src" "$out/$name" "$name" "cpp"
    exit_on_fail "Failed to copy source files"

    # Compile with old Dynamatic
    compile "$out/$name"
    echo_status "Failed to compile $name" "Done compiling $name"
    return $?
}

for name in $TESTSUITE_DYNAMATIC_PATH/*/; do
    bname="$(basename $name)"
    process_benchmark "$bname"
    echo ""
done

echo "[INFO] All done!"
