#!/bin/bash

# Get common functions
source ./utils.sh

# Check that required environment variables exist
check_env_variables \
    DYNAMATIC_PATH \
    LLVM_CLANG_BIN \
    LLVM_OPT_BIN \
    DYNAMATIC_SRC

# Destination folder for output of this script
DST="dynamatic-to-dot"

elastic_circuit () {
    local bench_dir=$1
    local name="$(basename $bench_dir)"
    local llvm_out="$bench_dir/llvm"

    # Generated files
    local f_ir="$bench_dir/ir.ll"
    local f_ir_opt="$bench_dir/ir_opt.ll"
    local f_dot="$bench_dir/$name.dot"
    local f_dot_bb="$bench_dir/${name}_bb.dot"
    local f_png="$bench_dir/$name.png"
    local f_png_bb="$bench_dir/${name}_bb.png"


    # Compile source to LLVM IR
	"$LLVM_CLANG_BIN" -Xclang -disable-O0-optnone -emit-llvm -S \
        -I "$bench_dir" \
        -c "$bench_dir/$name.c" \
        -o $bench_dir/ir.ll
    exit_on_fail "Failed to compile to LLVM IR" "Compiled to LLVM IR"

    # Apply standard optimizations to LLVM IR
	"$LLVM_OPT_BIN" -mem2reg -loop-rotate -constprop -simplifycfg -die \
        -instcombine -lowerswitch $f_ir -S -o "$f_ir_opt"
    exit_on_fail "Failed to apply tandard optimization" "Applied standard optimization"

    # Apply custom optimizations
	local passes_dir="$DYNAMATIC_PATH/dhls/etc/dynamatic/elastic-circuits/_build/"
    "$LLVM_OPT_BIN" \
        -load "$passes_dir/MemElemInfo/libLLVMMemElemInfo.so" \
        -load "$passes_dir/ElasticPass/libElasticPass.so" \
        -load "$passes_dir/OptimizeBitwidth/libLLVMOptimizeBitWidth.so" \
        -load "$passes_dir/MyCFGPass/libMyCFGPass.so" \
        -polly-process-unprofitable -mycfgpass -S "$f_ir_opt" \
        "-cfg-outdir=$bench_dir" > /dev/null 2>&1
    exit_on_fail "Failed to apply custom optimization" "Applied custom optimization"

    # Rename DOT files
    mv "$bench_dir/${name}_graph.dot" "$f_dot"
    mv "$bench_dir/${name}_bbgraph.dot" "$f_dot_bb"

    # Convert to PNG
    dot -Tpng "$f_dot" > "$f_png"
    exit_on_fail "Failed to convert DOT to PNG" "Converted DOT to PNG"
    dot -Tpng "$f_dot_bb" > "$f_png_bb"
    exit_on_fail "Failed to convert DOT (BB) to PNG" "Converted DOT (BB) to PNG"

    return 0
}

process_benchmark () {
    local name=$1

    echo "---- Compiling $name ----"

    # Copy benchmark from Dynamatic folder to local folder
    copy_src "$DYNAMATIC_SRC/$name/src" "$DST/$name" "$name" "cpp"
    exit_on_fail "Failed to copy source files" ""

    # Run elastic circuit pass with Dynamatic
    elastic_circuit "$DST/$name"
    exit_on_fail "Failed to run elastic circuit pass"
    echo 

    echo -e "---- Done! ----\n"
    return 0
}


for name in $DYNAMATIC_SRC/*/; do
    bname="$(basename $name)"
    process_benchmark "$bname"
    if [ $? -ne 0 ]; then 
        return 1
    fi
    echo ""
done

echo -e "---- All done! ----\n"
