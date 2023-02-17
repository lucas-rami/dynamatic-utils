#!/bin/bash

# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source ./utils.sh

# Check that required environment variables exist
check_env_variables \
    OLD_DYNAMATIC_PATH \
    LLVM_CLANG_BIN \
    LLVM_OPT_BIN \
    BENCHMARKS_PATH \
    OUT_PATH

compile () {
    local bench_dir="$1"
    local name="$(basename $bench_dir)"
    local out="$bench_dir/old_dynamatic"
    mkdir -p "$out"

    # Generated files
    local f_ir="$out/ir.ll"
    local f_ir_opt="$out/ir_opt.ll"
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

    # standard optimized LLVM IR -> elastic circuit
	local passes_dir="$OLD_DYNAMATIC_PATH/dhls/etc/dynamatic/elastic-circuits/_build/"
    "$LLVM_OPT_BIN" \
        -load "$passes_dir/MemElemInfo/libLLVMMemElemInfo.so" \
        -load "$passes_dir/ElasticPass/libElasticPass.so" \
        -load "$passes_dir/OptimizeBitwidth/libLLVMOptimizeBitWidth.so" \
        -load "$passes_dir/MyCFGPass/libMyCFGPass.so" \
        -polly-process-unprofitable -mycfgpass -S "$f_ir_opt" \
        "-cfg-outdir=$out" > /dev/null 2>&1
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
    local out="$OUT_PATH/compile"

    echo_section "Compiling $name"

    # Copy benchmark from Dynamatic folder to local folder
    copy_src "$BENCHMARKS_PATH/$name/src" "$out/$name" "$name" "cpp"
    exit_on_fail "Failed to copy source files"

    # Compile with old Dynamatic
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
