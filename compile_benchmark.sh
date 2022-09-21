#!/bin/bash
# === Usage ===
# Arguments are name of the benchmark to compile into LLVM IR

# TODO check that environment variables are defined

# Some paths
SCRIPT_DIR=$PWD
BENCHMARKS_DIR="$SCRIPT_DIR/benchmarks"
DYNAMATIC_ROOT="$DYNAMATIC_PATH/dhls/etc/dynamatic"
DYNAMATIC_BENCHMARKS="$DYNAMATIC_ROOT/Regression_test/examples"

AFFINE_PASSES="--affine-data-copy-generate --affine-expand-index-ops \
--affine-loop-coalescing --affine-loop-fusion \
--affine-loop-invariant-code-motion --affine-loop-normalize --affine-loop-tile \
--affine-loop-unroll --affine-loop-unroll-jam --affine-parallelize  \
--affine-pipeline-data-transfer --affine-scalrep --affine-simplify-structures \
--affine-super-vectorize"
STD_PASSES="-canonicalize -cse -sccp -symbol-dce -control-flow-sink \
-loop-invariant-code-motion -canonicalize"

# Source environment variables for Dynamatic
cd "$DYNAMATIC_PATH"
source "$DYNAMATIC_PATH/.env"


get_bench_local_path () {
    echo "$SCRIPT_DIR/benchmarks/$1"
}

get_bench_regression_path () {
    echo "$DYNAMATIC_BENCHMARKS/$1/src"
}

copy_src () {
    # === Usage ===
    # $1 is name of the benchmark 
    local name="$1"

    # Check whether file already exists in local folder
    local benchmark_dst="$(get_bench_local_path $name)"
    if [ -d "$benchmark_dst" ]; then
        echo "  SRC: Folder exists"
        return 0
    fi

    echo "  SRC: Copying benchmark from Dynamatic"
    mkdir -p "$BENCHMARKS_DIR/$name"
    
    # Copy benchmark from Dynamatic's regression tests to local benchmark
    # folder
    cd "$SCRIPT_DIR" 
    local benchmark_src="$(get_bench_regression_path $name)"
    local c_benchmark="$benchmark_dst/$name.c"
    local h_benchmark="$benchmark_dst/$name.h"
    cp "$benchmark_src/$name.cpp" "$c_benchmark"
    cp "$benchmark_src/$name.h" "$h_benchmark"
    return 0
}

compile_llvm () {
    # === Usage ===
    # $1 is name of the benchmark 
    # Returns:
    #   0 if the benchmark was compiled succesfully
    #   <make's return value> if the benchmark was not compiled succesfully
    local name=$1

    # Check whether LLVM folder already exists in local folder
    local llvm_dir="$(get_bench_local_path $name)/llvm"
    if [ -d "$llvm_dir" ]; then
        echo "  LLVM: Already compiled"
        return 0
    fi

    # Go to Dymatic folder with Makefile to compile benchmarks
    cd "$DYNAMATIC_ROOT/elastic-circuits/examples"

    # Temporarily copy source files from Regression_test to elastic-circuits
    local bench_folder="$(get_bench_regression_path $name)"
    cp "$bench_folder/$name.cpp" "./reg_$name.cpp"
    cp "$bench_folder/$name.h" "./$name.h"

    # Compile benchmark with Dynamatic
    make name="reg_$name" graph > /dev/null 2>&1

    # Delete temporarily copied source files
    rm "reg_$name.cpp"
    rm "$name.h"

    # Stop if make fails
    if [ $? -ne 0 ]; then 
        MAKE_RET=$?
        echo "  LLVM: Compile fail"
        return $MAKE_RET
    fi
    
    # Copy output files from Dynamatic folder to local one
    echo "  LLVM: Compile successfull"
    mkdir -p "$llvm_dir"
    cp "_build/reg_$name/reg_${name}_mem2reg_constprop_simplifycfg_die.ll" \
        "$llvm_dir/final.ll"
    return 0    
}

compile_mlir () {
    # === Usage ===
    # $1 is name of the benchmark 
    # Returns:
    #   0 if the benchmark was compiled succesfully
    #   1 if the benchmark was not compiled succesfully
    local name=$1

    local mlir_dir="$(get_bench_local_path $name)/mlir"
    if [ -d "$mlir_dir" ]; then
        echo "  MLIR: Already compiled"
        return 0
    fi

    local src_file="$(get_bench_local_path $name)/$name.c"
    mkdir -p "$mlir_dir"
    
    # Use Polygesit to compile to scf dialect 
    "$POLYGEIST_PATH/build/bin/cgeist" "$src_file" -function=$name -S -O3 > \
        "$mlir_dir/scf.mlir"
    
    # Lower scf to standard
    "$POLYGEIST_PATH/build/bin/mlir-opt" "$mlir_dir/scf.mlir" \
        -convert-scf-to-cf $STD_PASSES > "$mlir_dir/std.mlir"

    echo "  MLIR: Compile successfull"
    return 0
}

idx=1
N_BENCH=$#
for bench_name in "$@"; 
do
    echo "[$idx/$N_BENCH] Compiling benchmark $bench_name..."
    idx=$((idx+1))
    
    # Copy benchmark from dynamatic folder to local folder
    copy_src "$bench_name"

    # Compile with LLVM
    compile_llvm "$bench_name"

    # Compile with MLIR
    compile_mlir "$bench_name"
done

echo "Done!"
