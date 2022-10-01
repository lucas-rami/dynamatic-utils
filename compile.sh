#!/bin/bash
# === Usage ===
# Arguments are name of the benchmark to compile into LLVM IR

# Check that required environment variables are defined
if [[ -z "$DYNAMATIC_PATH" ]]; then
    echo "Environment variable \"DYNAMATIC_PATH\" is not defined. Abort."
    exit
fi
if [[ -z "$FRONTEND_PATH" ]]; then
    echo "Environment variable \"FRONTEND_PATH\" is not defined. Abort."
    exit
fi

# Convert potential relative path to absolute
DYNAMATIC_DIR=`realpath "$DYNAMATIC_PATH"`
FRONTEND_DIR=`realpath "$FRONTEND_PATH"`
echo "Using local Dynamatic installation at $DYNAMATIC_DIR"
echo "Using local frontend installation at $FRONTEND_DIR"
echo ""

# Parse arguments
COMPILE_ALL=0
FORCE=0
for arg in "$@"; 
do
    case "$arg" in 
        "--all")
            COMPILE_ALL=1
            ;;
        "--force")
            FORCE=1
            ;;
        *)
            ;;
    esac
done

# Define some paths
SCRIPT_DIR=$PWD
BENCHMARKS_DIR="$SCRIPT_DIR/benchmarks"
POLYGEIST_DIR="$FRONTEND_DIR/Polygeist-polymer"
POLYGEIST_BIN_DIR="$POLYGEIST_DIR/build/bin"
POLYMER_BIN_DIR="$FRONTEND_DIR/polymer/build/bin"
LLVM_BIN_DIR="$POLYGEIST_DIR/llvm-project/build/bin"
DYNAMATIC_ELASTIC_DIR="$DYNAMATIC_DIR/dhls/etc/dynamatic/elastic-circuits/examples"
DYNAMATIC_REGRESION_DIR="$DYNAMATIC_DIR/dhls/etc/dynamatic/Regression_test/examples"

get_bench_local_path () {
    echo "$BENCHMARKS_DIR/$1"
}

get_bench_regression_path () {
    echo "$DYNAMATIC_REGRESION_DIR/$1/src"
}

copy_src () {
    # === Usage ===
    # $1 is name of the benchmark 
    local name="$1"

    # Check whether file already exists in local folder
    local benchmark_dst="$(get_bench_local_path $name)"
    if [[ $FORCE -eq 0 && -d "$benchmark_dst" ]]; then
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
    if [ $? -ne 0 ]; then
        local ret=$? 
        echo "  SRC: Failed to copy source, abort"
        return ret
    fi
    cp "$benchmark_src/$name.h" "$h_benchmark"
    if [ $? -ne 0 ]; then
        local ret=$? 
        echo "  SRC: Failed to copy source, abort"
        return ret
    fi
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
    if [[ $FORCE -eq 0 && -d "$llvm_dir" ]] ; then
        echo "  LLVM: Already compiled"
        return 0
    fi

    # Go to Dymatic folder with Makefile to compile benchmarks
    cd "$DYNAMATIC_ELASTIC_DIR"

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
    if [[ $FORCE -eq 0 && -d "$mlir_dir" ]]; then
        echo "  MLIR: Already compiled"
        return 0
    fi
    mkdir -p "$mlir_dir"

    # C source file
    local f_src="$(get_bench_local_path $name)/$name.c"

    # affine dialect
    local f_affine="$mlir_dir/affine.mlir"

    # affine dialect optimized
    local f_affine_opt="$mlir_dir/affine_opt.mlir"

    # std dialect optimized
    local f_std_opt="$mlir_dir/std_opt.mlir"

    # scf dialect
    local f_scf="$mlir_dir/scf.mlir"

    # std dialect (non-optimized)
    local f_std="$mlir_dir/std.mlir"

    # Include path
    local include="$POLYGEIST_DIR/llvm-project/clang/lib/Headers/"

    # Passes to convert from scf to std
    local to_std_passes="-convert-scf-to-cf -canonicalize -cse -sccp \
        -symbol-dce -control-flow-sink -loop-invariant-code-motion \
        -canonicalize"

    #### Compile WITH polyhedral optimization
    # f_src -> f_affine -> f_affine_opt -> f_std_opt

    # # Use Polygeist to compile to affine dialect
    "$POLYGEIST_BIN_DIR/mlir-clang" "$f_src" -I "$include" -function=$name -S \
        -O3 -raise-scf-to-affine > "$f_affine"

    # # Use Polymer to optimize affine dialect
    "$POLYMER_BIN_DIR/polymer-opt" "$f_affine" -reg2mem -extract-scop-stmt \
        -pluto-opt -allow-unregistered-dialect > "$f_affine_opt" 2>/dev/null

    # # Lower scf to standard
    "$LLVM_BIN_DIR/mlir-opt" "$f_affine_opt" -lower-affine -inline \
        $to_std_passes > "$f_std_opt"

    #### Compile WITHOUT polyhedral optimization
    # f_src -> f_scf -> f_std

    # Use Polygeist to compile to scf dialect 
    "$POLYGEIST_BIN_DIR/mlir-clang" "$f_src" -I "$include" -function=$name -S \
        -O3 > "$f_scf"

    # Lower scf to standard
    "$LLVM_BIN_DIR/mlir-opt" "$f_scf" $to_std_passes > "$f_std"

    echo "  MLIR: Compile successfull"
    return 0
}

process_benchmark () {
    # === Usage ===
    # $1 is name of the benchmark 
    #   0 if the benchmark was copied succesfully
    #   1 if the benchmark was not copied succesfully
    local name=$1

    # Copy benchmark from dynamatic folder to local folder
    copy_src "$name"
    if [ $? -ne 0 ]; then 
        return 1
    fi

    # Compile with LLVM
    compile_llvm "$name"

    # Compile with MLIR
    compile_mlir "$name"
    return 0
}

# Source environment variables for Dynamatic
cd "$DYNAMATIC_DIR"
source .env

# Process benchmarks
if [ $COMPILE_ALL -eq 1 ]; then
    for name in $DYNAMATIC_REGRESION_DIR/*/; do
        bname="$(basename $name)"
        echo "Processing $bname"
        process_benchmark "$bname"
    done
else
    for name in "$@"; do
        if [[ $name != --* ]]; then
            echo "Processing $name"
            process_benchmark "$name"
        fi
    done
fi
echo "Done!"
