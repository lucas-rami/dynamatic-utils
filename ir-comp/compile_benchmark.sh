#!/bin/bash
# === Usage ===
# Arguments are name of the benchmark to compile into LLVM IR

# TODO check that environment variables are defined

# Some paths
SCRIPT_DIR=$PWD
BENCHMARKS_DIR="$SCRIPT_DIR/benchmarks"
DYNAMATIC_EXAMPLES_DIR="$DYNAMATIC_PATH/dhls/etc/dynamatic/elastic-circuits/examples" 

# Source environment variables for Dynamatic
cd "$DYNAMATIC_PATH"
source "$DYNAMATIC_PATH/.env"

copy_src () {
    # === Usage ===
    # $1 is name of the benchmark 
    
    SRC_FILE="$BENCHMARKS_DIR/$1/$1.cpp"
    if [ -f "$SRC_FILE" ]; then
        echo "  SRC: File exists"
    else
        echo "  SRC: Copy source"
        mkdir -p "$BENCHMARKS_DIR/$1"
        cp "$DYNAMATIC_EXAMPLES_DIR/$1.cpp" "$SRC_FILE"
    fi
}

compile_llvm () {
    # === Usage ===
    # $1 is name of the benchmark 
    # Returns:
    #   0 if the benchmark was compiled succesfully
    #   1 if the benchmark was already compiled
    #   2 if the benchmark was NOT compiled succesfully

    LLVM_DIR="$BENCHMARKS_DIR/$1/llvm"
    if [ -d "$LLVM_DIR" ]; then
        echo "  LLVM: Already compiled"
        return 1
    fi

    # Compile benchmark with Dynamatic
    cd "$DYNAMATIC_EXAMPLES_DIR"
    make name=$1 graph > /dev/null 2>&1

    # Stop if make fails
    if [ $? -ne 0 ]; then 
        echo "  LLVM: Compile fail"
        return 2
    fi
    
    echo "  LLVM: Compile successfull"
    
    # Move the build files from the [...]/examples/_build folder to the
    # folder where this script was ran from
    mkdir -p "$LLVM_DIR"
    cp "_build/$1/"*.ll "$LLVM_DIR/"
    return 0    
}

compile_mlir () {
    # === Usage ===
    # $1 is name of the benchmark 
    # Returns:
    #   0 if the benchmark was compiled succesfully
    #   1 if the benchmark was already compiled
    #   2 if the benchmark was NOT compiled succesfully

    MLIR_DIR="$BENCHMARKS_DIR/$1/mlir"
    if [ -d "$MLIR_DIR" ]; then
        echo "  MLIR: Already compiled"
        return 1
    fi
    mkdir -p "$MLIR_DIR"

    SRC_FILE="$BENCHMARKS_DIR/$1/$1.cpp"
    
    # Compile to scf dialect 
    "$POLYGEIST_PATH/build/bin/cgeist" "$SRC_FILE" -function=* -S > \
        "$MLIR_DIR/scf.mlir"
    
    # Compile to affine dialect 
    "$POLYGEIST_PATH/build/bin/cgeist" "$SRC_FILE" -function=* -S \
        -raise-scf-to-affine > "$MLIR_DIR/affine.mlir"

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
