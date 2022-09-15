#!/bin/bash
# === Usage ===
# Arguments are name of the benchmark to compile into LLVM IR

# Remember where we started from
SCRIPT_DIR=$PWD

# Source environment variables for Dynamatic and go to folder containing benchmarks
cd "$DYNAMATIC_PATH"
source .env
cd dhls/etc/dynamatic/elastic-circuits/examples

idx=1
N_BENCH=$(($#-1))
for bench_name in "$@"; 
do
    LLVM_FOLDER="$SCRIPT_DIR/benchmarks/$bench_name/llvm"
    echo "[$idx/$N_BENCH] Compiling benchmark $bench_name..."
    idx=$((idx+1))
    
    # Stop if benchmark is already compiled
    if [ -d "$LLVM_FOLDER" ]; then
        echo "Already compiled"
        continue
    fi

    # Compile benchmark
    make name=$bench_name graph > /dev/null 2>&1

    # Stop if make fails
    if [ $? -ne 0 ]; then 
        echo "Failed"
        continue
    fi 

    # Move the build files from the [...]/examples/_build folder to the folder
    # where this script was ran from
    echo "Successful"
    mkdir -p "$LLVM_FOLDER"
    cp "_build/$bench_name/"*.ll "$LLVM_FOLDER/"
done

echo "Done!"