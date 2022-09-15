#!/bin/bash

# === Usage ===
# Arguments are name of the benchmark to compile into LLVM IR

# Remember where we started from
SCRIPT_DIR=$PWD

# Define some variables
DYNAMATIC_PATH=/home/ramirez/dynamatic
DYNAMATIC_TOOLS_PATH=/home/ramirez/dynamatic/tools

# Source environment variables for Dynamatic and go to folder containing benchmarks
cd $DYNAMATIC_PATH
source .env
cd dhls/etc/dynamatic/elastic-circuits/examples

for bench_name in "$@" 
do
    # Compile becnhmark
    echo "Compiling benchmark ${bench_name}..."
    make name=${bench_name} graph > /dev/null 2>&1

    # Check if make succeeded
    if [ $? -eq 0 ]; 
    then 
        # Move the build files from the [...]/examples/_build folder to the folder where this script was ran from
        echo "Successful. Copying results for benchmark ${bench_name}"
        LLVM_FOLDER="${SCRIPT_DIR}/benchmarks/${bench_name}/llvm"
        mkdir -p "${LLVM_FOLDER}"
        cp "_build/${bench_name}/"*.ll "${LLVM_FOLDER}/"
    else 
        echo "Failed to compile benchmark ${bench_name}" 
    fi
done