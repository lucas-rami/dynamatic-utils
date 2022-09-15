#!/bin/bash
# === Usage ===
# Run without arguments

# Remember where we started from
SCRIPT_DIR=$PWD

BENCHMARK_LIST="benchmark_names.txt"

# Delete previous list of benchmarks
rm "$BENCHMARK_LIST"

# Go to benchmark folder and get name of all benchmarks
cd "${DYNAMATIC_PATH}/dhls/etc/dynamatic/elastic-circuits/examples"
names=""
first=1
for filepath in "$PWD/"?*.cpp;
do
    filename=$(basename -- "$filepath")
    filename="${filename%.*}"
    if [ "$first" -eq 1 ]; then
        first=0
    else
        names+=$'\n'
    fi
    names+="${filename}"
done

# Create file to hold list of benchmarks and populate it
cd "$SCRIPT_DIR"
touch "$BENCHMARK_LIST"
echo "$names" > "$BENCHMARK_LIST"
