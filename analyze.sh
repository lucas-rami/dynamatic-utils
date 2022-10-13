#!/bin/bash

check_env_variables () {
    for env_var in "$@"; do
        local echo_in='echo $env_var' 
        local echo_out="echo \$$(eval $echo_in)"
        local env_val=`eval $echo_out`
        if [[ -z "$env_val" ]]; then
            echo "Environment variable $env_var is not defined, abort"
            exit
        else
            echo "Found $env_var ($env_val)"
        fi
    done
}

echo "---- Checking for environment variables ----"
check_env_variables \
    DYNAMATIC_DST \
    POLYBENCH_DST \
    LLVM_ANALYZE_BIN 
echo "---- Done! ----"
echo ""

# Parse arguments
USE_DYNAMATIC=0
USE_POLYBENCH=0
ALL=0
for arg in "$@"; 
do
    case "$arg" in 
        "--use-dynamatic")
            USE_DYNAMATIC=1
            ;;
        "--use-polybench")
            USE_POLYBENCH=1
            ;;
        "--all")
            ALL=1
            ;;
        *)
            ;;
    esac
done

if [[ $USE_DYNAMATIC -ne 0 && $USE_POLYBENCH -ne 0 ]]; then
    echo "Flags --use-dynamatic and --use-polybench are mutually exclusive"
    exit
fi

if [[ $USE_DYNAMATIC -eq 0 && $USE_POLYBENCH -eq 0 ]]; then
    echo "No test suite specified, defaulting to Dynamatic"
    USE_DYNAMATIC=1
fi

analyze_llvm () {
    local bench_dir=$1
    local llvm_pass="$PWD/build/lib/LLVMIrStats.so"
    local llvm_dir="$bench_dir/llvm"
    local llvm_ir="$llvm_dir/step_4.ll"
    "$LLVM_ANALYZE_BIN" -load "$llvm_pass" -ir-stats -enable-new-pm=0 \
        "$llvm_ir" > /dev/null 2> "$llvm_dir/stats.json"
    if [ $? -ne 0 ]; then 
        echo "  LLVM:  Analysis failed"
        return 1
    fi
    echo "  LLVM:  Analysis succeeded"
    return 0
}

analyze_mlir () {
    local bench_dir=$1
    local opt="$PWD/build/bin/mlir-analyze"
    local mlir_dir="$bench_dir/mlir"
    local mlir="$mlir_dir/std.mlir"
    local mlir_opt="$mlir_dir/std_opt.mlir"

    # Analyze non-optimized code
    "$opt" "$mlir" --ir-stats > /dev/null 2> "$mlir_dir/stats.json"
    if [ $? -ne 0 ]; then 
        echo "  MLIR:  Analysis (non-optimized) failed"
    fi
    echo "  MLIR:  Analysis (non-optimized) succeeded"

    # Analyze optimized code
    "$opt" "$mlir_opt" --ir-stats > /dev/null 2> "$mlir_dir/stats_opt.json"
    if [ $? -ne 0 ]; then 
        echo "  MLIR:  Analysis (optimized) failed"
        return 1
    fi
    echo "  MLIR:  Analysis (optimized) succeeded"
    return 0
}

analyze() {
    local bench_dir="$1"
    local name="$(basename $1)"
    echo "Analyzing $name"

    if [ ! -d "$bench_dir" ]; then
        echo "  SRC: Benchmark does not exist"
        return 1
    fi

    analyze_llvm "$bench_dir"
    analyze_mlir "$bench_dir"
}

# Process benchmarks
if [ $USE_DYNAMATIC -eq 1 ]; then
    if [ $ALL -eq 1 ]; then
        for name in $DYNAMATIC_DST/*/; do
            analyze "${name%/}"
            echo ""
        done
    else
        for name in "$@"; do
            if [[ $name != --* ]]; then
                analyze "$DYNAMATIC_DST/$name/"
                echo ""
            fi
        done
    fi
fi    
    
if [ $USE_POLYBENCH -eq 1 ]; then
    if [ $ALL -eq 1 ]; then
        for name in $POLYBENCH_DST/*/; do
            analyze "${name%/}"
            echo ""
        done
    else
        for name in "$@"; do
            if [[ $name != --* ]]; then
                analyze "$POLYBENCH_DST/$name/"
                echo ""
            fi
        done
    fi
fi

echo "Done!"
