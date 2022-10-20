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
    local kernel_name=$2
    local llvm_pass="$PWD/build/lib/LLVMStatisticsPass.so"
    local llvm_dir="$bench_dir/llvm"
    local llvm_ir="$llvm_dir/step_4.ll"

    "$LLVM_ANALYZE_BIN" "$llvm_ir" -enable-new-pm=0 -load "$llvm_pass" \
        -ir-stats -kernel $kernel_name -filename "$llvm_dir/stats.json" \
        > /dev/null  
    if [ $? -ne 0 ]; then 
        echo "  LLVM:  Analysis failed"
        return 1
    fi
    echo "  LLVM:  Analysis succeeded"
    return 0
}

analyze_mlir () {
    local bench_dir=$1
    local kernel_name=$2
    local opt="$PWD/build/bin/mlir-analyze"
    local mlir_dir="$bench_dir/mlir"
    local mlir="$mlir_dir/std.mlir"
    local mlir_opt="$mlir_dir/std_opt.mlir"
    
    # Analyze non-optimized code
    "$opt" "$mlir" \
        --ir-stats="kernel=$kernel_name filename=$mlir_dir/stats.json" \
        > /dev/null 
    if [ $? -ne 0 ]; then 
        echo "  MLIR:  Analysis (non-optimized) failed"
    else
        echo "  MLIR:  Analysis (non-optimized) succeeded"
    fi

    # Analyze optimized code
    # "$opt" "$mlir_opt" \
    #     --ir-stats="kernel=$kernel_name filename=$mlir_dir/stats_opt.json" \
    #     > /dev/null 
    # if [ $? -ne 0 ]; then 
    #     echo "  MLIR:  Analysis (optimized) failed"
    #     return 1
    # fi
    # echo "  MLIR:  Analysis (optimized) succeeded"
    return 0
}

analyze() {
    local bench_dir="$1"
    local kernel_name="$2"
    echo "Analyzing $(basename $1)"

    if [ ! -d "$bench_dir" ]; then
        echo "  SRC: Benchmark does not exist"
        return 1
    fi

    analyze_llvm "$bench_dir" $kernel_name
    analyze_mlir "$bench_dir" $kernel_name
}

get_kernel_name() {
    local name=`echo $(basename $1) | sed -r 's/\-/_/g'`
    echo $name
}

# Process benchmarks
if [ $USE_DYNAMATIC -eq 1 ]; then
    if [ $ALL -eq 1 ]; then
        for name in $DYNAMATIC_DST/*/; do
            bench_dir="${name%/}"
            analyze "$bench_dir" $(get_kernel_name $bench_dir)
            echo ""
        done
    else
        for name in "$@"; do
            if [[ $name != --* ]]; then
                bench_dir="$DYNAMATIC_DST/$name"
                analyze "$bench_dir" $(get_kernel_name $bench_dir)
                echo ""
            fi
        done
    fi
fi    
    
if [ $USE_POLYBENCH -eq 1 ]; then
    if [ $ALL -eq 1 ]; then
        for name in $POLYBENCH_DST/*/; do
            bench_dir="${name%/}"
            analyze "$bench_dir" kernel_$(get_kernel_name $bench_dir)
            echo ""
        done
    else
        for name in "$@"; do
            if [[ $name != --* ]]; then
                bench_dir="$POLYBENCH_DST/$name"
                analyze "$bench_dir" kernel_$(get_kernel_name $bench_dir)
                echo ""
            fi
        done
    fi
fi

echo "Done!"
