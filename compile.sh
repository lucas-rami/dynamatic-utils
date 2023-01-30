#!/bin/bash

source ./utils.sh

check_env_variables \
    DYNAMATIC_PATH \
    POLYBENCH_PATH \
    LLVM_CLANG_BIN \
    LLVM_OPT_BIN \
    MLIR_CLANG_BIN \
    POLYGEIST_OPT_BIN \
    POLYMER_OPT_BIN \
    MLIR_OPT_BIN \
    CIRCT_OPT_BIN \
    DYNAMATIC_SRC \
    DYNAMATIC_DST \
    POLYBENCH_SRC \
    POLYBENCH_DST 

# Parse arguments
USE_DYNAMATIC=0
USE_POLYBENCH=0
ALL=0
FORCE=0
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
        "--force")
            FORCE=1
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

compile_llvm () {
    local bench_dir=$1
    local kernel_name=$2
    local name="$(basename $bench_dir)"

    # Check whether LLVM folder already exists in local folder
    local llvm_out="$bench_dir/llvm"
    if [[ $FORCE -eq 0 && -d "$llvm_out" ]] ; then
        echo "[LLVM] Already compiled"
        return 0
    fi
    mkdir -p "$llvm_out"

    # Compile source to LLVM IR
	"$LLVM_CLANG_BIN" -Xclang -disable-O0-optnone -emit-llvm -S \
        -I "$bench_dir" \
        -c "$bench_dir/$name.c" \
        -o $llvm_out/step_0.ll
    if [ $? -ne 0 ]; then 
        echo "[LLVM] Compilation to LLVM IR failed"
        return 1
    else
        echo "[LLVM] Compilation to LLVM IR succeeded"
    fi

    # Apply standard optimizations to LLVM IR
	"$LLVM_OPT_BIN" -mem2reg \
        "$llvm_out/step_0.ll" -S -o "$llvm_out/step_1.ll"
	"$LLVM_OPT_BIN" -loop-rotate -constprop \
        "$llvm_out/step_1.ll" -S -o "$llvm_out/step_2.ll"
	"$LLVM_OPT_BIN" -simplifycfg \
        "$llvm_out/step_2.ll" -S -o "$llvm_out/step_3.ll"
	"$LLVM_OPT_BIN" -die -instcombine -lowerswitch \
        "$llvm_out/step_3.ll" -S -o "$llvm_out/step_4.ll"
    if [ $? -ne 0 ]; then 
        echo "[LLVM] Standard optimization failed"
        return 1
    else
        echo "[LLVM] Standard optimization succeeded"
    fi

    # Apply custom optimizations
	local passes_dir="$DYNAMATIC_PATH/dhls/etc/dynamatic/elastic-circuits/_build/"
    "$LLVM_OPT_BIN" \
        -load "$passes_dir/MemElemInfo/libLLVMMemElemInfo.so" \
        -load "$passes_dir/ElasticPass/libElasticPass.so" \
        -load "$passes_dir/OptimizeBitwidth/libLLVMOptimizeBitWidth.so" \
        -load "$passes_dir/MyCFGPass/libMyCFGPass.so" \
        -polly-process-unprofitable -mycfgpass \
        "$llvm_out/step_4.ll" -S "-cfg-outdir=$llvm_out" \
            "-kernel=$kernel_name" > /dev/null 2>&1

    if [ ! -f "$llvm_out/$name.dot" ]; then
        echo "[LLVM] Creation of Graphviz visualization failed"
        return 1
    fi

    # Convert to PNG
    dot -Tpng "$llvm_out/$name.dot" > "$llvm_out/$name.png"
    echo "[LLVM] Creation of Graphviz visualization succeeded"
    return 0
}

compile_mlir_internal() {
    local function_name=$1
    local f_src=$2
    local f_affine=$3
    local f_affine_opt=$4
    local f_std=$5

    # Include path
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"

    # Passes to convert from scf to std
    local to_std_passes="-convert-scf-to-cf -canonicalize -cse -sccp \
        -symbol-dce -control-flow-sink -loop-invariant-code-motion \
        -canonicalize"

    # source code -> affine dialect 
    "$MLIR_CLANG_BIN" "$f_src" \
        -I "$include" -function=$function_name -S -O3 -raise-scf-to-affine \
        -memref-fullrank \
        > "$f_affine"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] source code -> affine dialect failed"
        return 1
    fi

    # affine dialect -> optimized affine dialect 
    "$POLYGEIST_OPT_BIN" "$f_affine" \
        -mem2reg \
        > "$f_affine_opt"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] affine dialect -> optimized affine dialect failed"
        return 1
    fi

    # optimized affine dialect -> standard dialect
    "$MLIR_OPT_BIN" "$f_affine_opt" \
        -lower-affine $to_std_passes \
        > "$f_std"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] optimized affine dialect -> standard dialect failed"
        return 1
    fi

    return 0
}

compile_handshake() {
    local kernel_name=$1
    local f_std=$2
    local f_handshake=$3
    local f_handshake_opt=$4
    local f_handshake_dot=$5
    local f_handshake_png=$6

    # standard dialect -> handshake dialect
    "$CIRCT_OPT_BIN" "$f_std" \
        -allow-unregistered-dialect --flatten-memref --flatten-memref-calls \
        --lower-std-to-handshake \
        > "$f_handshake"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] standard dialect -> handshake dialect failed"
        return 1
    fi

    # handshake dialect -> optimized handshake dialect
    "$CIRCT_OPT_BIN" "$f_handshake" \
        -allow-unregistered-dialect --handshake-materialize-forks-sinks \
        --canonicalize \
        > "$f_handshake_opt"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] handshake dialect -> optimized handshake dialect failed"
        return 1
    fi
        
    # Create DOT graph at handshake level
    "$CIRCT_OPT_BIN" "$f_handshake_opt" \
        -allow-unregistered-dialect --handshake-print-dot \
        > /dev/null 2>&1 
    if [ $? -ne 0 ]; then
        # DOT gets generated in script directory, remove it 
        rm "$kernel_name.dot" 
        
        echo "[MLIR] Creation of handshake Graphviz visualization failed"
        return 1
    else
        # DOT gets generated in script directory, move it to the right place
        mv "$kernel_name.dot" "$f_handshake_dot"

        # Convert DOT graph to PNG
        dot -Tpng "$f_handshake_dot" > "$f_handshake_png"
        echo "[MLIR] Creation of handshake Graphviz visualization succeeded"
    fi

    return 0
}

compile_mlir () {
    local bench_dir=$1
    local kernel_name=$2
    local name="$(basename $bench_dir)"

    # Check whether MLIR folder already exists in local folder
    local mlir_out="$bench_dir/mlir"
    if [[ $FORCE -eq 0 && -d "$mlir_out" ]] ; then
        echo "[MLIR] Already compiled"
        return 0
    fi
    mkdir -p "$mlir_out"

    # C source file
    local f_src="$bench_dir/$name.c"

    # ---- Compile all functions ---- #

    # Files
    local f_affine="$mlir_out/affine.mlir"
    local f_affine_opt="$mlir_out/affine_opt.mlir"
    local f_std="$mlir_out/std.mlir"

    # Compile
    compile_mlir_internal "*" "$f_src" "$f_affine" "$f_affine_opt" "$f_std"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] Compilation of all functions failed" 
    else 
        echo "[MLIR] Compilation of all functions succeeded" 
    fi

    # ---- Compile kernel function only ---- #

    # Files
    local f_affine_fun="$mlir_out/affine_fun.mlir"
    local f_affine_opt_fun="$mlir_out/affine_opt_fun.mlir"
    local f_std_fun="$mlir_out/std_fun.mlir"
    local f_dot="$mlir_out/$name.dot"
    local f_png="$mlir_out/$name.png"

    # Compile
    compile_mlir_internal "$kernel_name" "$f_src" "$f_affine_fun" \
        "$f_affine_opt_fun" "$f_std_fun"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] Compilation of kernel function only failed" 
        return 1
    else 
        echo "[MLIR] Compilation of kernel function only succeeded" 
    fi

    # Create DOT graph
    "$MLIR_OPT_BIN" "$f_std_fun" -view-op-graph > /dev/null 2> "$f_dot"
    if [ $? -ne 0 ]; then 
        echo "[MLIR] Creation of Graphviz visualization failed"
        return 1
    else
        # Convert DOT graph to PNG
        dot -Tpng "$f_dot" > "$f_png"
        echo "[MLIR] Creation of Graphviz visualization succeeded"
    fi

    # Compile to handshake
    compile_handshake "$kernel_name" "$f_std_fun" "$mlir_out/handshake.mlir" \
        "$mlir_out/handshake_opt.mlir" "$mlir_out/handshake.dot" \
        "$mlir_out/handshake.png"

    return 0

    #### Compile WITH polyhedral optimization
    # f_src -> f_affine -> f_affine_opt -> f_std_opt

    # # Use Polygeist to compile to affine dialect
    # "$MLIR_CLANG_BIN" "$f_src" -I "$bench_dir" -I "$include" -function=* -S \
    #     -O3 -raise-scf-to-affine -memref-fullrank > "$f_affine"
    # if [ $? -ne 0 ]; then 
    #     echo "  MLIR: Failed during compilation to affine dialect, abort"
    #     return 1
    # fi

    # # Use Polymer to optimize affine dialect
    # "$POLYMER_OPT_BIN" "$f_affine" -reg2mem -extract-scop-stmt \
    #     -pluto-opt -allow-unregistered-dialect > "$f_affine_opt"
    # if [ $? -ne 0 ]; then 
    #     echo "  MLIR: Failed during affine optimization, abort"
    #     return 1
    # fi

    # # Lower optimized affine to standard
    # "$MLIR_OPT_BIN" "$f_affine_opt" -lower-affine -inline \
    #     $to_std_passes > "$f_std_opt"
    # if [ $? -ne 0 ]; then 
    #     echo "  MLIR: Failed during lowering to standard dialect from \
    #         optimized code, abort"
    #     return 1
    # fi
}

process_benchmark_dynamatic () {
    local name=$1

    echo "---- Compiling $name ----"

    # Copy benchmark from dynamatic folder to local folder
    copy_src "$DYNAMATIC_SRC/$name/src" "$DYNAMATIC_DST/$name" "$name" "cpp"
    if [ $? -ne 0 ]; then 
        return 1
    fi

    # Compile with LLVM
    compile_llvm "$DYNAMATIC_DST/$name" "$name"

    # Compile with MLIR
    compile_mlir "$DYNAMATIC_DST/$name" "$name"
    
    echo "---- Done! ----"
    echo ""
    return 0
}


process_benchmark_polybench () {
    local bench_subpath=$1
    
    local src_dir="$(dirname $bench_subpath)"
    local name="$(basename $bench_subpath .c)"

    echo "---- Compiling $name ----"
    # Copy benchmark from Polybench folder to local folder
    copy_src "$POLYBENCH_SRC/$src_dir" "$POLYBENCH_DST/$name" "$name" "c" 
    if [ $? -ne 0 ]; then 
        return 1
    fi

    # Also copy polybench.h to the benchmark directory
    cp "$POLYBENCH_SRC/utilities/polybench.h" "$POLYBENCH_DST/$name"
    if [ $? -ne 0 ]; then
        echo "[SRC] Failed to copy polybench.h"
        return 1
    fi
    
    # Replace #include <polybench.h> by #include <polybench.h> in source 
    sed -i 's/<polybench.h>/"polybench.h"/g' "$POLYBENCH_DST/$name/$name.c" 

    # Make functions non-static so that they aren't inlined by MLIR
    sed -i 's/^static//g' "$POLYBENCH_DST/$name/$name.c" 

    # Replace - with _ in kernel names
    local kernel_name="kernel_`echo $name | sed -r 's/\-/_/g'`"

    # Compile with LLVM
    compile_llvm "$POLYBENCH_DST/$name" "$kernel_name"

    # Compile with MLIR
    compile_mlir "$POLYBENCH_DST/$name" "$kernel_name"
    
    echo "---- Done! ----"
    echo ""
    return 0
}

# Process benchmarks
if [ $USE_DYNAMATIC -eq 1 ]; then
    if [ $ALL -eq 1 ]; then
        for name in $DYNAMATIC_SRC/*/; do
            bname="$(basename $name)"
            process_benchmark_dynamatic "$bname"
            echo ""
        done
    else
        for name in "$@"; do
            if [[ $name != --* ]]; then
                process_benchmark_dynamatic "$name"
                echo ""
            fi
        done
    fi
fi    
    
if [ $USE_POLYBENCH -eq 1 ]; then
    if [ $ALL -eq 1 ]; then
        for name in `cat "$POLYBENCH_PATH/utilities/benchmark_list"`; do
            process_benchmark_polybench "$name"
            echo ""
        done
    else
        for name in "$@"; do
            if [[ $name != --* ]]; then
                path=`cat "$POLYBENCH_PATH/utilities/benchmark_list"| grep $name`
                process_benchmark_polybench "$path"
                echo ""
            fi
        done
    fi
fi

echo "---- All done! ----"
echo ""
