#!/bin/bash

# ===- runner.sh - Run benchmarks through various flows --------*- Bash -*-=== #
# 
# TODO
# 
# ===----------------------------------------------------------------------=== #

# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source utils.sh

# Check that required environment variables exist
check_env_variables \
    DOT2VHDL_BIN \
    DYNAMATIC_OPT_BIN \
    LEGACY_DYNAMATIC_ROOT
    LLVM_CLANG_BIN \
    LLVM_OPT_BIN \
    MLIR_OPT_BIN \
    POLYGEIST_CLANG_BIN \
    POLYGEIST_PATH \
    TESTSUITE_DYNAMATIC_PATH \
    TESTSUITE_FPL22_PATH

# Testsuite-related variables 
PARSE_TESTSUITE=0
TESTSUITE_DYNAMATIC="dynamatic"
TESTSUITE_FPL22="fpl22"
TESTSUITE_PATH="$TESTSUITE_DYNAMATIC"

# Flow-related variables 
PARSE_FLOW=0
FLOW_DYNAMATIC="dynamatic"
FLOW_LEGACY="legacy"
FLOW_BRIDGE="bridge"
FLOW="$FLOW_DYNAMATIC"

# Simulation flag
SIMULATE=0

# Explicit list of benchmarks to build
TO_BUILD=()

# Display list of possible options and exit.
print_help_and_exit () {
    echo -e \
"./$0 [options] <bench_name>*

List of options:
  --testsuite <suite name>      : run a specific testsuite (dynamatic, fpl22)
  --flow <flow name>            : run a specific flow (dynamatic, legacy, bridge)
  --simulate                    : enable VHDL simulation/verification with Modelsim
  --help | -h                   : display this help message
"
    exit $2
}

# Copies a .cpp file to a .c file (useful to avoid annoying C++ function name
# mangling).
cpp_to_c() {
    local name="$(basename $1)"
    cp "$1/src/$name.cpp" "$1/src/$name.c"
}

# Deletes the .c file that is created by a call to cpp_to_c with the same
# argument.
delete_c() {
    local name="$(basename $1)"
    rm "$1/src/$name.c"
}

# Compile down to VHDL and simulate.
simulate() {
    local bench_path="$1"
    local flow_path="$bench_path/$FLOW"
    local name="$(basename $bench_path)"
    local out="$flow_path/sim"

    # Generated files
    local f_vhdl="$flow_path/$name.vhd"
    local f_report="$out/report.txt"

    # Generated directories
    local sim_src_dir="$out/C_SRC"
    local sim_vhdl_src_dir="$out/VHDL_SRC"
    local sim_verify_dir="$out/HLS_VERIFY"

    # Convert DOT graph to VHDL
    "$DOT2VHDL_BIN" "$flow_path/$name" > /dev/null
    echo_status "Failed to convert DOT to VHDL" "Converted DOT to VHDL"
    if [[ $? -ne 0 ]]; then
        return $?
    fi

    # Remove useless TCL files
    rm -f "$flow_path"/*.tcl
    
    # Create simulation directories
    mkdir -p "$out/C_OUT" "$sim_src_dir" "$sim_verify_dir" \
        "$out/INPUT_VECTORS" "$out/VHDL_OUT" "$sim_vhdl_src_dir"
    echo "[INFO] Created simulation directories"

    # Move VHDL module and copy VHDL components to dedicated folder
    mv "$f_vhdl" "$sim_vhdl_src_dir"
    cp "$LEGACY_DYNAMATIC_ROOT"/components/*.vhd "$sim_vhdl_src_dir"
    echo "[INFO] Copied VHDL components to VHDL_SRC directory"

    # Copy sources to dedicated folder
    cp "$bench_path/src/$name.cpp" "$sim_src_dir/$name.c" 
    cp "$bench_path/src/$name.h" "$sim_src_dir"
    echo "[INFO] Copied source to C_SRC directory"

    # Simulate and verify design
    echo "[INFO] Launching simulation"
    cd "$sim_verify_dir"
    "$LEGACY_DYNAMATIC_ROOT/Regression_test/hls_verifier/HlsVerifier/build/hlsverifier" \
         cover -aw32 "$sim_src_dir/$name.c" "$sim_src_dir/$name.c" $name \
         > "$f_report"
    local sim_ret=$?
    cd - > /dev/null
    return $sim_ret
}

# Dynamatic flow.
dynamatic () {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local out="$bench_path/$FLOW"

    # Remove output directory if it exists and recreate it 
    rm -rf "$out"
    mkdir -p "$out"

    # Generated files
    local f_scf="$out/scf.mlir"
    local f_affine="$out/affine.mlir"
    local f_affine_mem="$out/affine_mem.mlir"
    local f_std="$out/std.mlir"
    local f_handshake="$out/handshake.mlir"
    local f_netlist="$out/netlist.mlir"
    local f_dot="$out/$name.dot"
    local f_png="$out/$name.png"

    # source code -> affine dialect
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    "$POLYGEIST_CLANG_BIN" "$bench_path/src/$name.c" -I "$include" \
        --function="$name" -S -O3 --memref-fullrank --raise-scf-to-affine \
        > "$f_affine" 2>/dev/null
    exit_on_fail "Failed source -> affine conversion" "Lowered to affine"
    
    # memory analysis 
    "$DYNAMATIC_OPT_BIN" "$f_affine" --allow-unregistered-dialect \
        --name-memory-ops --analyze-memory-accesses > "$f_affine_mem"
    exit_on_fail "Failed memory analysis" "Passed memory analysis"

    # affine dialect -> scf dialect
    "$DYNAMATIC_OPT_BIN" "$f_affine_mem" --allow-unregistered-dialect \
        --lower-affine-to-scf > "$f_scf"
    exit_on_fail "Failed affine -> scf conversion" "Lowered to scf"

    # scf dialect -> standard dialect
    "$MLIR_OPT_BIN" "$f_scf" --allow-unregistered-dialect --convert-scf-to-cf \
        --canonicalize --cse --sccp --symbol-dce --control-flow-sink \
        --loop-invariant-code-motion --canonicalize > "$f_std"
    exit_on_fail "Failed scf -> std conversion" "Lowered to std"

    # standard dialect -> handshake dialect
    "$DYNAMATIC_OPT_BIN" "$f_std" --allow-unregistered-dialect \
        --flatten-memref-row-major --flatten-memref-calls --push-constants \
        --lower-std-to-handshake-fpga18="id-basic-blocks" > "$f_handshake"
    exit_on_fail "Failed std -> handshake conversion" "Lowered to handshake"

    # Create DOT graph
    "$DYNAMATIC_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
        --export-dot > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        # DOT gets generated in script directory, remove it 
        rm "$name.dot" 
        echo "[ERROR] Failed to create DOT graph"
        return 1
    fi
    echo "[INFO] Created DOT graph"

    # DOT gets generated in script directory, move it to the right place
    mv "$name.dot" "$f_dot"

    # Convert DOT graph to PNG
    dot -Tpng "$f_dot" > "$f_png"
    exit_on_fail "Failed to convert DOT to PNG" "Converted DOT to PNG"
    return 0
}

# Path to build folders containing legacy Dynamatic object files
ELASTIC_BUILD_PATH="$LEGACY_DYNAMATIC_ROOT/elastic-circuits/_build"
FREQUENCY_COUNTER_PATH="$ELASTIC_BUILD_PATH/FrequencyCounterPass"
FREQUENCY_DATA_PATH="$ELASTIC_BUILD_PATH/FrequencyDataGatherPass"

# Legacy flow.
legacy () {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local out="$bench_path/$FLOW"

    # Remove output directory if it exists and recreate it 
    rm -rf "$out"
    mkdir -p "$out"

    # Generated files
    local f_ir="$out/ir.ll"
    local f_ir_opt="$out/ir_opt.ll"
    local f_ir_opt_obj="$out/ir_opt.o"
    local f_dot="$out/$name.dot"
    local f_dot_bb="$out/${name}_bbgraph.dot"
    local f_png="$out/$name.png"
    local f_png_bb="$out/${name}_bbgraph.png"

    # source code -> LLVM IR
	"$LLVM_CLANG_BIN" -Xclang -disable-O0-optnone -emit-llvm -S \
        -I "$bench_path" \
        -c "$bench_path/src/$name.c" \
        -o $out/ir.ll
    exit_on_fail "Failed to compile to LLVM IR" "Compiled to LLVM IR"
    
    # LLVM IR -> standard optimized LLVM IR
	"$LLVM_OPT_BIN" -mem2reg -loop-rotate -constprop -simplifycfg -die \
        -instcombine -lowerswitch $f_ir -S -o "$f_ir_opt"
    exit_on_fail "Failed to apply standard optimization" "Applied standard optimization"

    # Run frequency analysis
	"$LLVM_CLANG_BIN" -fPIC -Xclang -load -Xclang \
        "$FREQUENCY_COUNTER_PATH/libFrequencyCounterPass.so" -c "$f_ir_opt" \
        -o "$f_ir_opt_obj"
	"$LLVM_CLANG_BIN" -fPIC "$f_ir_opt_obj" \
        "$FREQUENCY_COUNTER_PATH/log_FrequencyCounter.o"
	./a.out
	rm a.out
	"$LLVM_CLANG_BIN" -Xclang -load -Xclang \
        "$FREQUENCY_DATA_PATH/libFrequencyDataGatherPass.so" $f_ir_opt"" -S
	rm *.s

    # standard optimized LLVM IR -> elastic circuit
    "$LLVM_OPT_BIN" \
        -load "$ELASTIC_BUILD_PATH/MemElemInfo/libLLVMMemElemInfo.so" \
        -load "$ELASTIC_BUILD_PATH/ElasticPass/libElasticPass.so" \
        -load "$ELASTIC_BUILD_PATH/OptimizeBitwidth/libLLVMOptimizeBitWidth.so" \
        -load "$ELASTIC_BUILD_PATH/MyCFGPass/libMyCFGPass.so" \
        -polly-process-unprofitable -mycfgpass -S \
        -simple-buffers=true -use-lsq=false \
        "$f_ir_opt" \
        "-cfg-outdir=$out" > /dev/null 2>&1

    # Remove temporary build files
    rm *_freq.txt mapping.txt out.txt

    # Can't check the return value here, as this often crashes right after
    # generating the files we want
    if [[ ! -f "$out/${name}_graph.dot" || ! -f "$out/${name}_bbgraph.dot" ]]; then
        echo "[ERROR] Failed to generate DOTs"
        return 1
    fi
    echo "[INFO] Applied elastic pass"

    # Rename DOT file
    mv "$out/${name}_graph.dot" "$f_dot"

    # Convert DOTs to PNG
    dot -Tpng "$f_dot" > "$f_png"
    echo_status "Failed to convert DOT to PNG" "Converted DOT to PNG"
    dot -Tpng "$f_dot_bb" > "$f_png_bb"
    echo_status "Failed to convert DOT (BB) to PNG" "Converted DOT (BB) to PNG"

    return 0
}

# Bridge flow.
bridge () {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local out="$bench_path/$FLOW"

    # Remove output directory if it exists and recreate it 
    rm -rf "$out"
    mkdir -p "$out"

    # Generated files
    local f_affine="$out/affine.mlir"
    local f_affine_mem="$out/affine_mem.mlir"
    local f_scf="$out/scf.mlir"
    local f_std="$out/std.mlir"
    local f_cfg_dot="$out/${name}_bbgraph.dot"
    local f_cfg_png="$out/${name}_bbgraph.png"
    local f_llvm_ir="$out/llvm.ll"
    local f_handshake="$out/handshake.mlir"
    local f_handshake_ready="$out/handshake_ready.mlir"
    local f_netlist="$out/netlist.mlir"
    local f_dfg_dot="$out/$name.dot"
    local f_dfg_png="$out/$name.png"

    # source code -> affine dialect
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    "$POLYGEIST_CLANG_BIN" "$bench_path/src/$name.c" -I "$include" \
        --function="$name" -S -O3 --memref-fullrank --raise-scf-to-affine \
        > "$f_affine" 2>/dev/null
    exit_on_fail "Failed source -> affine conversion" "Lowered to affine"
    
    # memory analysis 
    "$DYNAMATIC_OPT_BIN" "$f_affine" --allow-unregistered-dialect \
        --name-memory-ops --analyze-memory-accesses > "$f_affine_mem"
    exit_on_fail "Failed memory analysis" "Passed memory analysis"

    # affine dialect -> scf dialect
    "$DYNAMATIC_OPT_BIN" "$f_affine_mem" --allow-unregistered-dialect \
        --lower-affine-to-scf > "$f_scf"
    exit_on_fail "Failed affine -> scf conversion" "Lowered to scf"

    # scf dialect -> standard dialect
    "$MLIR_OPT_BIN" "$f_scf" --allow-unregistered-dialect --convert-scf-to-cf \
        --canonicalize --cse --sccp --symbol-dce --control-flow-sink \
        --loop-invariant-code-motion --canonicalize > "$f_std"
    exit_on_fail "Failed scf -> std conversion" "Lowered to std"

    # std dialect -> CFG graph
    "$DYNAMATIC_OPT_BIN" "$f_std" --allow-unregistered-dialect --export-cfg \
        > /dev/null
    echo_status "Failed to create CFG DOT" "Created CFG DOT"
    if [ $? -ne 0 ]; then
        rm "${name}_bbgraph.dot" 
        exit 1
    fi
    mv "${name}_bbgraph.dot" "$f_cfg_dot"

    # Convert DOT graph to PNG
    dot -Tpng "$f_cfg_dot" > "$f_cfg_png"
    exit_on_fail "Failed to convert CFG DOT to PNG" "Converted CFG DOT to PNG"

    # standard dialect -> handshake dialect
    "$DYNAMATIC_OPT_BIN" "$f_std" --allow-unregistered-dialect \
        --arith-optimize-area --canonicalize --flatten-memref-row-major \
        --flatten-memref-calls --push-constants \
        --lower-std-to-handshake-fpga18="id-basic-blocks" \
        > "$f_handshake"
    exit_on_fail "Failed std -> handshake conversion" "Lowered to handshake"

    # handshake dialect -> legacy-compatible handshake
    "$DYNAMATIC_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-prepare-for-legacy --handshake-infer-basic-blocks \
        > "$f_handshake_ready"
    exit_on_fail "Failed handshake -> legacy-compatible handshake" \
        "Lowered to legacy-compatible handshake"

    # Create DOT graph
    "$DYNAMATIC_OPT_BIN" "$f_handshake_ready" --allow-unregistered-dialect \
        --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
        --handshake-insert-buffers="buffer-size=2 strategy=cycles" \
        --handshake-infer-basic-blocks \
        --export-dot="legacy pretty-print=false" > /dev/null
    echo_status "Failed to create DFG DOT" "Created DFG DOT"
    if [ $? -ne 0 ]; then
        rm "${name}.dot" 
        exit 1
    fi
    mv "$name.dot" "$f_dfg_dot"

    # Convert DOT graph to PNG
    dot -Tpng "$f_dfg_dot" > "$f_dfg_png"
    exit_on_fail "Failed to convert DFG DOT to PNG" "Converted DFG DOT to PNG"

    return 0
}

# Run the specified flow for a benchmark.
benchmark () {
    local bench_path=$1
    local name="$(basename $path)"

    echo_section "Compiling $name"

    # Check that a source file exists at the expected location
    local src_path="$bench_path/src/$name.cpp"
    if [[ ! -f "$src_path" ]]; then
        echo "[ERROR] No source file exists at \"$src_path\", skipping this benchmark."
    else
        local compile_ret=1

        # Run the appropriate compile flow
        cpp_to_c "$bench_path"
        case "$FLOW" in 
            $FLOW_DYNAMATIC)
                dynamatic "$bench_path"
                compile_ret=$?
            ;;
            $FLOW_LEGACY)
                legacy "$bench_path"
                compile_ret=$?
            ;;
            $FLOW_BRIDGE)
                bridge "$bench_path"
                compile_ret=$?
            ;;
        esac
        delete_c "$bench_path"
        
        if [[ $compile_ret -eq 0 ]]; then
            echo "[INFO] Compilation succeeded!"
            if [[ $SIMULATE -eq 1 ]]; then
                if [[ $FLOW == $FLOW_DYNAMATIC ]]; then
                    echo "[INF0] Simulation is not yet supported on the dynamatic flow"
                else
                    # Simulate the design
                    echo ""
                    simulate "$bench_path"
                    if [[ $? -eq 0 ]]; then
                        echo "[INFO] Simulation succeeded!"
                    else
                        echo "[ERROR] Simulation failed!"
                    fi
                fi
            fi
        else
            echo "[ERROR] Compilation failed!"
        fi
    fi
    echo ""
}

echo_section "Parsing arguments"

# Parse all arguments
for arg in "$@"; 
do
    if [[ $PARSE_TESTSUITE -eq 1 ]]; then
        # Parse the name of the testsuite to use
        case "$arg" in 
            $TESTSUITE_DYNAMATIC)
                TESTSUITE_PATH="$TESTSUITE_DYNAMATIC_PATH"
                ;;
            $TESTSUITE_FPL22)
                TESTSUITE_PATH="$TESTSUITE_FPL22_PATH"
                ;;
            *)
                echo "Unknown testsuite \"$arg\", choices are"
                echo "  1. dynamatic (default)"
                echo "  2. fpl22"
                echo "Aborting"
                exit 1
                ;;
        esac
        echo "[INFO] Setting testsuite to \"$arg\""
        PARSE_TESTSUITE=0

    elif [[ $PARSE_FLOW -eq 1 ]]; then
        # Parse the name of the flow to use
        case "$arg" in 
            $FLOW_DYNAMATIC)
                FLOW="$FLOW_DYNAMATIC"
                ;;
            $FLOW_LEGACY)
                FLOW="$FLOW_LEGACY"
                ;;
            $FLOW_BRIDGE)
                FLOW="$FLOW_BRIDGE"
                ;;
            *)
                echo "Unknown flow \"$arg\", choices are"
                echo "  1. dynamatic (default)"
                echo "  2. legacy"
                echo "  3. bridge"
                echo "Aborting"
                exit 1
                ;;
        esac
        echo "[INFO] Setting flow to \"$arg\""
        PARSE_FLOW=0
    
    else
        # Parse the next argument
        case "$arg" in 
            "--testsuite")
                PARSE_TESTSUITE=1
                ;;
            "--flow")
                PARSE_FLOW=1
                ;;
            "--simulate")
                SIMULATE=1
                echo "[INFO] Enabling simulation"
                ;;
            "--help" | "-h")
                echo "[INFO] Printing help"
                print_help_and_exit $0
                ;;
            *)
                TO_BUILD+=("$arg")
                echo "[INFO] Registering benchmark \"$arg\" in the build list"
                ;;
        esac
    fi
done
echo ""

# Build benchmarks
if [ ${#TO_BUILD[@]} -eq 0 ]; then
    echo "[INFO] No benchmark provided, building all benchmarks in testsuite..."
    for path in $TESTSUITE_PATH/*/; do
        benchmark "${path%/}"
    done
else
    for path in ${TO_BUILD[@]}; do
        benchmark "$TESTSUITE_PATH/$path"
    done
fi

echo "[INFO] All done!"
