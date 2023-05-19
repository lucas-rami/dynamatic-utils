#!/bin/bash

# ===- runner.sh - Run benchmarks through various flows --------*- Bash -*-=== #
# 
# This script facilitates the compilation/simulation of benchmarks from multiple
# testsuites using multiple DHLS flows. Run the script with the --help flag to
# see the list of available options.
# 
# ===----------------------------------------------------------------------=== #

# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source utils.sh

# Check that required environment variables exist
check_env_variables \
    BUFFERS_BIN \
    CIRCT_HANDSHAKE_RUNNER_BIN \
    DOT2VHDL_BIN \
    DYNAMATIC_OPT_BIN \
    LEGACY_DYNAMATIC_ROOT \
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

# Flags
SIMULATE=0
SYNTHESIZE=0
SMART_BUFFERS=0
NO_COMPILE=0

# Explicit list of benchmarks to build
TO_BUILD=()

# Path to build folders containing legacy Dynamatic object files
ELASTIC_BUILD_PATH="$LEGACY_DYNAMATIC_ROOT/elastic-circuits/_build"
FREQUENCY_COUNTER_PATH="$ELASTIC_BUILD_PATH/FrequencyCounterPass"
FREQUENCY_DATA_PATH="$ELASTIC_BUILD_PATH/FrequencyDataGatherPass"

# Display list of possible options and exit.
print_help_and_exit () {
    echo -e \
"$0
    [--testsuite <suite-name>] [--flow <flow-name>] 
    [--no-compile] [--smart-buffers] [--simulate] [--synthesize]
    [--help|-h] 
    [<bench-name> ]...

List of options:
  --testsuite <suite-name>      : run a specific testsuite (dynamatic [default], fpl22)
  --flow <flow-name>            : run a specific flow (dynamatic [default], legacy, bridge)
  --no-compile                  : do not re-compile benchmarks, only use cached DOTs
  --smart-buffers               : enable smart buffer placement (instead of stupid buffer placement)
  --simulate                    : enable VHDL simulation/verification with Modelsim
  --synthesize                  : enable VHDL synthesization with Vivado
  <bench-name>...               : run the selected flow only on specific benchmarks from the selected testsuite   
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
    local name="$(basename $bench_path)"
    local out="$bench_path/$FLOW"

    # Generated directories
    local sim_dir="$out/sim"
    local sim_src_dir="$sim_dir/C_SRC"
    local sim_vhdl_src_dir="$sim_dir/VHDL_SRC"
    local sim_verify_dir="$sim_dir/HLS_VERIFY"

    # Generated files
    local f_vhdl="$out/$name.vhd"
    local f_report="$sim_dir/report.txt"

    # Remove output directories if they exist and recreate them 
    rm -rf "$sim_dir"
    mkdir -p "$sim_dir"

    # Convert DOT graph to VHDL
    "$DOT2VHDL_BIN" "$out/$name" > /dev/null
    echo_status "Failed to convert DOT to VHDL" "Converted DOT to VHDL"
    if [[ $? -ne 0 ]]; then
        return $?
    fi

    # Remove useless TCL files
    rm -f "$out"/*.tcl
    
    # Create simulation directories
    mkdir -p "$sim_dir/C_OUT" "$sim_src_dir" "$sim_verify_dir" \
        "$sim_dir/INPUT_VECTORS" "$sim_dir/VHDL_OUT" "$sim_vhdl_src_dir"
    
    # Move VHDL module and copy VHDL components to dedicated folder
    mv "$f_vhdl" "$sim_vhdl_src_dir"
    cp "$LEGACY_DYNAMATIC_ROOT"/components/*.vhd "$sim_vhdl_src_dir"

    # Copy sources to dedicated folder
    cp "$bench_path/src/$name.cpp" "$sim_src_dir/$name.c" 
    cp "$bench_path/src/$name.h" "$sim_src_dir"

    # Simulate and verify design
    echo "[INFO] Launching Modelsim simulation"
    cd "$sim_verify_dir"
    "$LEGACY_DYNAMATIC_ROOT/Regression_test/hls_verifier/HlsVerifier/build/hlsverifier" \
         cover -aw32 "$sim_src_dir/$name.c" "$sim_src_dir/$name.c" $name \
         > "$f_report"
    local sim_ret=$?
    cd - > /dev/null
    return $sim_ret
}

gen_synth_script() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local out="$bench_path/$FLOW"
    local synth_dir="$out/synth"

    # Generated files
    local f_script="$synth_dir/synthesize.tcl"
    local f_period="$synth_dir/period_4.xdc"
    local f_utilization_syn="$synth_dir/utilization_post_syn.rpt"
    local f_timing_syn="$synth_dir/timing_post_syn.rpt"
    local f_utilization_pr="$synth_dir/utilization_post_pr.rpt"
    local f_timing_pr="$synth_dir/timing_post_pr.rpt"

    echo -e \
"set_param general.maxThreads 8
read_vhdl -vhdl2008 [glob $synth_dir/hdl/*.vhd]
read_xdc "$f_period"
synth_design -top $name -part xc7k160tfbg484-2 -no_iobuf -mode out_of_context
report_utilization > $f_utilization_syn
report_timing > $f_timing_syn
opt_design
place_design
phys_opt_design
route_design
phys_opt_design
report_utilization > $f_utilization_pr
report_timing > $f_timing_pr
exit" > "$f_script"

    echo -e \
"create_clock -name clk -period 4.000 -waveform {0.000 2.000} [get_ports clk]
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports clk]

#set_input_delay 0 -clock CLK  [all_inputs]
#set_output_delay 0 -clock CLK [all_outputs]" > "$f_period"

    return 0
}

synthesize() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local out="$bench_path/$FLOW"

    # Generated directories
    local synth_dir="$out/synth"
    local hdl_dir="$synth_dir/hdl"

    # Generated files
    local f_tcl="$synth_dir/synthesize.tcl"

    # Remove output directory if it exists and recreate it 
    rm -rf "$synth_dir"
    mkdir -p "$synth_dir"

    # Convert DOT graph to VHDL
    "$DOT2VHDL_BIN" "$out/$name" > /dev/null
    echo_status "Failed to convert DOT to VHDL" "Converted DOT to VHDL"
    if [[ $? -ne 0 ]]; then
        return $?
    fi

    # Copy all synthesizable components to specific folder for Vivado
    mkdir -p "$hdl_dir"
    mv "$out/$name.vhd" "$hdl_dir"
    cp "$LEGACY_DYNAMATIC_ROOT"/components/*.vhd "$hdl_dir"

    # Generate synthesization scripts
    gen_synth_script $1
    exit_on_fail "Failed to create synthesization scripts" "Created synthesization scripts"

    echo "[INFO] Launching Vivado synthesization"
    vivado -mode tcl -source "$f_tcl" > /dev/null
    local ret=$?
    rm -f *.jou *.log
    return $ret
}

smart_buffers() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local out="$bench_path/$FLOW"

    # Generated files
    local f_dfg_buf_dot="$out/${name}_graph_buf.dot"
    local f_dfg_buf_png="$out/${name}_graph_buf.png"
    local f_cfg_buf_dot="$out/${name}_bbgraph_buf.dot"
    local f_cfg_buf_png="$out/${name}_bbgraph_buf.png"
    local f_report="$out/buffers/report.txt"

    # Generated directories
    local buffers="$out/buffers"

    # Remove output directory if it exists and recreate it 
    rm -rf "$buffers"
    mkdir -p "$buffers"

    # Run smart buffer placement
    echo "[INFO] Placing smart buffers"
    "$BUFFERS_BIN" buffers -filename="$out/$name" -period=4 -model_mode=mixed \
        -solver=gurobi_cl > $f_report
    echo_status "Failed to place smart buffers" "Placed smart buffers"
    if [[ $? -ne 0 ]]; then
        rm -f *.sol *.lp *.txt 
        return 1 
    fi

    # Move all generated files to output folder 
    mv *.log *.sol *.lp *.txt "$buffers" > /dev/null

    # Convert bufferized DFG DOT graph to PNG
    dot -Tpng "$f_dfg_buf_dot" > "$f_dfg_buf_png"
    echo_status "Failed to convert bufferized DFG DOT to PNG" "Converted bufferized DFG DOT to PNG"

    # Convert bufferized CFG DOT graph to PNG
    dot -Tpng "$f_cfg_buf_dot" > "$f_cfg_buf_png"
    echo_status "Failed to convert bufferized CFG DOT to PNG" "Converted bufferized CFG DOT to PNG"

    # Rename unbufferized DOTS
    mv "$out/$name.dot" "$out/${name}_nobuf.dot" 
    mv "$out/$name.png" "$out/${name}_nobuf.png" 
    mv "$out/${name}_bbgraph.dot" "$out/${name}_bbgraph_nobuf.dot" 
    mv "$out/${name}_bbgraph.png" "$out/${name}_bbgraph_nobuf.png" 

    # Rename bufferized DOTs
    mv "$f_dfg_buf_dot" "$out/$name.dot" 
    mv "$f_dfg_buf_png" "$out/$name.png" 
    mv "$f_cfg_buf_dot" "$out/${name}_bb_graph.dot"

    return $ret
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
        "$FREQUENCY_DATA_PATH/libFrequencyDataGatherPass.so" "$f_ir_opt" -S
	rm *.s

    # standard optimized LLVM IR -> elastic circuit
    local buffers=""
    if [[ $SMART_BUFFERS -eq 0 ]]; then
        buffers="-simple-buffers=true"
    fi
    "$LLVM_OPT_BIN" \
        -load "$ELASTIC_BUILD_PATH/MemElemInfo/libLLVMMemElemInfo.so" \
        -load "$ELASTIC_BUILD_PATH/ElasticPass/libElasticPass.so" \
        -load "$ELASTIC_BUILD_PATH/OptimizeBitwidth/libLLVMOptimizeBitWidth.so" \
        -load "$ELASTIC_BUILD_PATH/MyCFGPass/libMyCFGPass.so" \
        -polly-process-unprofitable -mycfgpass -S $buffers -use-lsq=false \
        "$f_ir_opt" "-cfg-outdir=$out" > /dev/null 2>&1

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

    local ret=$?
    if [[ $SMART_BUFFERS -eq 1 ]]; then
        # Run smart buffer placement pass
        ret=$(smart_buffers "$1")
    fi
    return $?
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
    local f_handshake="$out/handshake.mlir"
    local f_handshake_ready="$out/handshake_ready.mlir"
    local f_dfg_dot="$out/$name.dot"
    local f_dfg_png="$out/$name.png"
    local f_cfg_dot="$out/${name}_bbgraph.dot"
    local f_cfg_png="$out/${name}_bbgraph.png"

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
        --arith-reduce-area --canonicalize --flatten-memref-row-major \
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
    if [[ $SMART_BUFFERS -eq 0 ]]; then
        "$DYNAMATIC_OPT_BIN" "$f_handshake_ready" --allow-unregistered-dialect \
            --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
            --handshake-insert-buffers="buffer-size=2 strategy=cycles" \
            --handshake-infer-basic-blocks \
            --export-dot="legacy pretty-print=false" > /dev/null
    else
        "$DYNAMATIC_OPT_BIN" "$f_handshake_ready" --allow-unregistered-dialect \
            --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
            --handshake-infer-basic-blocks \
            --export-dot="legacy pretty-print=false" > /dev/null
    fi
    echo_status "Failed to create DFG DOT" "Created DFG DOT"
    if [ $? -ne 0 ]; then
        rm "${name}.dot" 2> /dev/null 
        return 1
    fi
    mv "$name.dot" "$f_dfg_dot"

    # Convert DOT graph to PNG
    dot -Tpng "$f_dfg_dot" > "$f_dfg_png"
    echo_status "Failed to convert DFG DOT to PNG" "Converted DFG DOT to PNG"

    if [[ $SMART_BUFFERS -eq 0 ]]; then
        return 0
    fi

    # Read function arguments from file
    local test_input="$bench_path/src/test_input.txt"
    local kernel_args=""
    while read -r line
    do
        kernel_args="$kernel_args $line"
    done < "$test_input"

    # Create CFG graph
    "$CIRCT_HANDSHAKE_RUNNER_BIN" "$f_std" --top-level-function="$name" \
        $kernel_args > /dev/null
    echo_status "Failed to create CFG DOT" "Created CFG DOT"
    if [ $? -ne 0 ]; then
        rm "${name}_bbgraph.dot" 2> /dev/null
        exit 1
    fi
    mv "${name}_bbgraph.dot" "$f_cfg_dot"

    # Convert DOT graph to PNG
    dot -Tpng "$f_cfg_dot" > "$f_cfg_png"
    exit_on_fail "Failed to convert CFG DOT to PNG" "Converted CFG DOT to PNG"

    # Run smart buffer pass
    smart_buffers "$1"
    return $?
}

compile_wrap() {    
    if [[ $NO_COMPILE -ne 0 ]]; then
        return 0
    fi

    # Run the appropriate compile flow
    cpp_to_c "$1"
    case "$FLOW" in 
        $FLOW_DYNAMATIC)
            dynamatic "$1"
        ;;
        $FLOW_LEGACY)
            legacy "$1"
        ;;
        $FLOW_BRIDGE)
            bridge "$1"
        ;;
    esac
    local ret=$?
    delete_c "$1"
    
    echo_status_arg $ret "Compilation failed!" "Compilation succeeded!"
    echo ""
    return $ret
}

simulate_wrap() {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    if [[ $SIMULATE -eq 0 ]]; then
        return 0
    fi

    if [[ ! -f "$bench_path/$FLOW/$name.dot" ]]; then
        echo "[ERROR] DOT file does not exist, skipping simulation"
        echo ""
        return 1
    fi
    
    if [[ $FLOW == $FLOW_DYNAMATIC ]]; then
        echo "[ERROR] Simulation is not yet supported on the dynamatic flow"
        echo ""
        return 1
    fi

    # Simulate the design
    simulate "$1"
    local ret=$?

    echo_status_arg $ret "Simulation failed!" "Simulation succeeded!"
    echo ""
    return $ret
}

synthesize_wrap() {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    if [[ $SYNTHESIZE -eq 0 ]]; then
        return 0
    fi

    if [[ ! -f "$bench_path/$FLOW/$name.dot" ]]; then
        echo "[ERROR] DOT file does not exist, skipping synthesization"
        return 1
    fi
    
    if [[ $FLOW == $FLOW_DYNAMATIC ]]; then
        echo "[ERROR] Synthesization is not yet supported on the dynamatic flow"
        return 1
    fi

    # Synthesize the design
    synthesize "$1"
    local ret=$?

    echo_status_arg $ret "Synthesization failed!" "Synthesization succeeded!"
    echo ""
    return $ret
}

# Run the specified flow for a benchmark.
benchmark() {
    local bench_path=$1
    local name="$(basename $bench_path)"

    echo_section "Benchmarking $name"

    # Check that a source file exists at the expected location
    local src_path="$bench_path/src/$name.cpp"
    if [[ ! -f "$src_path" ]]; then
        echo "[ERROR] No source file exists at \"$src_path\", skipping this benchmark."
    else        
        compile_wrap "$1"
        simulate_wrap "$1"
        synthesize_wrap "$1"
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
                echo "Unknown testsuite \"$arg\", printing help and exiting"
                print_help_and_exit
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
                echo "Unknown flow \"$arg\", printing help and exiting"
                print_help_and_exit
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
             "--synthesize")
                SYNTHESIZE=1
                echo "[INFO] Enabling synthesization"
                ;;
             "--no-compile")
                NO_COMPILE=1
                echo "[INFO] Skipping compilation, will use cached DOTs only"
                ;;
             "--smart-buffers")
                SMART_BUFFERS=1
                echo "[INFO] Using smart buffers"
                ;;
            "--help" | "-h")
                echo "[INFO] Printing help"
                print_help_and_exit
                ;;
            *)
                if [[ $arg == -* ]]; then
                    echo "[ERROR] Unknown option \"$arg\", printing help and exiting"
                    print_help_and_exit
                fi
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
