#!/bin/bash

# ===- runner.sh - Run benchmarks through various flows --------*- Bash -*-=== #
# 
# This script facilitates the compilation/simulation/synthesization of
# benchmarks from multiple testsuites using multiple DHLS flows. Run the script
# with the --help flag to see the list of available options.
# 
# ===----------------------------------------------------------------------=== #

# Kill the whole script on Ctrl+C
trap "exit" INT

# Get common functions
source utils.sh


# Flow-related variables 
PARSE_FLOW=0
FLOW_DYNAMATIC="dynamatic"
FLOW_LEGACY="legacy"
FLOW_BRIDGE="bridge"
FLOW="$FLOW_DYNAMATIC"

# Output path related variables
PARSE_OUTPUT=0
OUTPUT_PATH=""

# Flags
SIMULATE=0
SYNTHESIZE=0
SMART_BUFFERS=0
NO_LOOP_ROTATE=0
NO_COMPILE=0
CLEAN=0

# Explicit list of benchmarks to build
TO_BUILD=()
TESTSUITE_PATH="$DYNAMATIC_PATH/integration-test"

# Path to build folders containing legacy Dynamatic object files
ELASTIC_BUILD_PATH="$LEGACY_DYNAMATIC_ROOT/elastic-circuits/_build"
FREQUENCY_COUNTER_PATH="$ELASTIC_BUILD_PATH/FrequencyCounterPass"
FREQUENCY_DATA_PATH="$ELASTIC_BUILD_PATH/FrequencyDataGatherPass"

# Displays list of possible options and exit.
print_help_and_exit () {
    echo -e \
"$0
    [--flow <flow-name>] [--output <output-path>] 
    [--no-compile] [--simulate] [--synthesize] [--clean]
    [--smart-buffers] [--loop-rotate]
    [--help|-h] 
    [<bench-name> ]...

List of options:
  --flow <flow-name>            : run a specific flow (dynamatic [default], legacy, bridge)
  --output <output-path>        : output path where to store results (relative path w.r.t. each benchmark directory)
  --no-compile                  : do not re-compile benchmarks, only use cached DOTs
  --simulate                    : enable VHDL simulation/verification with Modelsim
  --synthesize                  : enable VHDL synthesization with Vivado
  --clean                       : delete all output directories in selected benchmarks
  --smart-buffers               : enable smart buffer placement (instead of stupid buffer placement)
  --no-loop-rotate              : disable loop rotation pass when compiling
  <bench-name>...               : run the selected flow only on specific benchmarks from the selected testsuite   
  --help | -h                   : display this help message
"
    exit $2
}

# Deletes and recreate a directory.
#   $1: path to directory
reset_output_dir() {
    rm -rf "$1"
    mkdir -p "$1"    
}

# Simulates a benchmark using legacy Dynamatic's Modelsim flow.
#   $1: absolute path to benchmark directory (without trailing slash)
simulate() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local d_comp="$bench_path/$OUTPUT_PATH/comp"

    # Generated directories/files
    local d_sim="$bench_path/$OUTPUT_PATH/sim"
    local d_c_src="$d_sim/C_SRC"
    local d_c_out="$d_sim/C_OUT"
    local d_vhdl_src="$d_sim/VHDL_SRC"
    local d_vhdl_out="$d_sim/VHDL_OUT"
    local d_input_vectors="$d_sim/INPUT_VECTORS"
    local d_hls_verify="$d_sim/HLS_VERIFY"
    local f_report="$d_sim/report.txt"
    
    reset_output_dir "$d_sim"
    
    # Create simulation directories
    mkdir -p "$d_c_src" "$d_c_out" "$d_vhdl_src" "$d_vhdl_out" \
        "$d_input_vectors" "$d_hls_verify"
    
    # Copy integration headers to sim directory to make it visible by the HLS verifier
    cp "$bench_path/../integration_utils.h" "$d_sim"

    # Copy VHDL module and VHDL components to dedicated folder
    cp "$d_comp/$name.vhd" "$d_vhdl_src"
    cp "$d_comp/"LSQ*.v "$d_vhdl_src" 2> /dev/null
    cp "$LEGACY_DYNAMATIC_ROOT"/components/*.vhd "$d_vhdl_src"

    # Copy sources to dedicated folder
    cp "$bench_path/$name.c" "$d_c_src" 
    cp "$bench_path/$name.h" "$d_c_src"

    # Simulate and verify design
    echo_info "Launching Modelsim simulation"
    cd "$d_hls_verify"
    "$HLS_VERIFIER_BIN" cover -aw32 "$d_c_src/$name.c" "$d_c_src/$name.c" \
        $name > "$f_report"
    local ret=$?
    cd - > /dev/null
    return $ret
}

# Synthesizes a benchmark and generate utilization/timing reports using Vivado.
#   $1: absolute path to benchmark directory (without trailing slash)
synthesize() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local d_comp="$bench_path/$OUTPUT_PATH/comp"

    # Generated directories/files
    local d_synth="$bench_path/$OUTPUT_PATH/synth"
    local d_hdl="$d_synth/hdl"
    local f_script="$d_synth/synthesize.tcl"
    local f_period="$d_synth/period_4.xdc"
    local f_utilization_syn="$d_synth/utilization_post_syn.rpt"
    local f_timing_syn="$d_synth/timing_post_syn.rpt"
    local f_utilization_pr="$d_synth/utilization_post_pr.rpt"
    local f_timing_pr="$d_synth/timing_post_pr.rpt"

    reset_output_dir "$d_synth"

    # Copy all synthesizable components to specific folder for Vivado
    mkdir -p "$d_hdl"
    cp "$d_comp/$name.vhd" "$d_hdl"
    cp "$LEGACY_DYNAMATIC_ROOT"/components/*.vhd "$d_hdl"

    # See if we should include any LSQ in the synthesis script
    local read_verilog=""
    if ls "$d_comp"/LSQ*.v 1> /dev/null 2>&1; then
        cp "$d_comp/"LSQ*.v "$d_hdl"
        read_verilog="read_verilog [glob $d_synth/hdl/*.v]"
    fi

    # Generate synthesization scripts
    echo -e \
"set_param general.maxThreads 8
read_vhdl -vhdl2008 [glob $d_synth/hdl/*.vhd]
$read_verilog
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
    echo_info "Created synthesization scripts"

    echo_info "Launching Vivado synthesization"
    vivado -mode tcl -source "$f_script" > /dev/null
    local ret=$?
    rm -rf *.jou *.log .Xil
    return $ret
}

# Converts a DOT file with the name <benchmark-name>.dot inside the compile
# directory to a VHDL design using the legacy dot2vhdl tool. The resulting VHDL
# is placed under the same directory with the name <benchmark-name>.dot.
#   $1: absolute path to benchmark directory (without trailing slash)
legacy_dot2vhdl() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local d_comp="$bench_path/$OUTPUT_PATH/comp"

    # Convert DOT graph to VHDL
    cd "$d_comp"
    "$DOT2VHDL_BIN" "$d_comp/$name" >/dev/null
    exit_on_fail "Failed to convert DOT to VHDL" "Converted DOT to VHDL"
    local ret=$?
    cd - > /dev/null
    return $ret
}

# Runs Dynamatic++'s profiler tool. Input and output files are read from/written
# to "$(basename $1)/$OUTPUT_PATH/comp/
# located at "$(basename $1)/$OUTPUT_PATH/comp/std.mlir".
#   $1: absolute path to benchmark directory (without trailing slash)
#   $2: input filename (std-level IR), without extension (assumes .mlir)
#   $3: 1 if running the profiler in legacy-mode, 0 otherwise
run_dynamatic_profiler() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local d_comp="$bench_path/$OUTPUT_PATH/comp"

    local f_src="$d_comp/$2.mlir"
    local legacy_mode=$3

    # Generated files
    local f_out=""
    local f_cfg_png="$d_comp/${name}_bbgraph.png"
    if [[ $legacy_mode -eq 0 ]]; then
        f_out="$d_comp/frequencies.csv"
    else
        f_out="$d_comp/${name}_bbgraph.dot"
    fi

    local test_input="$bench_path/src/test_input.txt"
    if [[ ! -f "$test_input" ]]; then
        echo_error "Failed to find test input for frequency counting"
    fi

    # Run profiler
    local print_dot=""
    if [[ $legacy_mode -ne 0 ]]; then
        print_dot="--print-dot"
    fi
    "$DYNAMATIC_PROFILER_BIN" "$f_src" --top-level-function="$name" \
        --input-args-file="$test_input" $print_dot > $f_out 
    echo_status "Failed to run profiler" "Profiled std-level kernel"
    if [[ $? -ne 0 ]]; then
        return $?
    fi

    if [[ $legacy_mode -ne 0 ]]; then
        # Create the PNG corresponding to the DOT in legacy mode
        dot -Tpng "$f_out" > "$f_cfg_png"
        exit_on_fail "Failed to convert CFG DOT to PNG" "Converted CFG DOT to PNG"
    fi
    return 0
}

# Runs Dynamatic++'s smart buffer placement pass on an input Handshake-level IR.
#   $1: absolute path to benchmark directory (without trailing slash)
#   $2: path to input Handshake-level IR
smart_buffers() {
    local bench_path="$1"
    local f_handshake_input="$2"
    local name="$(basename $bench_path)"
    local d_comp="$bench_path/$OUTPUT_PATH/comp"
   
    # Generated directories/files
    local f_handshake_buffered="$d_comp/handshake_buffered.mlir"

    # Run profiler    
    run_dynamatic_profiler "$1" "std_dyn_transformed" 0
    echo_status "Failed to run frequency profiler" "Ran frequency profiler"
    if [[ $? -ne 0 ]]; then
        return 1 
    fi

    # Run Buffer placement
    echo_info "Placing smart buffers"
    "$DYNAMATIC_OPT_BIN" "$f_handshake_input" \
        --allow-unregistered-dialect \
        --handshake-set-buffering-properties="version=fpga20" \
        --handshake-place-buffers="algorithm=fpga20-legacy timing-models=$DYNAMATIC_PATH/data/components.json frequencies=$d_comp/frequencies.csv timeout=300 dump-logs" \
        > "$f_handshake_buffered"
    echo_status "Failed to buffer IR" "Buffered handshake"
    local ret=$?
    mv buffer-placement "$d_comp" > /dev/null 2>&1 
    if [[ $ret -ne 0 ]]; then
        return 1 
    fi
    return 0
}

# Runs legacy Dynamatic's smart buffer placement pass on an input DOT.
#   $1: absolute path to benchmark directory (without trailing slash)
legacy_smart_buffers() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local d_comp="$bench_path/$OUTPUT_PATH/comp"

    # Generated directories/files
    local d_buffers="$d_comp/buffers"
    local f_dfg_buf_dot="$d_comp/${name}_graph_buf.dot"
    local f_dfg_buf_png="$d_comp/${name}_graph_buf.png"
    local f_cfg_buf_dot="$d_comp/${name}_bbgraph_buf.dot"
    local f_cfg_buf_png="$d_comp/${name}_bbgraph_buf.png"
    local f_report="$d_buffers/report.txt"

    reset_output_dir "$d_buffers"

    # Run smart buffer placement
    echo_info "Placing smart buffers"
    cd $d_buffers
    "$BUFFERS_BIN" buffers -filename="../$name" -period=4 \
        -model_mode=mixed -solver=gurobi_cl > "$f_report" 2>&1
    echo_status "Failed to place smart buffers" "Placed smart buffers"
    if [[ $? -ne 0 ]]; then
        return 1 
    fi
    cd - > /dev/null

    # Convert bufferized DFG DOT graph to PNG
    dot -Tpng "$f_dfg_buf_dot" > "$f_dfg_buf_png"
    exit_on_fail "Failed to convert bufferized DFG DOT to PNG" "Converted bufferized DFG DOT to PNG"

    # Convert bufferized CFG DOT graph to PNG
    dot -Tpng "$f_cfg_buf_dot" > "$f_cfg_buf_png"
    exit_on_fail "Failed to convert bufferized CFG DOT to PNG" "Converted bufferized CFG DOT to PNG"

    # Rename unbufferized DOTS
    mv "$d_comp/$name.dot" "$d_comp/${name}_nobuf.dot" 
    mv "$d_comp/$name.png" "$d_comp/${name}_nobuf.png" 
    mv "$d_comp/${name}_bbgraph.dot" "$d_comp/${name}_bbgraph_nobuf.dot" 
    mv "$d_comp/${name}_bbgraph.png" "$d_comp/${name}_bbgraph_nobuf.png" 

    # Rename bufferized DOTs
    mv "$f_dfg_buf_dot" "$d_comp/$name.dot" 
    mv "$f_dfg_buf_png" "$d_comp/$name.png" 
    mv "$f_cfg_buf_dot" "$d_comp/${name}_bb_graph.dot"

    return $ret
}

# Exports Handshake-level IR to DOT using Dynamatic++, then converts the DOT to
# a PNG using xdot. Input and output files are read from/written to
# "$(basename $1)/$OUTPUT_PATH/comp/
#   $1: absolute path to benchmark directory (without trailing slash)
#   $2: input filename (Handshake-level IR), without extension (assumes .mlir)
#   $3: mode to run the tool in; options are "visual", "legacy", "legacy-buffers"
#   $4: edge style to use in the DOT; options are "spline", "ortho"
#   $5: output filename (if not provided, same as the input filename), without
#       extension (will use .dot and .png)
export_dot() {
    local bench_path="$1"
    local name="$(basename $bench_path)"
    local d_comp="$bench_path/$OUTPUT_PATH/comp"
    
    local f_src="$d_comp/$2.mlir"
    local mode="$3"
    local edge_style="$4"

    # Generated directories/files
    local f_dot="$d_comp/$name.dot"
    local f_png="$d_comp/$name.png"
    if [[ ! -z $5 ]]; then
        f_dot="$d_comp/$5.dot"
        f_png="$d_comp/$5.png"
    fi

    # Export to DOT
    "$DYNAMATIC_EXPORT_DOT_BIN" "$f_src" "--mode=$mode" \
        "--edge-style=$edge_style" \
        "--timing-models=$DYNAMATIC_PATH/data/components.json" \
        > "$f_dot"
    exit_on_fail "Failed to create DOT" "Created DOT"

    # Convert DOT graph to PNG
    dot -Tpng "$f_dot" > "$f_png"
    exit_on_fail "Failed to convert DOT to PNG" "Converted DOT to PNG"
    return 0
}

# Lowers a C/C++ benchmark all the way down to Handshake using Dynamatic++'s
# flow. The lowest-level output of this file (Handshake-level IR) is located at
# "$(basename $1)/$OUTPUT_PATH/comp/handshake.mlir".
#   $1: absolute path to benchmark directory (without trailing slash)
mlir_to_handshake() {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    # Generated directories/files
    local d_comp="$bench_path/$OUTPUT_PATH/comp"
    local f_affine="$d_comp/affine.mlir"
    local f_affine_mem="$d_comp/affine_mem.mlir"
    local f_scf="$d_comp/scf.mlir"
    local f_scf_transformed="$d_comp/scf_transformed.mlir"
    local f_std="$d_comp/std.mlir"
    local f_std_transformed="$d_comp/std_transformed.mlir"
    local f_std_dyn_transformed="$d_comp/std_dyn_transformed.mlir"
    local f_handshake="$d_comp/handshake.mlir"

    reset_output_dir "$d_comp"

    # source code -> affine dialect
    local include="$POLYGEIST_PATH/llvm-project/clang/lib/Headers/"
    "$POLYGEIST_CLANG_BIN" "$bench_path/$name.c" -I "$include" \
        --function="$name" -S -O3 --memref-fullrank --raise-scf-to-affine \
        > "$f_affine" 2>/dev/null
    exit_on_fail "Failed source -> affine conversion" "Lowered to affine"
    
    # memory analysis 
    "$DYNAMATIC_OPT_BIN" "$f_affine" --allow-unregistered-dialect \
        --force-memory-interface="force-lsq" \
        > "$f_affine_mem"
    exit_on_fail "Failed memory analysis" "Passed memory analysis"

    # affine dialect -> scf dialect
    "$DYNAMATIC_OPT_BIN" "$f_affine_mem" --allow-unregistered-dialect \
        --lower-affine-to-scf > "$f_scf"
    exit_on_fail "Failed affine -> scf conversion" "Lowered to scf"

    # scf transformations
	local loop_rotate="--scf-rotate-for-loops"
    if [[ $NO_LOOP_ROTATE -ne 0 ]]; then
        loop_rotate=""
    fi 
    "$DYNAMATIC_OPT_BIN" "$f_scf" --allow-unregistered-dialect \
        --scf-simple-if-to-select $loop_rotate \
        > "$f_scf_transformed"
    exit_on_fail "Failed to transform scf IR" "Transformed scf"

    # scf dialect -> std dialect
    "$DYNAMATIC_OPT_BIN" "$f_scf_transformed" --allow-unregistered-dialect \
        --lower-scf-to-cf > "$f_std"
    exit_on_fail "Failed scf -> std conversion" "Lowered to std"

    # std transformations (1)
    "$MLIR_OPT_BIN" "$f_std" --allow-unregistered-dialect --canonicalize --cse \
        --sccp --symbol-dce --control-flow-sink --loop-invariant-code-motion \
        --canonicalize \
        > "$f_std_transformed"
    exit_on_fail "Failed to transform std IR (1)" "Transformed std (1)"

    # std transformations (2)   
    "$DYNAMATIC_OPT_BIN" "$f_std_transformed" --allow-unregistered-dialect \
        --flatten-memref-row-major --flatten-memref-calls \
        --arith-reduce-strength --push-constants \
        > "$f_std_dyn_transformed"
    exit_on_fail "Failed to transform std IR (2)" "Transformed std (2)"

    # std dialect -> handshake dialect
    "$DYNAMATIC_OPT_BIN" "$f_std_dyn_transformed" --allow-unregistered-dialect \
        --lower-std-to-handshake-fpga18 \
        --handshake-fix-arg-names="source=$bench_path/src/$name.c" \
        > "$f_handshake"
    exit_on_fail "Failed std -> handshake conversion" "Lowered to handshake"
    return 0
}

# Compiles a benchmark using Dynamatic's flow. At the moment, this only lowers
# the source code down to handshake.
#   $1: absolute path to benchmark directory (without trailing slash)
dynamatic () {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    # Generated directories/files
    local d_comp="$bench_path/$OUTPUT_PATH/comp"
    local f_handshake="$d_comp/handshake.mlir"
    local f_handshake_transformed="$d_comp/handshake_transformed.mlir"
    local f_handshake_buffered="$d_comp/handshake_buffered.mlir"
    local f_handshake_export="$d_comp/handshake_export.mlir"
    local f_netlist="$d_comp/netlist.mlir"
    local f_netlist_explicit="$d_comp/netlist_explicit.mlir"

    mlir_to_handshake "$1"
    
    # handshake transformations
    "$DYNAMATIC_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-concretize-index-type="width=32" \
        --handshake-minimize-cst-width --handshake-optimize-bitwidths="legacy" \
        --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
        > "$f_handshake_transformed"    
    exit_on_fail "Failed to transform handshake IR" "Transformed handshake"

    export_dot "$1" "handshake_transformed" "visual" "ortho" "pre_buf"

    # Buffer placement
    if [[ $SMART_BUFFERS -eq 0 ]]; then
        "$DYNAMATIC_OPT_BIN" "$f_handshake_transformed" \
            --allow-unregistered-dialect \
            --handshake-place-buffers="algorithm=on-merges" \
            > "$f_handshake_buffered"
        exit_on_fail "Failed to buffer IR" "Buffered handshake"
    else
        # Run buffer placement
        smart_buffers "$1" "$f_handshake_transformed"
        if [[ $? -ne 0 ]]; then
            return 1
        fi
    fi

    # Canonicalization
    "$DYNAMATIC_OPT_BIN" "$f_handshake_buffered" \
        --allow-unregistered-dialect --handshake-canonicalize \
        > "$f_handshake_export"
    exit_on_fail "Failed to canonicalize handshake" "Canonicalized handshake"

    # handshake dialect -> netlist    
    # "$DYNAMATIC_OPT_BIN" "$f_handshake_export" --allow-unregistered-dialect \
    #     --lower-handshake-to-netlist -debug-only=dialect-conversion > "$f_netlist"
    # exit_on_fail "Failed handshake -> netlist conversion" "Lowered to netlist"

    # Export to DOT (one clean for viewing and one compatible with legacy)
    export_dot "$1" "handshake_export" "visual" "ortho" "visual"
    export_dot "$1" "handshake_export" "legacy" "ortho" "$name"

    # Convert to VHDL
    legacy_dot2vhdl "$1"
    return $?
}

# Compiles a benchmark using Dynamatic's flow. At the end, generates a
# legacy-compatible DOT that can be passed to legacy Dynamatic's passes.
#   $1: absolute path to benchmark directory (without trailing slash)
bridge () {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    # Generated directories/files
    local d_comp="$bench_path/$OUTPUT_PATH/comp"
    local f_handshake="$d_comp/handshake.mlir"
    local f_handshake_transformed="$d_comp/handshake_transformed.mlir"
    local f_handshake_buffered="$d_comp/handshake_buffered.mlir"

    mlir_to_handshake "$1"

    # handshake transformations
    "$DYNAMATIC_OPT_BIN" "$f_handshake" --allow-unregistered-dialect \
        --handshake-prepare-for-legacy \
        --handshake-concretize-index-type="width=32" \
        --handshake-minimize-cst-width --handshake-optimize-bitwidths="legacy" \
        --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
        > "$f_handshake_transformed"
    exit_on_fail "Failed to transform Handshake IR" "Transformed handshake"

    # Buffer placement
    if [[ $SMART_BUFFERS -eq 0 ]]; then
        "$DYNAMATIC_OPT_BIN" "$f_handshake_transformed" \
            --allow-unregistered-dialect \
            --handshake-place-buffers="algorithm=on-merges" \
            > "$f_handshake_buffered"
        exit_on_fail "Failed to buffer IR" "Buffered handshake"
        export_dot "$1" "handshake_buffered" "legacy" "ortho" "$name"
    else
        export_dot "$1" "handshake_transformed" "legacy-buffers" "ortho" "$name"

        # Run profiler and smart buffer placement
        run_dynamatic_profiler "$1" "std_dyn_transformed" 1
        if [[ $? -ne 0 ]]; then
            return $?
        fi
        legacy_smart_buffers "$1"
        if [[ $? -ne 0 ]]; then
            return $?
        fi
    fi
    
    # Convert to VHDL
    legacy_dot2vhdl "$1"
    return $?
}

# Compiles a benchmark using legacy Dynamatic's flow.
#   $1: absolute path to benchmark directory (without trailing slash)
legacy () {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    # Generated directories/files
    local d_comp="$bench_path/$OUTPUT_PATH/comp"
    local f_ir="$d_comp/ir.ll"
    local f_ir_opt="$d_comp/ir_opt.ll"
    local f_ir_opt_obj="$d_comp/ir_opt.o"
    local f_dot="$d_comp/$name.dot"
    local f_dot_bb="$d_comp/${name}_bbgraph.dot"
    local f_png="$d_comp/$name.png"
    local f_png_bb="$d_comp/${name}_bbgraph.png"

    reset_output_dir "$d_comp"

    # source code -> LLVM IR
	"$LLVM_CLANG_BIN" -Xclang -disable-O0-optnone -emit-llvm -S \
        -I "$bench_path" \
        -c "$bench_path/$name.c" \
        -o $d_comp/ir.ll
    exit_on_fail "Failed to compile to LLVM IR" "Compiled to LLVM IR"
    
    # LLVM IR -> standard optimized LLVM IR
	local loop_rotate="-loop-rotate"
    if [[ $NO_LOOP_ROTATE -ne 0 ]]; then
        loop_rotate=""
    fi 

    "$LLVM_OPT_BIN" -mem2reg $loop_rotate -constprop -simplifycfg -die \
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
        -polly-process-unprofitable -mycfgpass -S $buffers -use-lsq=true \
        "$f_ir_opt" "-cfg-outdir=$d_comp" > /dev/null 2>&1

    # Remove temporary build files
    rm *_freq.txt mapping.txt out.txt

    # Can't check the return value here, as this often crashes right after
    # generating the files we want
    if [[ ! -f "$d_comp/${name}_graph.dot" || ! -f "$d_comp/${name}_bbgraph.dot" ]]; then
        echo_error "Failed to generate DOTs"
        return 1
    fi
    echo_info "Applied elastic pass"

    # Rename DOT file
    mv "$d_comp/${name}_graph.dot" "$f_dot"

    # Convert DOTs to PNG
    dot -Tpng "$f_dot" > "$f_png"
    echo_status "Failed to convert DOT to PNG" "Converted DOT to PNG"
    dot -Tpng "$f_dot_bb" > "$f_png_bb"
    echo_status "Failed to convert DOT (BB) to PNG" "Converted DOT (BB) to PNG"

    if [[ $SMART_BUFFERS -eq 1 ]]; then
        # Run smart buffer placement pass
        legacy_smart_buffers "$1"
        if [[ $? -ne 0 ]]; then
            return 1
        fi
    fi

    legacy_dot2vhdl "$1"
    return $?
}

# Wraps the compilation step, calling the flow specified by script arguments.
#   $1: absolute path to benchmark directory (without trailing slash)
compile_wrap() {    
    if [[ $NO_COMPILE -ne 0 ]]; then
        return 0
    fi

    # Run the appropriate compile flow
    echo_subsection "Compile"
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
    echo_status "Compilation failed!" "Compilation succeeded!"
    echo ""
    return $?
}

# Wraps the simulation step, checking that an input DOT is present at the right
# location before starting the simulation.
#   $1: absolute path to benchmark directory (without trailing slash)
simulate_wrap() {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    if [[ $SIMULATE -eq 0 ]]; then
        return 0
    fi

    if [[ ! -f "$bench_path/$OUTPUT_PATH/comp/$name.dot" ]]; then
        echo_error "DOT file does not exist, skipping simulation"
        echo ""
        return 1
    fi
    
    # Simulate the design
    echo_subsection "Simulate"
    simulate "$1"
    local ret=$?

    echo_status_arg $ret "Simulation failed!" "Simulation succeeded!"
    echo ""
    return $ret
}

# Wraps the synthesization step, checking that an input DOT is present at the
# right location before starting the simulation.
#   $1: absolute path to benchmark directory (without trailing slash)
synthesize_wrap() {
    local bench_path="$1"
    local name="$(basename $bench_path)"

    if [[ $SYNTHESIZE -eq 0 ]]; then
        return 0
    fi

    if [[ ! -f "$bench_path/$OUTPUT_PATH/comp/$name.dot" ]]; then
        echo_error "DOT file does not exist, skipping synthesization"
        return 1
    fi
    
    # Synthesize the design
    echo_subsection "Synthesize"
    synthesize "$1"
    local ret=$?

    echo_status_arg $ret "Synthesization failed!" "Synthesization succeeded!"
    echo ""
    return $ret
}

# Runs a benchmark through all the steps specified by script arguments.
#   $1: absolute path to benchmark directory (without trailing slash)
benchmark() {
    local bench_path=$1
    local name="$(basename $bench_path)"

    echo_section "Benchmarking $name"

    # Delete output directories if necessary
    if [[ $CLEAN -ne 0 ]]; then
        for dir in $bench_path/*/; do
            rm -rf $dir
        done
    fi

    # Check that a source file exists at the expected location
    local src_path="$bench_path/$name.c"
    if [[ ! -f "$src_path" ]]; then
        echo_error "No source file exists at \"$src_path\", skipping this benchmark."
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
    if [[ $PARSE_FLOW -eq 1 ]]; then
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
        echo_info "Setting flow to \"$arg\""
        PARSE_FLOW=0
    
    elif [[ $PARSE_OUTPUT -eq 1 ]]; then
        # Store the output path
        OUTPUT_PATH="$arg"
        echo_info "Setting output path to \"$arg\""
        PARSE_OUTPUT=0

    else
        # Parse the next argument
        case "$arg" in 
            "--flow")
                PARSE_FLOW=1
                ;;
            "--output")
                PARSE_OUTPUT=1
                ;;
            "--simulate")
                SIMULATE=1
                echo_info "Enabling simulation"
                ;;
             "--synthesize")
                SYNTHESIZE=1
                echo_info "Enabling synthesization"
                ;;
             "--no-compile")
                NO_COMPILE=1
                echo_info "Skipping compilation, will use cached DOTs only"
                ;;
             "--smart-buffers")
                SMART_BUFFERS=1
                echo_info "Using smart buffers"
                ;;
            "--no-loop-rotate")
                NO_LOOP_ROTATE=1
                echo_info "Disabling loop rotation pass"
                ;;
            "--clean")
                CLEAN=1
                echo_info "Deleting output directories for selected benchmarks"
                ;;
            "--help" | "-h")
                echo_info "Printing help"
                print_help_and_exit
                ;;
            *)
                if [[ $arg == -* ]]; then
                    echo_error "Unknown option \"$arg\", printing help and exiting"
                    print_help_and_exit
                fi
                TO_BUILD+=("$arg")
                echo_info "Registering benchmark \"$arg\" in the build list"
                ;;
        esac
    fi
done

# Set default output path if it wasn't provided as an argument
if [[ ${#OUTPUT_PATH} -eq 0 ]]; then
    OUTPUT_PATH="$FLOW"
fi

# Check that required environment variables exist
check_env_variables \
    BUFFERS_BIN \
    DOT2VHDL_BIN \
    DYNAMATIC_EXPORT_DOT_BIN \
    DYNAMATIC_OPT_BIN \
    DYNAMATIC_PROFILER_BIN \
    HLS_VERIFIER_BIN \
    LEGACY_DYNAMATIC_ROOT \
    LLVM_CLANG_BIN \
    LLVM_OPT_BIN \
    MLIR_OPT_BIN \
    POLYGEIST_CLANG_BIN \
    POLYGEIST_PATH

# Build benchmarks
if [ ${#TO_BUILD[@]} -eq 0 ]; then
    echo_info "No benchmark provided, building all benchmarks in testsuite..."
    echo ""
    for path in $TESTSUITE_PATH/*/; do
        benchmark "${path%/}"
    done
else
    echo ""
    for path in ${TO_BUILD[@]}; do
        benchmark "$TESTSUITE_PATH/$path"
    done
fi

echo_info "All done!"
