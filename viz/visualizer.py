# Libraries
import argparse
import glob
import os
import numpy as np
from dataclasses import dataclass
from enum import Enum
from bokeh.layouts import layout, row, column
from bokeh.models import Row
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.plotting import figure, show

# Local
import dot_parser
from utils import BokehParams, nested_barchart, save_to_disk

# Typing
from typing import (
    Callable,
    ClassVar,
    Final,
    Generic,
    Mapping,
    Protocol,
    Self,
    Sequence,
    Type,
    TypeVar,
)

# ===------------------------------------------------------------------------------=== #
# Type and protocol definitions
# ===------------------------------------------------------------------------------=== #

DataType = TypeVar("DataType", bound="SupportsFromFile")
DataStorage = TypeVar("DataStorage", covariant=True)
T = TypeVar("T", contravariant=True)


class SupportsFromFile(Protocol):
    @classmethod
    def from_file(cls: Type[DataType], filepath: str) -> DataType:
        ...


class ChartGenerator(Generic[DataStorage], Protocol):
    def __call__(
        self,
        dtype: Type[T],
        extract: Callable[[DataStorage], T],
        positive: bool = ...,
        fig_args: BokehParams = ...,
        vbar_args: BokehParams = ...,
    ) -> figure:
        ...


# ===------------------------------------------------------------------------------=== #
# Class definitions
# ===------------------------------------------------------------------------------=== #


@dataclass(frozen=True)
class DataInfo(Generic[DataStorage]):
    benchmarks: Final[Sequence[str]]
    data: Final[Mapping[str, Sequence[DataStorage]]]

    def __post_init__(self):
        if len(self.benchmarks) == 0:
            raise ValueError(
                "Reports instance need to be created with at least one benchmark"
            )
        if len(self.data) == 0:
            raise ValueError(
                "Reports instance need to be created with at least one benchmark"
            )

        for flow, data in self.data.items():
            if len(data) != len(self.benchmarks):
                raise ValueError(
                    f"Incorrect number of data points found for flow {flow}. Expected "
                    f"{len(self.benchmarks)} but got {len(data)}."
                )


@dataclass(frozen=True)
class DotData:
    counts: Final[Mapping["DotData.ComponentType", int]]

    @classmethod
    def from_file(cls: Type[Self], filepath: str) -> Self:
        counts: dict[DotData.ComponentType, int] = {}

        with open(filepath, "r") as report:
            while line := report.readline():
                line = line.strip()
                if not dot_parser.is_node(line):
                    continue
                attributes = dot_parser.get_attributes(line)
                if (typeAttr := attributes.get("type", None)) is not None:
                    comp: DotData.ComponentType | None = None
                    if typeAttr == "Operator":
                        if (opAttr := attributes.get("op", None)) is not None:
                            comp = _DECODE_COMPONENT_TYPE.get(opAttr, None)
                    else:
                        comp = _DECODE_COMPONENT_TYPE.get(typeAttr, None)
                    if comp is not None:
                        counts[comp] = counts.get(comp, 0) + 1
                    else:
                        raise ValueError(
                            f"Found unknown component in DOT\n\tIn: {line}"
                        )

        return DotData(counts)

    class ComponentType(Enum):
        # Dataflow components
        ENTRY = "Entry"
        CONTROL_MERGE = "CntrlMerge"
        MERGE = "Merge"
        MUX = "Mux"
        BRANCH = "Branch"
        BUFFER = "Buffer"
        MEMORY_CONTROLLER = "MC"
        Fork = "Fork"
        SOURCE = "Source"
        SINK = "Sink"
        CONSTANT = "Constant"
        EXIT = "Exit"
        LOAD = "mc_load_op"
        STORE = "mc_store_op"
        RETURN = "ret_op"
        # Arithmetic components
        SELECT = "select_op"
        INDEX_CAST = "zext_op"
        ADDI = "add_op"
        ADDF = "fadd_op"
        SUBI = "sub_op"
        SUBF = "fsub_op"
        ANDI = "and_op"
        ORI = "or_op"
        XORI = "xor_op"
        MULI = "mul_op"
        MULF = "fmul_op"
        DIVUI = "udiv_op"
        DIVSI = "sdiv_op"
        DIVF = "fdiv_op"
        SITOFP = "sitofp_op"
        REMSI = "urem_op"
        EXTSO = "sext_op"
        EXTUI = "zext_op"
        TRUCI = "trunc_op"
        SHRSI = "ashr_op"
        SHLI = "shl_op"
        GET_ELEMENT_PTR = "getelementptr_op"
        # Integer comparisons
        EQ = "icmp_eq_op"
        NE = "icmp_ne_op"
        SLT = "icmp_slt_op"
        SLE = "icmp_sle_op"
        SGT = "icmp_sgt_op"
        SGE = "icmp_sge_op"
        ULT = "icmp_ult_op"
        ULE = "icmp_ule_op"
        UGT = "icmp_ugt_op"
        UGE = "icmp_uge_op"
        # FLoating comparisons
        ALWAYS_FALSE = "fcmp_false_op"
        F_OEQ = "fcmp_oeq_op"
        F_OGT = "fcmp_ogt_op"
        F_OGE = "fcmp_oge_op"
        F_OLT = "fcmp_olt_op"
        F_OLE = "fcmp_ole_op"
        F_ONE = "fcmp_one_op"
        F_ORD = "fcmp_orq_op"
        F_UEQ = "fcmp_ueq_op"
        F_UGT = "fcmp_ugt_op"
        F_UGE = "fcmp_uge_op"
        F_ULT = "fcmp_ult_op"
        F_ULE = "fcmp_ule_op"
        F_UNE = "fcmp_une_op"
        F_UNO = "fcmp_uno_op"
        ALWAYS_TRUE = "fcmp_true_op"

        def is_actually_dataflow(self: Self) -> bool:
            return self in (
                DotData.ComponentType.LOAD,
                DotData.ComponentType.STORE,
                DotData.ComponentType.RETURN,
            )

        @classmethod
        def get_dataflow(cls: Type[Self]) -> list[Self]:
            return list(
                filter(
                    lambda comp: comp.is_actually_dataflow() or not "_op" in comp.value,
                    DotData.ComponentType,
                )
            )

        @classmethod
        def get_arithmetic(cls: Type[Self]) -> list[Self]:
            return list(
                filter(
                    lambda comp: not comp.is_actually_dataflow()
                    and "_op" in comp.value
                    and not "icmp_" in comp.value
                    and not "fcmp_" in comp.value,
                    DotData.ComponentType,
                )
            )

        @classmethod
        def get_icmp(cls: Type[Self]) -> list[Self]:
            return list(
                filter(
                    lambda comp: "icmp_" in comp.value,
                    DotData.ComponentType,
                )
            )

        @classmethod
        def get_fcmp(cls: Type[Self]) -> list[Self]:
            return list(
                filter(
                    lambda comp: "fcmp_" in comp.value,
                    DotData.ComponentType,
                )
            )

        @classmethod
        def build_decoder(cls: Type[Self]) -> dict[str, Self]:
            return {comp.value: comp for comp in DotData.ComponentType}


_DECODE_COMPONENT_TYPE: Final[
    Mapping[str, DotData.ComponentType]
] = DotData.ComponentType.build_decoder()


@dataclass(frozen=True)
class SimData:
    time: Final[int]

    @classmethod
    def from_file(cls: Type[Self], filepath: str) -> Self:
        time: int = -1

        def was_data_parsed(name: str, val: int):
            if val < 0:
                raise ValueError(
                    f"Failed to find {name} in simulation report @ {filepath}"
                )

        with open(filepath, "r") as report:
            while line := report.readline():
                line = line.strip()
                if "# ** Note: simulation done!" == line:
                    if nextline := report.readline():
                        time = int(split_on_whitespace(nextline.strip())[2])
                    else:
                        raise ValueError(f'Unexpected EOF @ "{filepath}"')

        was_data_parsed("time", time)
        return SimData(time)


@dataclass(frozen=True)
class AreaData:
    lut_logic: Final[int]
    lut_ram: Final[int]
    lut_reg: Final[int]
    reg_ff: Final[int]
    reg_latch: Final[int]
    dsp: Final[int]

    @property
    def luts(self) -> int:
        return self.lut_logic + self.lut_ram + self.lut_reg

    @property
    def regs(self) -> int:
        return self.reg_ff + self.reg_latch

    @classmethod
    def from_file(cls: Type[Self], filepath: str) -> Self:
        section: AreaData._ReportSection | None = None

        lut_logic: int = 0
        lut_ram: int = 0
        lut_reg: int = 0
        reg_ff: int = 0
        reg_latch: int = 0
        dsp: int = 0

        def get_used_column(line: str) -> int:
            return int(line.split("|")[2].strip())

        # Parse the utilization report section by section
        with open(filepath, "r") as report:
            while line := report.readline():
                line = line.strip()
                if (new_section := AreaData._DECODE.get(line, None)) is not None:
                    section = new_section
                elif section == AreaData._ReportSection.SLICE_LOGIC:
                    if "LUT as Logic" in line:
                        lut_logic = get_used_column(line)
                    elif "LUT as Distributed RAM" in line:
                        lut_ram = get_used_column(line)
                    elif "LUT as Shift Register" in line:
                        lut_reg = get_used_column(line)
                    elif "Register as Flip Flop" in line:
                        reg_ff = get_used_column(line)
                    elif "Register as Latch" in line:
                        reg_latch = get_used_column(line)
                elif section == AreaData._ReportSection.DSP:
                    if "DSPs" in line:
                        dsp = get_used_column(line)

        return AreaData(lut_logic, lut_ram, lut_reg, reg_ff, reg_latch, dsp)

    class _ReportSection(Enum):
        SLICE_LOGIC = 0
        REGISTER_SUMMARY = 1
        SLICE_LOGIC_DISTRIB = 2
        MEMORY = 3
        DSP = 4
        IO_GT = 5
        CLOCKING = 6
        SPECIFIC_FEATURE = 7
        PRIMITIVES = 8
        BLACK_BOXES = 9
        NETLISTS = 10

    _DECODE: ClassVar[Mapping[str, "AreaData._ReportSection"]] = {
        "1. Slice Logic": _ReportSection.SLICE_LOGIC,
        "1.1 Summary of Registers by Type": _ReportSection.REGISTER_SUMMARY,
        "2. Slice Logic Distribution": _ReportSection.SLICE_LOGIC_DISTRIB,
        "3. Memory": _ReportSection.MEMORY,
        "4. DSP": _ReportSection.DSP,
        "5. IO and GT Specific": _ReportSection.IO_GT,
        "6. Clocking": _ReportSection.CLOCKING,
        "7. Specific Feature": _ReportSection.SPECIFIC_FEATURE,
        "8. Primitives": _ReportSection.PRIMITIVES,
        "9. Black Boxes": _ReportSection.BLACK_BOXES,
        "10. Instantiated Netlists": _ReportSection.NETLISTS,
    }


@dataclass(frozen=True)
class TimingData:
    requirement: Final[float]
    slack: Final[float]
    data_path: Final[float]

    @classmethod
    def from_file(cls: Type[Self], filepath: str) -> Self:
        requirement: float | None = None
        slack: float | None = None
        data_path: float | None = None

        def was_data_parsed(name: str, val: float | None):
            if val is None:
                raise ValueError(f"Failed to find {name} in timing report @ {filepath}")

        with open(filepath, "r") as report:
            while line := report.readline():
                line = line.strip()
                if "Requirement:" in line:
                    requirement = float(split_on_whitespace(line)[1][:-2])
                elif "Slack" in line:
                    slack = float(split_on_whitespace(line)[3][:-2])
                elif "Data Path Delay:" in line:
                    data_path = float(split_on_whitespace(line)[3][:-2])

        was_data_parsed("requirement", requirement)
        was_data_parsed("slack", slack)
        was_data_parsed("data path delay", data_path)

        return TimingData(requirement, slack, data_path)  # type: ignore


# ===------------------------------------------------------------------------------=== #
# Function definitions
# ===------------------------------------------------------------------------------=== #


def split_on_whitespace(line: str) -> list[str]:
    tokens: list[str] = []

    last_is_whitespace: bool = True
    start_idx: int = 0
    for i, c in enumerate(line):
        if last_is_whitespace:
            if not c.isspace():
                start_idx = i
                last_is_whitespace = False
        else:
            if c.isspace():
                tokens.append(line[start_idx:i])
                last_is_whitespace = True

    return tokens


def print_list(title: str, elems: Sequence[str]):
    print(title)
    for i, e in enumerate(elems):
        print(f"\t{i}. {e}")


def cla() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Vizualizer",
        description=(
            "Visualize and compare benchmark resource utilization and performance"
        ),
    )

    parser.add_argument(
        "--dot",
        metavar="filename",
        help="Filename of DOT output",
    )
    parser.add_argument(
        "--simulation",
        metavar="filename",
        help="Filename of ModelSim output",
    )
    parser.add_argument(
        "--area",
        metavar="filename",
        help="Filename of area reports",
    )
    parser.add_argument(
        "--timing",
        metavar="filename",
        help="Filename of timing reports",
    )
    parser.add_argument(
        "--flows",
        metavar="names",
        help="Flow names (comma-separated, one per provided report path)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="output-path",
        help="Save visualization to an HTML file (directory must exist)",
    )
    parser.add_argument(
        "paths",
        metavar="report-path",
        nargs="+",
        help="Paths to reports (should include wildcard)",
    )

    return parser.parse_args()


def find_and_parse_reports(
    pattern: str, filename: str, dataType: Type[DataType]
) -> dict[str, DataType]:
    full_path = os.path.join(pattern, filename)
    full_path_tokens = full_path.split(os.path.sep)

    # Look for wildcards in path
    wildcards: list[int] = []
    for i, token in enumerate(full_path_tokens):
        if "*" in token:
            if "**" in token:
                raise ValueError("** wildcards aren't supported")
            if token.count("*") > 1:
                raise ValueError("Only one wildcard is supported per element")

            wildcards.append(i)

    # There must be at least one wildcard in the path pattern for it to match more that
    # one thing
    if len(wildcards) == 0:
        raise ValueError("There are no wildcards in the path")

    # Associate a unique name to each filepath that was matched by the pattern
    reports: dict[str, DataType] = {}

    # Make up a unique name for each file using the parts of the path that were matched
    # by the wildcards
    for fp in glob.glob(full_path):
        fp_tokens = fp.split(os.path.sep)

        # Identify what the first wildcard was replaced with
        wc = wildcards[0]
        pattern_token = full_path_tokens[wc]
        expanded_token = fp_tokens[wc]
        wc_idx = pattern_token.find("*")
        wc_len = len(expanded_token) - len(pattern_token) + 1
        replacement = expanded_token[wc_idx : wc_idx + wc_len]

        # Verify that each other wildcard was replaced by the same thing. If not, ignore
        # the pattern match
        if not all(
            fp_tokens[wc_idx] == full_path_tokens[wc_idx].replace("*", replacement)
            for wc_idx in wildcards[1:]
        ):
            continue

        # Try to parse the data from the file
        data: DataType
        try:
            data = dataType.from_file(fp)
        except ValueError as e:
            print(f"Found report @ {fp} but failed to parse it\n\t-> {e}")
            continue

        # The text that was replaced by the wildcard becomes the name of the benchmark
        reports[replacement] = data

    # Sort the bencmarks in the returned dictionnary in alphabetical order
    return {name: reports[name] for name in sorted(list(reports))}


def get_chart_generator(
    info: DataInfo[DataStorage],
    base_fig_args: BokehParams = None,
    base_vbar_args: BokehParams = None,
) -> ChartGenerator[DataStorage]:
    base_fig_args = {} if base_fig_args is None else base_fig_args
    base_vbar_args = {} if base_vbar_args is None else base_vbar_args

    def get_chart(
        dtype: Type[T],
        extract: Callable[[DataStorage], T],
        positive: bool = True,
        fig_args: BokehParams = None,
        vbar_args: BokehParams = None,
    ) -> figure:
        fig_args = {} if fig_args is None else fig_args
        vbar_args = {} if vbar_args is None else vbar_args

        return nested_barchart(
            info.benchmarks,
            list(info.data),
            {
                flow: np.fromiter((extract(d) for d in flow_data), dtype=dtype)
                for flow, flow_data in info.data.items()
            },
            positive,
            fig_args={
                "height": 400,
                "sizing_mode": "stretch_width",
                **base_fig_args,
                **fig_args,
            },
            vbar_args={**base_vbar_args, **vbar_args},
        )

    return get_chart


def viz_dot(info: DataInfo[DotData]) -> TabPanel:
    chart_gen = get_chart_generator(info)

    def get_category_panel(
        comps: Sequence[DotData.ComponentType], title: str
    ) -> TabPanel:
        figs = [
            chart_gen(
                int,
                lambda d: d.counts.get(comp, 0),
                fig_args={"title": f"Number of {comp.value}"},
            )
            for comp in comps
        ]

        # Split figures, 4 on each row
        rows: list[Row] = []
        idx: int = 0
        while idx < len(figs):
            rows.append(row(*figs[idx : min(len(figs), idx + 4)]))
            idx += 4

        return TabPanel(child=layout(*rows, sizing_mode="stretch_width"), title=title)

    dataflow_panel = get_category_panel(
        DotData.ComponentType.get_dataflow(), "Dataflow"
    )
    arithmetic_panel = get_category_panel(
        DotData.ComponentType.get_arithmetic(), "Arithmetic"
    )
    icmp_panel = get_category_panel(DotData.ComponentType.get_icmp(), "Integer CMP")
    fcmp_panel = get_category_panel(DotData.ComponentType.get_fcmp(), "Floating CMP")

    tabs = Tabs(
        tabs=[dataflow_panel, arithmetic_panel, icmp_panel, fcmp_panel],
        sizing_mode="stretch_width",
    )
    return TabPanel(child=tabs, title="Circuit")


def viz_simulation(info: DataInfo[SimData]) -> TabPanel:
    chart_gen = get_chart_generator(info)
    time = chart_gen(int, lambda d: d.time, fig_args={"title": "Simulation time (ns)"})
    return TabPanel(child=row(time, sizing_mode="stretch_width"), title="Simuation")


def viz_area(info: DataInfo[AreaData]) -> TabPanel:
    chart_gen = get_chart_generator(info)

    # Quick shortcut
    def area_gen(extract: Callable[[AreaData], int], title: str) -> figure:
        return chart_gen(int, extract, fig_args={"title": title})

    lut = area_gen(lambda r: r.luts, "Number of LUTs")
    lut_logic = area_gen(lambda r: r.lut_logic, "Number of LUTs (as logic)")
    lut_ram = area_gen(lambda r: r.lut_ram, "Number of LUTs (as distributed RAM)")
    lut_reg = area_gen(lambda r: r.lut_reg, "Number of LUTs (as shift register)")
    reg = area_gen(lambda r: r.regs, "Number of registers")
    reg_ff = area_gen(lambda r: r.reg_ff, "Number of registers (as flip-flop)")
    reg_latch = area_gen(lambda r: r.reg_latch, "Number of registers (as latch)")
    dsp = area_gen(lambda r: r.dsp, "Number of DSPs")

    panel_lut = TabPanel(
        child=layout(
            row(lut), row(lut_logic, lut_ram, lut_reg), sizing_mode="stretch_width"
        ),
        title="LUT Utilization",
    )
    panel_reg = TabPanel(
        child=layout(row(reg), row(reg_ff, reg_latch), sizing_mode="stretch_width"),
        title="Register Utilization",
    )
    panel_dsp = TabPanel(child=dsp, title="DSP Utilization")

    return TabPanel(
        child=Tabs(tabs=[panel_lut, panel_reg, panel_dsp], sizing_mode="stretch_width"),
        title="Area",
    )


def viz_timing(info: DataInfo[TimingData]) -> TabPanel:
    chart_gen = get_chart_generator(info)

    slack = chart_gen(
        float,
        lambda r: r.slack,
        False,
        fig_args={
            "title": (
                "Slack (ns) | Negative numbers mean that the clock requirement is "
                "unsatisfied"
            )
        },
    )
    data_delay = chart_gen(
        float, lambda r: r.data_path, fig_args={"title": "Data path delay (ns)"}
    )

    return TabPanel(
        child=column(slack, data_delay, sizing_mode="stretch_width"), title="Timing"
    )


def visualizer():
    args = cla()

    paths: list[str] = args.paths
    dot_filename: str | None = args.dot
    sim_filename: str | None = args.simulation
    area_filename: str | None = args.area
    timing_filename: str | None = args.timing
    flows: str | None = args.flows
    output: str | None = args.output

    def get_flow_names() -> tuple[str]:
        # If flow names were provided as arguments, make sure they are unique and
        # consistent with the number of report paths
        if flows is not None:
            flow_names = tuple(map(lambda s: s.strip(), flows.split(",")))
            if len(set(flow_names)) != len(flow_names):
                raise ValueError("Provided flow names must all be different!")
            if len(flow_names) != len(paths):
                raise ValueError(
                    f"Number of flow names must match number of report paths. Expected "
                    f"{len(paths)}, but got {len(flow_names)}."
                )

            return flow_names

        # Use a generic name when there is only one flow
        if len(paths) == 1:
            return ("flow",)

        # Look for a unique token in all path patterns and use it as the flow names
        for tokens in zip(*[p.split(os.path.sep) for p in paths]):
            if len(set(tokens)) == len(tokens):
                return tokens

        # Default on generic flow names
        return tuple(f"flow{i + 1}" for i in range(len(paths)))

    flow_names = get_flow_names()

    def collect_data(filename: str, dataType: Type[DataType]) -> DataInfo[DataType]:
        # Collect set of benchmarks available for each flow
        bench_per_flow: list[set[str]] = []
        data_per_bench: list[dict[str, DataType]] = []

        for p in paths:
            bench_to_data = find_and_parse_reports(p, filename, dataType)
            bench_per_flow.append(set(bench_to_data))
            data_per_bench.append(bench_to_data)

        # Determine set of benchmarks that are available on all flows
        benchmarks: set[str] = bench_per_flow[0]  # there is at least 1 flow so is safe
        for flow_benchmarks in bench_per_flow[1:]:
            benchmarks &= flow_benchmarks
        if len(benchmarks) == 0:
            print(f"No benchmark available for all flows for {dataType}")

        sorted_benchmarks = sorted(benchmarks)
        return DataInfo(
            sorted_benchmarks,
            {
                flow: [data_per_bench[i][name] for name in sorted_benchmarks]
                for i, flow in enumerate(flow_names)
            },
        )

    panels: list[TabPanel] = []

    if dot_filename is not None:
        info = collect_data(dot_filename, DotData)
        panels.append(viz_dot(info))

    if sim_filename is not None:
        info = collect_data(sim_filename, SimData)
        panels.append(viz_simulation(info))

    if area_filename is not None:
        info = collect_data(area_filename, AreaData)
        panels.append(viz_area(info))

    if timing_filename is not None:
        info = collect_data(timing_filename, TimingData)
        panels.append(viz_timing(info))

    # Show the final figure (and potentialyl save it to disk)
    fig = Tabs(tabs=panels, sizing_mode="stretch_width")
    if output is not None:
        save_to_disk(fig, output, "Report")
    show(fig)


if __name__ == "__main__":
    visualizer()
