# Libraries
import argparse
import glob
import os
import math
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


class GeomeanGenerator(Generic[DataStorage], Protocol):
    def __call__(
        self,
        dtype: Type[T],
        name: str,
        extract: Callable[[DataStorage], T],
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
    bb_count: Final[int]
    counts: Final[Mapping["DotData.ComponentType", int]]

    @property
    def num_components(self) -> int:
        return sum(self.counts.values())

    @classmethod
    def from_file(cls: Type[Self], filepath: str) -> Self:
        bb_count: int = 0
        counts: dict[DotData.ComponentType, int] = {}

        # Add the passed component to the total counts
        def count(comp: DotData.ComponentType | None):
            if comp is not None:
                counts[comp] = counts.get(comp, 0) + 1
            else:
                raise ValueError(f"Found unknown component in DOT\n\tIn: {line}")

        # Decode the component using Dynamatic++ convention
        def decode(attributes: dict[str, str]):
            if (mlir_attr := attributes.get("mlir_op", None)) is not None:
                return count(_DECODE_COMPONENT_TYPE.get(mlir_attr, None))
            raise ValueError(f"Failed to decode component type\n\tIn: {line}")

        # Decode the component using legacy Dynamatic convention
        def decode_legacy(attributes: dict[str, str]):
            if (type_attr := attributes.get("type", None)) is not None:
                if type_attr == "Operator":
                    if (op_attr := attributes.get("op", None)) is not None:
                        return count(_DECODE_COMPONENT_TYPE_LEGACY.get(op_attr, None))
                    raise ValueError(f"Failed to decode component type\n\tIn: {line}")
                return count(_DECODE_COMPONENT_TYPE_LEGACY.get(type_attr, None))
            raise ValueError(f"Failed to decode component type\n\tIn: {line}")

        # Iterate over all lines in the DOT to find subgraph and node declarations
        with open(filepath, "r") as report:
            while line := report.readline():
                if dot_parser.is_subgraph_decl(line):
                    bb_count += 1
                elif dot_parser.is_node(line):
                    attributes = dot_parser.get_attributes(line)
                    if "mlir_op" in attributes:
                        decode(attributes)
                    else:
                        decode_legacy(attributes)

        return cls(bb_count, counts)

    class ComponentType(Enum):
        # Dataflow components
        ENTRY = ("handshake.arg", "Entry")
        CONTROL_MERGE = ("handshake.control_merge", "CntrlMerge")
        MERGE = ("handshake.merge", "Merge")
        MUX = ("handshake.mux", "Mux")
        BRANCH = ("handshake.br", "<does not exist>")
        CBRANCH = ("handshake.cond_br", "Branch")
        BUFFER = ("handshake.buffer", "Buffer")
        FORK = ("handshake.fork", "Fork")
        LAZY_FORK = ("handshake.lazy_fork", "Lazy fork")
        SINK = ("handshake.sink", "Sink")
        SOURCE = ("handshake.source", "Source")
        CONSTANT = ("handshake.constant", "Constant")
        EXIT = ("handshake.end", "Exit")
        RETURN = ("handshake.d_return", "ret_op")
        # Memory components
        MEMORY_CONTROLLER = ("handshake.mem_controller", "MC")
        LSQ = ("handshake.lsq", "LSQ")
        MC_LOAD = ("handshake.mc_load", "mc_load_op")
        MC_STORE = ("handshake.mc_store", "mc_store_op")
        LSQ_LOAD = ("handshake.lsq_load", "lsq_load_op")
        LSQ_STORE = ("handshake.lsq_store", "lsq_store_op")
        # Arithmetic components
        SELECT = ("arith.select", "select_op")
        INDEX_CAST = ("arith.extui", "zext_op")
        ADDI = ("arith.addi", "add_op")
        ADDF = ("arith.addf", "fadd_op")
        SUBI = ("arith.subi", "sub_op")
        SUBF = ("arith.subf", "fsub_op")
        ANDI = ("arith.andi", "and_op")
        ORI = ("arith.ori", "or_op")
        XORI = ("arith.xori", "xor_op")
        MULI = ("arith.muli", "mul_op")
        MULF = ("arith.mulf", "fmul_op")
        DIVUI = ("arith.divui", "udiv_op")
        DIVSI = ("arith.divsi", "sdiv_op")
        DIVF = ("arith.divf", "fdiv_op")
        SITOFP = ("arith.sitofp", "sitofp_op")
        REMSI = ("arith.remsi", "urem_op")
        EXTSI = ("arith.extsi", "sext_op")
        EXTUI = ("arith.extui", "zext_op")
        TRUNCI = ("arith.trunci", "trunc_op")
        SHRSI = ("arith.shrsi", "ashr_op")
        SHLI = ("arith.shli", "shl_op")
        GET_ELEMENT_PTR = ("arith.get_element_ptr", "getelementptr_op")
        # Integer comparisons
        EQ = ("arith.cmpi==", "icmp_eq_op")
        NE = ("arith.cmpi!=", "icmp_ne_op")
        SLT = ("arith.cmpi<", "icmp_slt_op")
        SLE = ("arith.cmpi,<=", "icmp_sle_op")
        SGT = ("arith.cmpi>", "icmp_sgt_op")
        SGE = ("arith.cmpi>=", "icmp_sge_op")
        ULT = ("arith.cmpi<", "icmp_ult_op")
        ULE = ("arith.cmpi<=", "icmp_ule_op")
        UGT = ("arith.cmpi>", "icmp_ugt_op")
        UGE = ("arith.cmpi>=", "icmp_uge_op")
        # FLoating comparisons
        ALWAYS_FALSE = ("arith.cmpffalse", "fcmp_false_op")
        F_OEQ = ("arith.cmpf==", "fcmp_oeq_op")
        F_OGT = ("arith.cmpf>", "fcmp_ogt_op")
        F_OGE = ("arith.cmpf>=", "fcmp_oge_op")
        F_OLT = ("arith.cmpf<", "fcmp_olt_op")
        F_OLE = ("arith.cmpf<=", "fcmp_ole_op")
        F_ONE = ("arith.cmpf!=", "fcmp_one_op")
        F_ORD = ("arith.cmpfordered?", "fcmp_orq_op")
        F_UEQ = ("arith.cmpf==", "fcmp_ueq_op")
        F_UGT = ("arith.cmpf>", "fcmp_ugt_op")
        F_UGE = ("arith.cmpf>=", "fcmp_uge_op")
        F_ULT = ("arith.cmpf<", "fcmp_ult_op")
        F_ULE = ("arith.cmpf<=", "fcmp_ule_op")
        F_UNE = ("arith.cmpf!=", "fcmp_une_op")
        F_UNO = ("arith.cmpfunordered?", "fcmp_uno_op")
        ALWAYS_TRUE = ("arith.cmpftrue", "fcmp_true_op")

        @staticmethod
        def get_dataflow() -> list["DotData.ComponentType"]:
            return list(
                filter(
                    lambda comp: comp.value[0].startswith("handshake.")
                    and not (
                        comp.value[1].lower().startswith("mc")
                        or comp.value[1].lower().startswith("lsq")
                    ),
                    DotData.ComponentType,
                )
            )

        @staticmethod
        def get_memory() -> list["DotData.ComponentType"]:
            return list(
                filter(
                    lambda comp: comp.value[1].lower().startswith("mc")
                    or comp.value[1].lower().startswith("lsq"),
                    DotData.ComponentType,
                )
            )

        @staticmethod
        def get_arithmetic() -> list["DotData.ComponentType"]:
            return list(
                filter(
                    lambda comp: comp.value[0].startswith("arith.")
                    and not comp.value[0].startswith("arith.cmp"),
                    DotData.ComponentType,
                )
            )

        @staticmethod
        def get_icmp() -> list["DotData.ComponentType"]:
            return list(
                filter(
                    lambda comp: comp.value[0].startswith("arith.cmpi"),
                    DotData.ComponentType,
                )
            )

        @staticmethod
        def get_fcmp() -> list["DotData.ComponentType"]:
            return list(
                filter(
                    lambda comp: comp.value[0].startswith("arith.cmpf"),
                    DotData.ComponentType,
                )
            )

        @staticmethod
        def build_decoder() -> dict[str, "DotData.ComponentType"]:
            return {comp.value[0]: comp for comp in DotData.ComponentType}

        @staticmethod
        def build_legacy_decoder() -> dict[str, "DotData.ComponentType"]:
            return {comp.value[1]: comp for comp in DotData.ComponentType}


_DECODE_COMPONENT_TYPE: Final[
    Mapping[str, DotData.ComponentType]
] = DotData.ComponentType.build_decoder()


_DECODE_COMPONENT_TYPE_LEGACY: Final[
    Mapping[str, DotData.ComponentType]
] = DotData.ComponentType.build_legacy_decoder()


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
        return cls(time)


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
                elif "DSPs" in line:
                    dsp = get_used_column(line)

        return cls(lut_logic, lut_ram, lut_reg, reg_ff, reg_latch, dsp)


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


def geometric_mean(data: np.ndarray) -> float:
    filtered_data = data[data != 0]
    if len(filtered_data) == 0:
        return 0.0
    logs = np.log(filtered_data)
    return np.exp(logs.mean())


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
        "--period",
        metavar="period",
        type=float,
        default=4,
        help="Target period to assume in ns (only relevant for simulation results)",
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
    full_path = pattern + filename
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

        fig: figure = nested_barchart(
            info.benchmarks,
            list(info.data),
            {
                flow: np.fromiter((extract(d) for d in flow_data), dtype=dtype)
                for flow, flow_data in info.data.items()
            },
            positive=positive,
            fig_args={
                "height": 400,
                "sizing_mode": "stretch_width",
                "tools": "reset,save,xpan,xwheel_zoom,ywheel_zoom",
                "active_drag": "xpan",
                "active_scroll": "xwheel_zoom",
                **base_fig_args,
                **fig_args,
            },
            vbar_args={**base_vbar_args, **vbar_args},
        )
        fig.xaxis.major_label_orientation = math.pi / 2
        return fig

    return get_chart


def get_geomean_generator(
    info: DataInfo[DataStorage],
    base_fig_args: BokehParams = None,
    base_vbar_args: BokehParams = None,
) -> GeomeanGenerator[DataStorage]:
    base_fig_args = {} if base_fig_args is None else base_fig_args
    base_vbar_args = {} if base_vbar_args is None else base_vbar_args

    def geomean_chart_gen(
        dtype: Type[T],
        name: str,
        extract: Callable[[DataStorage], T],
        fig_args: BokehParams = None,
        vbar_args: BokehParams = None,
    ) -> figure:
        fig_args = {} if fig_args is None else fig_args
        vbar_args = {} if vbar_args is None else vbar_args
        return nested_barchart(
            [name],
            list(info.data),
            {
                flow: np.array(
                    [
                        geometric_mean(
                            np.fromiter((extract(d) for d in flow_data), dtype=dtype)
                        )
                    ]
                )
                for flow, flow_data in info.data.items()
            },
            legend=False,
            fig_args={
                "height": 400,
                "width": 250,
                "title": "Geometric mean",
                **base_fig_args,
                **fig_args,
            },
            vbar_args={**base_vbar_args, **vbar_args},
        )

    return geomean_chart_gen


def viz_dot(info: DataInfo[DotData]) -> TabPanel:
    chart_gen = get_chart_generator(info)
    num_figs_per_row: int = 2

    def get_category_panel(
        comps: Sequence[DotData.ComponentType], title: str
    ) -> TabPanel:
        figs = [
            chart_gen(
                int,
                lambda d: d.counts.get(comp, 0),
                fig_args={"title": f"Number of {comp.value[0]}"},
            )
            for comp in comps
        ]

        # Split figures, 4 on each row
        rows: list[Row] = []
        idx: int = 0
        while idx < len(figs):
            rows.append(row(*figs[idx : min(len(figs), idx + num_figs_per_row)]))
            idx += num_figs_per_row
        return TabPanel(child=layout(*rows, sizing_mode="stretch_width"), title=title)  # type: ignore

    dataflow_groups: dict[str, set[DotData.ComponentType]] = {
        "Merge-like components": {
            DotData.ComponentType.MERGE,
            DotData.ComponentType.CONTROL_MERGE,
            DotData.ComponentType.MUX,
        },
        "Branch-like components": {
            DotData.ComponentType.BRANCH,
            DotData.ComponentType.CBRANCH,
        },
        "Memory operations": {
            DotData.ComponentType.MC_LOAD,
            DotData.ComponentType.MC_STORE,
            DotData.ComponentType.LSQ_LOAD,
            DotData.ComponentType.LSQ_STORE,
        },
        "Memory interfaces": {
            DotData.ComponentType.MEMORY_CONTROLLER,
            DotData.ComponentType.LSQ,
        },
    }

    meta_dataflow_rows: list[Row] = [
        row(
            chart_gen(
                int,
                lambda d: sum(d.counts.get(comp, 0) for comp in components),
                fig_args={"title": title},
            )
        )
        for title, components in dataflow_groups.items()
    ]

    meta_panel = TabPanel(
        child=layout(
            row(
                chart_gen(
                    int,
                    lambda d: d.bb_count,
                    fig_args={"title": f"Number of basic blocks"},
                )
            ),
            row(
                chart_gen(
                    int,
                    lambda d: d.num_components,
                    fig_args={"title": f"Number of components"},
                )
            ),
            *meta_dataflow_rows,
            sizing_mode="stretch_width",
        ),
        title="Meta",
    )

    dataflow_panel = get_category_panel(
        DotData.ComponentType.get_dataflow(), "Dataflow"
    )
    memory_panel = get_category_panel(DotData.ComponentType.get_memory(), "Memory")
    arithmetic_panel = get_category_panel(
        DotData.ComponentType.get_arithmetic(), "Arithmetic"
    )
    icmp_panel = get_category_panel(DotData.ComponentType.get_icmp(), "Integer CMP")
    fcmp_panel = get_category_panel(DotData.ComponentType.get_fcmp(), "Floating CMP")

    tabs = Tabs(
        tabs=[
            meta_panel,
            dataflow_panel,
            memory_panel,
            arithmetic_panel,
            icmp_panel,
            fcmp_panel,
        ],
        sizing_mode="stretch_width",
    )
    return TabPanel(child=tabs, title="Circuit")  # type: ignore


def viz_simulation(info: DataInfo[SimData], period: float) -> TabPanel:
    chart_gen = get_chart_generator(info)
    geo_gen = get_geomean_generator(info)
    extract = lambda d: d.time / period
    time = chart_gen(
        int,
        extract,
        fig_args={"title": "Simulation time (clock cycles)"},
    )
    geo = geo_gen(float, "Simulation time", extract)
    return TabPanel(
        child=row(time, geo, sizing_mode="stretch_width"), title="Simulation"
    )  # type: ignore


def viz_area(info: DataInfo[AreaData]) -> TabPanel:
    chart_gen = get_chart_generator(info)
    geo_gen = get_geomean_generator(info)

    # Quick shortcut
    def area_gen(
        extract: Callable[[AreaData], int], title: str
    ) -> tuple[figure, figure]:
        return (
            chart_gen(int, extract, fig_args={"title": title}),
            geo_gen(int, title, extract),
        )

    lut, geo_lut = area_gen(lambda r: r.luts, "Number of LUTs")
    lut_logic, geo_lut_logic = area_gen(
        lambda r: r.lut_logic, "Number of LUTs (as logic)"
    )
    lut_ram, geo_lut_ram = area_gen(
        lambda r: r.lut_ram, "Number of LUTs (as distributed RAM)"
    )
    lut_reg, geo_lut_reg = area_gen(
        lambda r: r.lut_reg, "Number of LUTs (as shift register)"
    )
    reg, geo_reg = area_gen(lambda r: r.regs, "Number of registers")
    reg_ff, geo_reg_ff = area_gen(
        lambda r: r.reg_ff, "Number of registers (as flip-flop)"
    )
    reg_latch, geo_reg_latch = area_gen(
        lambda r: r.reg_latch, "Number of registers (as latch)"
    )
    dsp, geo_dsp = area_gen(lambda r: r.dsp, "Number of DSPs")

    panel_lut = TabPanel(
        child=layout(
            row(lut, geo_lut),
            row(lut_logic, geo_lut_logic, lut_ram, geo_lut_ram, lut_reg, geo_lut_reg),
            sizing_mode="stretch_width",
        ),
        title="LUT Utilization",
    )
    panel_reg = TabPanel(
        child=layout(
            row(reg, geo_reg),
            row(reg_ff, geo_reg_ff, reg_latch, geo_reg_latch),
            sizing_mode="stretch_width",
        ),
        title="Register Utilization",
    )
    panel_dsp = TabPanel(
        child=row(dsp, geo_dsp, sizing_mode="stretch_width"), title="DSP Utilization"
    )

    return TabPanel(
        child=Tabs(tabs=[panel_lut, panel_reg, panel_dsp], sizing_mode="stretch_width"),
        title="Area",
    )  # type: ignore


def viz_timing(info: DataInfo[TimingData]) -> TabPanel:
    chart_gen = get_chart_generator(info)
    geo_gen = get_geomean_generator(info)

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
    extract_data_delay = lambda r: r.requirement - r.slack
    data_delay = chart_gen(
        float,
        extract_data_delay,
        fig_args={"title": "Critical path delay (ns)"},
    )
    geo_delay = geo_gen(float, "Critical path delay (ns)", extract_data_delay)

    return TabPanel(
        child=layout(
            row(slack), row(data_delay, geo_delay), sizing_mode="stretch_width"
        ),
        title="Timing",
    )  # type: ignore


def visualizer():
    args = cla()

    paths: list[str] = args.paths
    dot_filename: str | None = args.dot
    sim_filename: str | None = args.simulation
    area_filename: str | None = args.area
    timing_filename: str | None = args.timing
    flows: str | None = args.flows
    period: float = args.period
    output: str | None = args.output

    def get_flow_names() -> tuple[str, ...]:
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
        panels.append(viz_simulation(info, period))

    if area_filename is not None:
        info = collect_data(area_filename, AreaData)
        panels.append(viz_area(info))

    if timing_filename is not None:
        info = collect_data(timing_filename, TimingData)
        panels.append(viz_timing(info))

    # Show the final figure (and potentialyl save it to disk)
    fig: Tabs = Tabs(tabs=panels, sizing_mode="stretch_width")  # type: ignore
    if output is not None:
        save_to_disk(fig, output, "Report")
    show(fig)


if __name__ == "__main__":
    visualizer()
