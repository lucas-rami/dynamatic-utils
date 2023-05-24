# Libraries
import argparse
import glob
import os
import numpy as np
from dataclasses import dataclass
from enum import Enum
from bokeh.layouts import layout, row, column
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.plotting import figure, show

# Local
from utils import nested_barchart, save_to_disk

# Typing
from typing import Any, Callable, Final, Mapping, Sequence


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


@dataclass(frozen=True)
class ReportsInfo:
    flows: Final[Sequence[str]]
    benchmarks: Final[Mapping[str, Sequence[str]]]

    def __post_init__(self):
        if len(self.flows) == 0:
            raise ValueError(
                "Reports instance need to be created with at least one flow"
            )
        if len(self.benchmarks) == 0:
            raise ValueError(
                "Reports instance need to be created with at least one benchmark"
            )

        for name, filepaths in self.benchmarks.items():
            if len(filepaths) != len(self.flows):
                raise ValueError(
                    f"Incorrect number of reports found for benchmark {name}. Expected "
                    f"{len(self.flows)} but got {len(filepaths)}."
                )

    def __len__(self) -> int:
        return len(self.flows)

    def get_names(self) -> list[str]:
        return list(self.benchmarks)


@dataclass(frozen=True)
class UtilizationReport:
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

    @staticmethod
    def from_file(filepath: str) -> "UtilizationReport":
        section: UtilizationReport._ReportSection | None = None

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
                if (new_section := _UTILIZATION_SECTIONS.get(line, None)) is not None:
                    section = new_section
                elif section == UtilizationReport._ReportSection.SLICE_LOGIC:
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
                elif section == UtilizationReport._ReportSection.DSP:
                    if "DSPs" in line:
                        dsp = get_used_column(line)

        return UtilizationReport(lut_logic, lut_ram, lut_reg, reg_ff, reg_latch, dsp)

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


@dataclass(frozen=True)
class TimingReport:
    requirement: Final[float]
    slack: Final[float]
    data_path: Final[float]

    @staticmethod
    def from_file(filepath: str) -> "TimingReport":
        requirement: float | None = None
        slack: float | None = None
        data_path: float | None = None

        def report_error(name: str, val: float | None) -> float:
            if val is None:
                raise ValueError(f"Failed to find {name} in timing report @ {filepath}")
            return val

        with open(filepath, "r") as report:
            while line := report.readline():
                line = line.strip()
                if "Requirement:" in line:
                    requirement = float(split_on_whitespace(line)[1][:-2])
                elif "Slack" in line:
                    slack = float(split_on_whitespace(line)[3][:-2])
                elif "Data Path Delay:" in line:
                    data_path = float(split_on_whitespace(line)[3][:-2])

        report_error("requirement", requirement)
        report_error("slack", slack)
        report_error("data path delay", data_path)

        return TimingReport(requirement, slack, data_path)  # type: ignore


def print_list(title: str, elems: Sequence[str]):
    print(title)
    for i, e in enumerate(elems):
        print(f"\t{i}. {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Viz Vivado",
        description="Visualize and compare synthesization results from Vivado",
    )

    # Positional arguments
    parser.add_argument(
        "--utilization",
        metavar="filename",
        help="Filename of utilization reports",
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


def get_reports(pattern: str, filename: str) -> dict[str, str]:
    filepaths: list[str] = glob.glob(f"{pattern}/{filename}")

    # Associate a unique name to each filepath that was matched by the pattern
    reports: dict[str, str] = {}

    # Look for wildcards in path
    wildcards: list[int] = []
    for i, token in enumerate(pattern.split(os.path.sep)):
        if "**" in token:
            raise ValueError("** wildcards are not supported yet")
        elif "*" in token:
            wildcards.append(i)

    if len(wildcards) == 0:
        raise ValueError("There are no wildcards in the path")

    # Make up a unique name for each file using the parts of the path that were matched
    # by the wildcards
    for fp in filepaths:
        tokens = fp.split(os.path.sep)
        name: str = "_".join(tokens[wc] for wc in wildcards)
        reports[name] = fp

    return {name: reports[name] for name in sorted(list(reports))}


def utilization_reports(info: ReportsInfo) -> TabPanel:
    # For each flow, gather the utilization data on all benchmarks
    results: dict[str, list[UtilizationReport]] = {
        flow: [
            UtilizationReport.from_file(filepaths[i])
            for filepaths in info.benchmarks.values()
        ]
        for i, flow in enumerate(info.flows)
    }

    def gen_data(extract: Callable[[UtilizationReport], int]) -> dict[str, np.ndarray]:
        """Generates data for each barchart."""
        return {
            flow: np.fromiter((extract(rep) for rep in reports), dtype=int)
            for flow, reports in results.items()
        }

    benchmarks = list(info.benchmarks)
    base_fig_args: dict[str, Any] = {
        "height": 400,
        "sizing_mode": "stretch_width",
    }

    def get_chart(extract: Callable[[UtilizationReport], int], title: str) -> figure:
        return nested_barchart(
            benchmarks,
            info.flows,
            gen_data(extract),
            fig_args={"title": title, **base_fig_args},
        )

    lut = get_chart(lambda r: r.luts, "Number of LUTs")
    lut_logic = get_chart(lambda r: r.lut_logic, "Number of LUTs (as logic)")
    lut_ram = get_chart(lambda r: r.lut_ram, "Number of LUTs (as distributed RAM)")
    lut_reg = get_chart(lambda r: r.lut_reg, "Number of LUTs (as shift register)")
    reg = get_chart(lambda r: r.regs, "Number of registers")
    reg_ff = get_chart(lambda r: r.reg_ff, "Number of registers (as flip-flop)")
    reg_latch = get_chart(lambda r: r.reg_latch, "Number of registers (as latch)")
    dsp = get_chart(lambda r: r.dsp, "Number of DSPs")

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


def timing_reports(info: ReportsInfo) -> TabPanel:
    # For each flow, gather the timing data on all benchmarks
    results: dict[str, list[TimingReport]] = {
        flow: [
            TimingReport.from_file(filepaths[i])
            for filepaths in info.benchmarks.values()
        ]
        for i, flow in enumerate(info.flows)
    }

    def gen_data(extract: Callable[[TimingReport], float]) -> dict[str, np.ndarray]:
        """Generates data for each barchart."""
        return {
            flow: np.fromiter((extract(rep) for rep in reports), dtype=float)
            for flow, reports in results.items()
        }

    benchmarks = list(info.benchmarks)
    base_fig_args: dict[str, Any] = {
        "height": 400,
        "sizing_mode": "stretch_width",
    }

    def get_chart(
        extract: Callable[[TimingReport], float], title: str, positive: bool = True
    ) -> figure:
        return nested_barchart(
            benchmarks,
            info.flows,
            gen_data(extract),
            positive,
            fig_args={"title": title, **base_fig_args},
        )

    slack = get_chart(
        lambda r: r.slack,
        "Slack (ns) | Negative numbers mean that the clock requirement is unsatisfied",
        False,
    )

    data_delay = get_chart(lambda r: r.data_path, "Data path delay (ns)")
    return TabPanel(
        child=column(slack, data_delay, sizing_mode="stretch_width"), title="Timing"
    )


def vivado():
    args = parse_args()

    paths: list[str] = args.paths
    util_filename: str | None = args.utilization
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

    def collect_reports(filename: str) -> dict[str, list[str]]:
        all_reports: dict[str, list[str]] = {}
        for p in paths:
            for name, fp in get_reports(p, filename).items():
                if name in all_reports:
                    all_reports[name].append(fp)
                else:
                    all_reports[name] = [fp]

        return {
            name: filepaths
            for name, filepaths in all_reports.items()
            if len(filepaths) == len(paths)
        }

    flow_names = get_flow_names()

    panels: list[TabPanel] = []

    if util_filename is not None:
        panels.append(
            utilization_reports(ReportsInfo(flow_names, collect_reports(util_filename)))
        )

    if timing_filename is not None:
        panels.append(
            timing_reports(ReportsInfo(flow_names, collect_reports(timing_filename)))
        )

    fig = Tabs(tabs=panels, sizing_mode="stretch_width")
    if output is not None:
        save_to_disk(fig, output, "Report")
    show(fig)


_UTILIZATION_SECTIONS: Final[Mapping[str, UtilizationReport._ReportSection]] = {
    "1. Slice Logic": UtilizationReport._ReportSection.SLICE_LOGIC,
    "1.1 Summary of Registers by Type": UtilizationReport._ReportSection.REGISTER_SUMMARY,
    "2. Slice Logic Distribution": UtilizationReport._ReportSection.SLICE_LOGIC_DISTRIB,
    "3. Memory": UtilizationReport._ReportSection.MEMORY,
    "4. DSP": UtilizationReport._ReportSection.DSP,
    "5. IO and GT Specific": UtilizationReport._ReportSection.IO_GT,
    "6. Clocking": UtilizationReport._ReportSection.CLOCKING,
    "7. Specific Feature": UtilizationReport._ReportSection.SPECIFIC_FEATURE,
    "8. Primitives": UtilizationReport._ReportSection.PRIMITIVES,
    "9. Black Boxes": UtilizationReport._ReportSection.BLACK_BOXES,
    "10. Instantiated Netlists": UtilizationReport._ReportSection.NETLISTS,
}

if __name__ == "__main__":
    vivado()
