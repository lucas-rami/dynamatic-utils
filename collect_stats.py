import os
import math
from dataclasses import dataclass

from bokeh.io import output_file, show, save
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.layouts import row, column, layout, LayoutDOM
from bokeh.colors import RGB
from bokeh.models.ranges import Range1d
from bokeh.models.layouts import Panel, Tabs
from bokeh.plotting.figure import Figure

from typing import Callable, Dict, Final, Sequence, Tuple

PATH: Final[str] = os.path.join("benchmarks")
LIBRARIES: Final[Sequence[str]] = ("llvm", "mlir")

COLORS: Final[Sequence[RGB]] = [
    RGB(2, 62, 138),  # dark blue
    RGB(255, 150, 131),  # orange
    RGB(144, 221, 240),  # light blue
    RGB(76, 185, 68),  # green
    RGB(196, 90, 146),  # purple
    RGB(119, 110, 135),  # gray-ish
    RGB(185, 46, 40),  # red
    RGB(167, 117, 77),  # brown
]
WIDTH_BENCH: Final[int] = 30


@dataclass(frozen=True)
class IRStats:
    n_blocks: int
    n_instructions: int

    @staticmethod
    def from_file(fp: str) -> "IRStats":
        with open(fp, "r") as f:
            n_blocks = int(f.readline())
            n_instructions = int(f.readline())
            return IRStats(n_blocks, n_instructions)


@dataclass(frozen=True)
class BenchStats:
    llvm: IRStats
    mlir: IRStats

    @staticmethod
    def from_file(bench: str) -> "BenchStats":
        llvm = IRStats.from_file(path_llvm(bench))
        mlir = IRStats.from_file(path_mlir(bench))
        return BenchStats(llvm, mlir)


def path_llvm(name: str) -> str:
    return os.path.join(PATH, name, "llvm", "stats.txt")


def path_mlir(name: str) -> str:
    return os.path.join(PATH, name, "mlir", "stats.txt")


def compare_scalar(
    stats: Dict[str, BenchStats], f_stats: Callable[[IRStats], int], scalar_name: str
) -> Tuple[Figure, Figure]:
    def chart_abs() -> Figure:
        x = [(bench, ir) for bench in stats for ir in LIBRARIES]
        llvm_blocks = [f_stats(s.llvm) for s in stats.values()]
        mlir_blocks = [f_stats(s.mlir) for s in stats.values()]
        counts = sum(zip(llvm_blocks, mlir_blocks), ())

        source = ColumnDataSource(data=dict(x=x, counts=counts))
        p = figure(
            x_range=FactorRange(*x, range_padding=0.1),
            y_range=Range1d(0, max(counts) + 1),
            width=(WIDTH_BENCH * len(stats)),
            height=500,
            title=f"Number of {scalar_name} per IR",
            toolbar_location=None,
            tools="",
        )

        p.vbar(
            x="x",
            top="counts",
            width=0.9,
            source=source,
            line_color="white",
            fill_color=factor_cmap(
                "x", palette=COLORS, factors=LIBRARIES, start=1, end=2
            ),
        )

        p.xaxis.group_label_orientation = math.pi / 2
        p.xaxis.major_label_orientation = math.pi / 2
        p.xgrid.grid_line_color = None

        return p

    def chart_rel() -> Figure:
        x = list(stats)
        counts = [f_stats(s.mlir) - f_stats(s.llvm) for s in stats.values()]
        max_val = max([abs(c) for c in counts])

        source = ColumnDataSource(data=dict(x=x, counts=counts))
        p = figure(
            x_range=FactorRange(*x, range_padding=0.1),
            y_range=Range1d(-max_val - 1, max_val + 1),
            width=(WIDTH_BENCH * len(stats)),
            height=500,
            title=f"Relative number of {scalar_name} in MLIR compared to LLVM IR",
            toolbar_location=None,
            tools="",
        )

        p.vbar(
            x="x",
            top="counts",
            width=0.9,
            source=source,
            line_color="white",
            fill_color=COLORS[2],
        )

        p.xaxis.major_label_orientation = math.pi / 2
        p.xgrid.grid_line_color = None

        return p

    return chart_abs(), chart_rel()


def save_to_disk(layout: LayoutDOM) -> None:
    fp = "analysis.html"
    output_file(filename=fp, title="Statistics")
    save(layout)
    print(f'Saved statistics to "{fp}"')


def run() -> None:
    benchmarks = sorted(os.listdir(PATH))
    print(f"Detected {len(benchmarks)} benchmarks")

    print(f"Parsing statistics")
    n_fail: int = 0
    stats: Dict[str, BenchStats] = {}
    for bench in benchmarks:
        try:
            stats[bench] = BenchStats.from_file(bench)
        except:
            n_fail += 1
    print(f"\t{n_fail}/{len(benchmarks)} benchmarks failed to parse")

    blocks = Panel(
        child=column(*compare_scalar(stats, lambda ir: ir.n_blocks, "BBs")),
        title="Basic Blocks",
    )
    instructions = Panel(
        child=column(
            *compare_scalar(stats, lambda ir: ir.n_instructions, "instructions")
        ),
        title="Instructions",
    )

    tabs = Tabs(tabs=[blocks, instructions])
    save_to_disk(tabs)
    show(tabs)


if __name__ == "__main__":
    run()
