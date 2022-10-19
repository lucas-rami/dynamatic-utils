# Libraries
import os
from bokeh.io import show
from bokeh.models.layouts import Panel, Tabs
from bokeh.io import show

# Local
from viz.parser import parse, Stats
from viz.figures import *


def gen_figures(stats: Stats, paths: Tuple[str, str], nest: int = 0) -> Tabs:
    if len(stats) == 0:
        return Tabs(tabs=[])

    blocks = Panel(
        child=Tabs(
            tabs=[
                Panel(
                    child=compare_scalar_pair(
                        stats, lambda ir: ir.basic_blocks.counts, paths, nest + 2
                    ),
                    title="Count",
                )
            ],
        ),
        title="Basic Blocks",
    )
    instructions = Panel(
        child=Tabs(
            tabs=[
                Panel(
                    child=compare_scalar_pair(
                        stats,
                        lambda ir: ir.instructions.n_real_instructions,
                        paths,
                        nest + 2,
                    ),
                    title="Global count",
                ),
                Panel(
                    child=compare_instruction_types(stats, paths, nest + 2),
                    title="Count per type",
                ),
            ]
        ),
        title="Instructions",
    )

    tabs = Tabs(tabs=[blocks, instructions])
    return tabs


def run() -> None:
    stats_dyn: Stats = parse(os.path.join("benchmarks", "dynamatic"))
    stats_poly: Stats = parse(os.path.join("benchmarks", "polybench"))

    c = 0
    for s in stats_dyn.values():
        if s.mlir_opt is None:
            c += 1

    dynamatic = gen_figures(
        {
            name: stats
            for name, stats in stats_dyn.items()
            if not (stats.llvm.is_empty or stats.mlir.is_empty)
        },
        ("llvm", "mlir"),
        1,
    )
    polybench = gen_figures(
        {
            name: stats
            for name, stats in stats_poly.items()
            if not (stats.mlir.is_empty or stats.mlir_opt.is_empty)
        },
        ("mlir", "mlir_opt"),
        1,
    )

    tabs = Tabs(
        tabs=[
            Panel(child=dynamatic, title="Dynamatic"),
            Panel(child=polybench, title="Polybench"),
        ]
    )
    save_to_disk(tabs)
    show(tabs)


if __name__ == "__main__":
    run()
