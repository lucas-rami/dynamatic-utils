# Libraries
import os
from bokeh.io import show
from bokeh.models.layouts import Panel, Tabs
from bokeh.io import show

# Local
from viz.parser import IRType, parse, Stats
from viz.figures import *


def gen_figures(stats: Stats, paths: Tuple[IRType, IRType], nest: int = 0) -> Tabs:
    # Filter out empty statistics
    filt_stats: Stats = {
        name: stats
        for name, stats in stats.items()
        if not (stats.ir(paths[0]).is_empty or stats.ir(paths[1]).is_empty)
    }

    # Don't do anything if there aren't any benchmarks left
    if len(filt_stats) == 0:
        return Tabs(tabs=[])

    # Generate panels
    blocks = Panel(
        child=Tabs(
            tabs=[
                Panel(
                    child=compare_scalar_pair(
                        filt_stats, lambda ir: ir.basic_blocks.counts, paths, nest + 2
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
                        filt_stats,
                        lambda ir: ir.instructions.n_real_instructions,
                        paths,
                        nest + 2,
                    ),
                    title="Global count",
                ),
                Panel(
                    child=compare_instruction_types(filt_stats, paths, nest + 2),
                    title="Count per type",
                ),
            ]
        ),
        title="Instructions",
    )

    # Make tabs with all panels and return
    tabs = Tabs(tabs=[blocks, instructions])
    return tabs


def main() -> None:
    stats_dyn: Stats = parse(os.path.join("benchmarks", "dynamatic"))
    stats_poly: Stats = parse(os.path.join("benchmarks", "polybench"))

    dynamatic = gen_figures(
        {
            name: stats
            for name, stats in stats_dyn.items()
            if not (stats.llvm.is_empty or stats.mlir.is_empty)
        },
        (IRType.LLVM, IRType.MLIR),
        1,
    )
    polybench = gen_figures(
        {
            name: stats
            for name, stats in stats_poly.items()
            if not (stats.llvm.is_empty or stats.mlir.is_empty)
        },
        (IRType.LLVM, IRType.MLIR),
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
    main()
