# Libraries
from bokeh.io import show
from bokeh.models.layouts import Panel, Tabs
from bokeh.io import show

# Local
from viz.parser import parse, Stats
from viz.figures import *

# Typing
from typing import Sequence, Tuple


def range_bb() -> None:
    stats: Stats = parse()

    def get_range(data: Sequence[Sequence[int]]) -> Tuple[int, int]:
        global_max = -1
        global_min = 100
        for d in data:
            global_max = max(global_max, max(d))
            global_min = min(global_min, min(d))
        return global_min, global_max

    llvm_preds = get_range(
        [bench.llvm.basic_blocks.pred_counts for bench in stats.values()]
    )
    llvm_succs = get_range(
        [bench.llvm.basic_blocks.succ_counts for bench in stats.values()]
    )
    mlir_preds = get_range(
        [bench.mlir.basic_blocks.succ_counts for bench in stats.values()]
    )
    mlir_succs = get_range(
        [bench.mlir.basic_blocks.succ_counts for bench in stats.values()]
    )
    print(f"Range for LLVM predecessors: {llvm_preds}")
    print(f"Range for LLVM successors: {llvm_succs}")
    print(f"Range for MLIR predecessors: {mlir_preds}")
    print(f"Range for MLIR successors: {mlir_succs}")


def run() -> None:
    stats: Stats = parse()

    blocks = Panel(
        child=Tabs(
            tabs=[
                Panel(
                    child=compare_scalar(stats, lambda ir: ir.basic_blocks.counts, 2),
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
                    child=compare_scalar(
                        stats, lambda ir: ir.instructions.n_real_instructions, 2
                    ),
                    title="Global count",
                ),
                Panel(
                    child=compare_instruction_types(stats, 2), title="Count per type"
                ),
            ]
        ),
        title="Instructions",
    )

    tabs = Tabs(tabs=[blocks, instructions])
    save_to_disk(tabs)
    show(tabs)


if __name__ == "__main__":
    run()
