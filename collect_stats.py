# Libraries
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models.layouts import Panel, Tabs

# Local
from viz.parser import parse, Stats
from viz.figures import *


def run() -> None:
    stats: Stats = parse()

    blocks = Panel(
        child=compare_scalar(stats, lambda ir: ir.n_blocks, "BBs"),
        title="Basic Blocks",
    )
    instructions = Panel(
        child=Tabs(
            tabs=[
                Panel(
                    child=compare_scalar(
                        stats, lambda ir: ir.instructions.counts, "instructions"
                    ),
                    title="All",
                ),
                Panel(
                    child=compare_scalar(
                        stats,
                        lambda ir: ir.instructions.n_real_instructions,
                        '"real"instructions',
                    ),
                    title='"Real"',
                ),
                Panel(child=compare_instruction_types(stats), title="Per type"),
            ]
        ),
        title="Instructions",
    )

    tabs = Tabs(tabs=[blocks, instructions])
    save_to_disk(tabs)
    show(tabs)


if __name__ == "__main__":
    run()
