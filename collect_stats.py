# Libraries
from bokeh.io import show
from bokeh.models.layouts import Panel, Tabs
from bokeh.io import show

# Local
from viz.parser import parse, Stats
from viz.figures import *


def run() -> None:
    stats: Stats = parse()

    blocks = Panel(
        child=compare_scalar(stats, lambda ir: ir.n_blocks),
        title="Basic Blocks",
    )
    instructions = Panel(
        child=Tabs(
            tabs=[
                Panel(
                    child=compare_scalar(stats, lambda ir: ir.instructions.counts),
                    title="All",
                ),
                Panel(
                    child=compare_scalar(
                        stats,
                        lambda ir: ir.instructions.n_real_instructions,
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
