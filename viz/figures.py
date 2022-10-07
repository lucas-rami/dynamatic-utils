# Libraries
import math
import numpy as np
from dataclasses import dataclass, field
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, dodge
from bokeh.layouts import column, LayoutDOM, row
from bokeh.colors import RGB
from bokeh.models.ranges import Range1d
from bokeh.models.layouts import Panel, Tabs
from bokeh.plotting.figure import Figure

# Local
from .parser import Stats, InstructionStats, IRStats

# Typing
from typing import Callable, Dict, Final, List, Sequence, Tuple, Optional, Any

BokehParams = Optional[Dict[str, Any]]


@dataclass
class BoxPlotParams:
    upper_box: Dict[str, Any] = field(default_factory=dict)
    lower_box: Dict[str, Any] = field(default_factory=dict)
    stems: Dict[str, Any] = field(default_factory=dict)
    whiskers: Dict[str, Any] = field(default_factory=dict)
    outliers: Dict[str, Any] = field(default_factory=dict)


def boxplot(
    data: Dict[str, np.ndarray],
    fig_args: BokehParams = None,
    boxplot_args: Optional[BoxPlotParams] = None,
) -> Figure:

    if fig_args is None:
        fig_args = {}
    if boxplot_args is None:
        boxplot_args = BoxPlotParams()

    categories = list(data.keys())
    n_cats = len(categories)

    # Draw figure
    fig_base_args = {
        "x_range": categories,
        "tools": "save",
        "toolbar_location": "above",
    }
    fig = figure(**{**fig_base_args, **fig_args})

    # Variables to store info for each serie
    stats = {
        "cats": categories,
        "q1": np.empty(n_cats),
        "q2": np.empty(n_cats),
        "q3": np.empty(n_cats),
        "upper": np.empty(n_cats),
        "lower": np.empty(n_cats),
    }
    outliers_x: List[str] = []
    outliers_y: List[float] = []

    def gen_stats(i: int, cat: str):
        serie = data[cat]

        # Compute quartiles
        quantiles = np.quantile(serie, [0.25, 0.50, 0.75])
        q1, q2, q3 = quantiles[0], quantiles[1], quantiles[2]
        iqr = q3 - q1

        # Identify outliers
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        outliers = list(np.unique(serie[serie > upper])) + list(
            np.unique(serie[serie < lower])
        )

        # Shrink lengths of stems to be no longer than the minimums or maximums
        upper = min(upper, np.max(serie))
        lower = max(lower, np.min(serie))

        # Save data
        stats["q1"][i] = q1
        stats["q2"][i] = q2
        stats["q3"][i] = q3
        stats["upper"][i] = upper
        stats["lower"][i] = lower

        outliers_x.extend([cat for _ in range(len(outliers))])
        outliers_y.extend(outliers)

    for i, cat in enumerate(categories):
        gen_stats(i, cat)

    src = ColumnDataSource(stats)

    # Boxes
    box_common = {
        "x": "cats",
        "width": 0.7,
        "source": src,
        "line_color": "black",
    }
    upper_box_base_args = {
        "bottom": "q2",
        "top": "q3",
        "fill_color": __COLORS[0],
        **box_common,
    }
    fig.vbar(**{**upper_box_base_args, **boxplot_args.upper_box})
    lower_box_base_args = {
        "bottom": "q1",
        "top": "q2",
        "fill_color": __COLORS[1],
        **box_common,
    }
    fig.vbar(**{**lower_box_base_args, **boxplot_args.lower_box})

    # Stems
    stem_common = {
        "x0": "cats",
        "x1": "cats",
        "source": src,
        "line_color": "black",
    }
    upper_stem_base_args = {"y0": "upper", "y1": "q3", **stem_common}
    fig.segment(**{**upper_stem_base_args, **boxplot_args.stems})
    lower_stem_base_args = {"y0": "lower", "y1": "q1", **stem_common}
    fig.segment(**{**lower_stem_base_args, **boxplot_args.stems})

    # Whiskers (almost-0 height rects simpler than segments)
    whisker_common = {
        "x": "cats",
        "width": 0.2,
        "height": 0.01,
        "source": src,
        "line_color": "black",
    }
    upper_whisker_base_args = {"y": "upper", **whisker_common}
    fig.rect(**{**upper_whisker_base_args, **boxplot_args.whiskers})
    lower_whisker_base_args = {"y": "lower", **whisker_common}
    fig.rect(**{**lower_whisker_base_args, **boxplot_args.whiskers})

    # Outliers
    outliers_base_args = {
        "x": outliers_x,
        "y": outliers_y,
        "size": 10,
        "color": __COLORS[-1],
        "fill_alpha": 0.6,
    }
    fig.circle(**{**outliers_base_args, **boxplot_args.outliers})

    return fig


def compare_scalar(stats: Stats, f_stats: Callable[[IRStats], int]) -> LayoutDOM:

    fig_args = {
        "width": __WIDTH_SCREEN - __WIDTH_BOXPLOT,
        "height": __HEIGHT_FIG,
        "tools": "save, xwheel_zoom, xpan, reset",
        "toolbar_location": "above",
        "active_drag": "xpan",
        "active_scroll": "xwheel_zoom",
    }

    src_abs = ColumnDataSource(
        {
            "benchmarks": list(stats),
            "llvm": [f_stats(s.llvm) for s in stats.values()],
            "mlir": [f_stats(s.mlir) for s in stats.values()],
        }
    )
    fig_abs = figure(
        x_range=FactorRange(*src_abs.data["benchmarks"], bounds="auto"),
        y_range=Range1d(
            0, max(max(src_abs.data["llvm"]), max(src_abs.data["mlir"])) + 1
        ),
        title="Absolute number per IR",
        **fig_args,
    )

    fig_abs.vbar(
        x=dodge("benchmarks", -0.2, range=fig_abs.x_range),
        top=__COMPILE_PATHS[0],
        width=0.3,
        source=src_abs,
        color=__LLVM_COLOR,
        legend_label="llvm",
    )

    fig_abs.vbar(
        x=dodge("benchmarks", 0.2, range=fig_abs.x_range),
        top=__COMPILE_PATHS[1],
        width=0.3,
        source=src_abs,
        color=__MLIR_COLOR,
        legend_label="mlir",
    )

    fig_abs.xaxis.group_label_orientation = math.pi / 2
    fig_abs.xaxis.major_label_orientation = math.pi / 2
    fig_abs.xgrid.grid_line_color = None
    fig_abs.legend.location = "top_left"
    fig_abs.legend.orientation = "horizontal"
    fig_abs.legend.click_policy = "mute"

    box_abs = boxplot(
        {
            x[0]: np.array(x[1])
            for x in zip(__COMPILE_PATHS, (src_abs.data["llvm"], src_abs.data["mlir"]))
        },
        fig_args={
            "title": "Distribution per IR",
            "height": __HEIGHT_FIG,
            "width": __WIDTH_BOXPLOT,
        },
        boxplot_args=BoxPlotParams(
            upper_box={"fill_color": __COLORS[2]},
            lower_box={"fill_color": __COLORS[3]},
        ),
    )

    counts_rel = [f_stats(s.mlir) - f_stats(s.llvm) for s in stats.values()]
    src_rel = ColumnDataSource(
        {
            "benchmarks": list(stats),
            "counts": counts_rel,
            "colors": [__MLIR_COLOR if c > 0 else __LLVM_COLOR for c in counts_rel],
        }
    )
    max_val = max([abs(c) for c in src_rel.data["counts"]])
    fig_rel = figure(
        x_range=fig_abs.x_range,
        y_range=Range1d(-max_val - 1, max_val + 1),
        title=f"Relative number (MLIR - LLVM)",
        **fig_args,
    )

    fig_rel.vbar(
        x="benchmarks",
        top="counts",
        width=0.9,
        source=src_rel,
        line_color="white",
        fill_color="colors",
    )

    fig_rel.xaxis.major_label_orientation = math.pi / 2
    fig_rel.xgrid.grid_line_color = None

    box_rel = boxplot(
        {"difference": np.array(src_rel.data["counts"])},
        fig_args={
            "title": "Distribution (MLIR - LLVM)",
            "height": __HEIGHT_FIG,
            "width": __WIDTH_BOXPLOT,
        },
        boxplot_args=BoxPlotParams(
            upper_box={"fill_color": __COLORS[2]},
            lower_box={"fill_color": __COLORS[3]},
        ),
    )

    return column(row(box_abs, fig_abs), row(box_rel, fig_rel))


def compare_instruction_types(stats: Stats) -> Tabs:
    panels: List[Panel] = []
    for instr_type in sorted(InstructionStats.all_instr_types):
        layout = compare_scalar(
            stats,
            lambda s: s.instructions.counts_per_type[instr_type],
        )
        panels.append(Panel(child=layout, title=instr_type))

    return Tabs(tabs=panels)


def save_to_disk(layout: LayoutDOM) -> None:
    fp = "analysis.html"
    output_file(filename=fp, title="Statistics")
    save(layout)
    print(f'Saved statistics to "{fp}"')


__COMPILE_PATHS: Final[Sequence[str]] = ("llvm", "mlir")

__COLORS: Final[Sequence[RGB]] = [
    RGB(2, 62, 138),  # dark blue
    RGB(255, 150, 131),  # orange
    RGB(144, 221, 240),  # light blue
    RGB(76, 185, 68),  # green
    RGB(196, 90, 146),  # purple
    RGB(119, 110, 135),  # gray-ish
    RGB(185, 46, 40),  # red
    RGB(167, 117, 77),  # brown
]
__LLVM_COLOR: Final[RGB] = __COLORS[0]
__MLIR_COLOR: Final[RGB] = __COLORS[1]
__WIDTH_BENCH: Final[int] = 50
__WIDTH_SCREEN: Final[int] = 1900
__WIDTH_BOXPLOT: Final[int] = 400
__HEIGHT_FIG: Final[int] = 500
