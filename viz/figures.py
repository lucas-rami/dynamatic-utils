# Libraries
import math
import numpy as np
from dataclasses import dataclass, field
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.layouts import column, LayoutDOM, row
from bokeh.colors import RGB
from bokeh.models.ranges import Range1d
from bokeh.models.layouts import Panel, Tabs
from bokeh.plotting.figure import Figure

# Local
from .parser import BenchStats, IRType, Stats, InstructionStats, IRStats

# Typing
from typing import Callable, Dict, Final, List, Sequence, Tuple, Optional, Any, Mapping

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


def compare_scalar_pair(
    stats: Stats,
    f_stats: Callable[[IRStats], int],
    paths: Tuple[IRType, IRType],
    nest: int = 0,
) -> LayoutDOM:

    path0 = __PATH_INFO[paths[0]]
    path1 = __PATH_INFO[paths[1]]

    fig_height = (__HEIGHT_SCREEN - nest * __HEIGHT_TAB) // 2
    fig_args = {
        "width": __WIDTH_SCREEN - __WIDTH_BOXPLOT,
        "height": fig_height,
        "tools": "save, xwheel_zoom, xpan, reset",
        "toolbar_location": "above",
        "active_drag": "xpan",
        "active_scroll": "xwheel_zoom",
    }

    src_abs = ColumnDataSource(
        {
            "benchmarks": list(stats),
            path0.name: [f_stats(path0.stats(s)) for s in stats.values()],
            path1.name: [f_stats(path1.stats(s)) for s in stats.values()],
        }
    )
    max_y = (
        max(
            max(src_abs.data[path0.name]),
            max(src_abs.data[path1.name]),
        )
        + 1
    )
    fig_abs = figure(
        x_range=FactorRange(*src_abs.data["benchmarks"], bounds="auto"),
        y_range=Range1d(0, max_y),
        title="Absolute number per IR",
        **fig_args,
    )

    shift = 0.19
    width = 0.35

    fig_abs.vbar(
        x=dodge("benchmarks", -shift, range=fig_abs.x_range),
        top=path0.name,
        width=width,
        source=src_abs,
        color=path0.color,
        legend_label=path0.name,
    )

    fig_abs.vbar(
        x=dodge("benchmarks", shift, range=fig_abs.x_range),
        top=path1.name,
        width=width,
        source=src_abs,
        color=path1.color,
        legend_label=path1.name,
    )

    fig_abs.xaxis.group_label_orientation = math.pi / 2
    fig_abs.xaxis.major_label_orientation = math.pi / 2
    fig_abs.xgrid.grid_line_color = None
    fig_abs.legend.location = "top_left"
    fig_abs.legend.orientation = "horizontal"
    fig_abs.legend.click_policy = "hide"

    box_abs = boxplot(
        {
            x[0]: np.array(x[1])
            for x in zip(
                (path0.name, path1.name),
                (src_abs.data[path0.name], src_abs.data[path1.name]),
            )
        },
        fig_args={
            "title": "Distribution per IR",
            "height": fig_height,
            "width": __WIDTH_BOXPLOT,
            "y_range": Range1d(0, max_y),
        },
        boxplot_args=BoxPlotParams(
            upper_box={"fill_color": __COLORS[3]},
            lower_box={"fill_color": __COLORS[4]},
        ),
    )

    counts_rel = [
        f_stats(path1.stats(s)) - f_stats(path0.stats(s)) for s in stats.values()
    ]
    src_rel = ColumnDataSource(
        {
            "benchmarks": list(stats),
            "counts": counts_rel,
            "colors": [path1.color if c > 0 else path0.color for c in counts_rel],
        }
    )

    max_val = max([abs(c) for c in src_rel.data["counts"]])

    fig_rel = figure(
        x_range=fig_abs.x_range,
        y_range=Range1d(-max_val - 1, max_val + 1),
        title=f"Relative number ({path1.name} - {path0.name})",
        **fig_args,
    )

    fig_rel.vbar(
        x="benchmarks",
        top="counts",
        width=0.9,
        source=src_rel,
        color="colors",
        line_color="white",
    )

    fig_rel.xaxis.major_label_orientation = math.pi / 2
    fig_rel.xgrid.grid_line_color = None

    box_rel = boxplot(
        {f"{path1.name} - {path0.name}": np.array(src_rel.data["counts"])},
        fig_args={
            "title": "Difference in distributions",
            "height": fig_height,
            "width": __WIDTH_BOXPLOT,
            "y_range": Range1d(-max_val - 1, max_val + 1),
        },
        boxplot_args=BoxPlotParams(
            upper_box={"fill_color": __COLORS[3]},
            lower_box={"fill_color": __COLORS[4]},
        ),
    )

    return column(row(box_abs, fig_abs), row(box_rel, fig_rel))


def compare_scalar(
    stats: Stats, f_stats: Callable[[IRStats], int], nest: int = 0, opt: bool = False
) -> LayoutDOM:

    fig_height = (__HEIGHT_SCREEN - nest * __HEIGHT_TAB) // 2
    fig_args = {
        "width": __WIDTH_SCREEN - __WIDTH_BOXPLOT,
        "height": fig_height,
        "tools": "save, xwheel_zoom, xpan, reset",
        "toolbar_location": "above",
        "active_drag": "xpan",
        "active_scroll": "xwheel_zoom",
    }

    src_abs = ColumnDataSource(
        {
            "benchmarks": list(stats),
            __LLVM: [f_stats(s.llvm) for s in stats.values()],
            __MLIR: [f_stats(s.mlir) for s in stats.values()],
            __MLIR_OPT: [
                0 if s.mlir_opt is None else f_stats(s.mlir_opt) for s in stats.values()
            ],
        }
    )
    max_y = (
        max(
            max(src_abs.data[__LLVM]),
            max(src_abs.data[__MLIR]),
            max(src_abs.data[__MLIR_OPT]) if opt else 0,
        )
        + 1
    )
    fig_abs = figure(
        x_range=FactorRange(*src_abs.data["benchmarks"], bounds="auto"),
        y_range=Range1d(0, max_y),
        title="Absolute number per IR",
        **fig_args,
    )

    shift = 0.27 if opt else 0.19
    width = 0.25 if opt else 0.35

    fig_abs.vbar(
        x=dodge("benchmarks", -shift, range=fig_abs.x_range),
        top=__LLVM,
        width=width,
        source=src_abs,
        color=__LLVM_COLOR,
        legend_label=__LLVM,
    )

    fig_abs.vbar(
        x=dodge("benchmarks", 0.0 if opt else shift, range=fig_abs.x_range),
        top=__MLIR,
        width=width,
        source=src_abs,
        color=__MLIR_COLOR,
        legend_label=__MLIR,
    )

    if opt:
        fig_abs.vbar(
            x=dodge("benchmarks", shift, range=fig_abs.x_range),
            top=__MLIR_OPT,
            width=width,
            source=src_abs,
            color=__MLIR_OPT_COLOR,
            legend_label=__MLIR_OPT,
        )

    fig_abs.xaxis.group_label_orientation = math.pi / 2
    fig_abs.xaxis.major_label_orientation = math.pi / 2
    fig_abs.xgrid.grid_line_color = None
    fig_abs.legend.location = "top_left"
    fig_abs.legend.orientation = "horizontal"
    fig_abs.legend.click_policy = "hide"

    box_abs = boxplot(
        {
            x[0]: np.array(x[1])
            for x in zip(
                (__LLVM, __MLIR, __MLIR_OPT) if opt else (__LLVM, __MLIR),
                (
                    src_abs.data[__LLVM],
                    src_abs.data[__MLIR],
                    list(filter(lambda x: x != 0, src_abs.data[__MLIR_OPT])),
                ),
            )
        },
        fig_args={
            "title": "Distribution per IR",
            "height": fig_height,
            "width": __WIDTH_BOXPLOT,
            "y_range": Range1d(0, max_y),
        },
        boxplot_args=BoxPlotParams(
            upper_box={"fill_color": __COLORS[3]},
            lower_box={"fill_color": __COLORS[4]},
        ),
    )

    counts_rel = [f_stats(s.mlir) - f_stats(s.llvm) for s in stats.values()]
    counts_rel_opt = [
        0 if s.mlir_opt is None else f_stats(s.mlir_opt) - f_stats(s.llvm)
        for s in stats.values()
    ]
    src_rel = ColumnDataSource(
        {
            "benchmarks": list(stats),
            "counts": counts_rel,
            "colors": [__MLIR_COLOR if c > 0 else __LLVM_COLOR for c in counts_rel],
        }
    )
    src_rel_opt = ColumnDataSource(
        {
            "benchmarks": list(stats),
            "counts": counts_rel_opt,
            "colors": [
                __MLIR_OPT_COLOR if c > 0 else __LLVM_COLOR for c in counts_rel_opt
            ],
        }
    )

    max_val = max(
        (
            max([abs(c) for c in src_rel.data["counts"]]),
            max([abs(c) for c in src_rel_opt.data["counts"]]) if opt else 0,
        )
    )

    def create_fig_rel(src: ColumnDataSource, path: str) -> Figure:

        fig = figure(
            x_range=fig_abs.x_range,
            y_range=Range1d(-max_val - 1, max_val + 1),
            title=f"Relative number ({path} - {__LLVM})",
            **fig_args,
        )

        fig.vbar(
            x="benchmarks",
            top="counts",
            width=0.9,
            source=src,
            color="colors",
            line_color="white",
        )

        fig.xaxis.major_label_orientation = math.pi / 2
        fig.xgrid.grid_line_color = None
        return fig

    fig_rel = create_fig_rel(src_rel, __MLIR)
    fig_rel_opt = create_fig_rel(src_rel_opt, __MLIR_OPT)

    data = {f"{__MLIR} - {__LLVM}": np.array(src_rel.data["counts"])}
    if opt:
        # Make both relative figures half as height
        fig_rel.height = fig_rel.height // 2  # type: ignore
        fig_rel_opt.height = fig_rel_opt.height // 2  # type: ignore

        # Hide x axis for top figure
        fig_rel.xaxis.visible = False

        # Add data to boxplot
        data[f"{__MLIR_OPT} - {__LLVM}"] = np.array(src_rel_opt.data["counts"])

    box_rel = boxplot(
        data,
        fig_args={
            "title": "Distribution (difference between MLIR and LLVM)",
            "height": fig_height,
            "width": __WIDTH_BOXPLOT,
            "y_range": Range1d(-max_val - 1, max_val + 1),
        },
        boxplot_args=BoxPlotParams(
            upper_box={"fill_color": __COLORS[3]},
            lower_box={"fill_color": __COLORS[4]},
        ),
    )

    return column(
        row(box_abs, fig_abs),
        row(box_rel, column(fig_rel, fig_rel_opt) if opt else fig_rel),
    )


def compare_instruction_types(
    stats: Stats, paths: Tuple[IRType, IRType], nest: int = 0
) -> Tabs:
    panels: List[Panel] = []
    for instr_type in sorted(InstructionStats.all_instr_types):
        layout = compare_scalar_pair(
            stats, lambda s: s.instructions.counts_per_type[instr_type], paths, nest + 1
        )
        panels.append(Panel(child=layout, title=instr_type))

    return Tabs(tabs=panels)


def save_to_disk(layout: LayoutDOM) -> None:
    fp = "analysis.html"
    output_file(filename=fp, title="Statistics")
    save(layout)
    print(f'Saved statistics to "{fp}"')


__LLVM: Final[str] = "LLVM"
__MLIR: Final[str] = "MLIR"
__MLIR_OPT: Final[str] = "MLIR (opt)"

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
__MLIR_OPT_COLOR: Final[RGB] = __COLORS[2]
__WIDTH_SCREEN: Final[int] = 1900
__HEIGHT_SCREEN: Final[int] = 1000
__WIDTH_BOXPLOT: Final[int] = 400
__HEIGHT_TAB: Final[int] = 30


@dataclass(frozen=True)
class PathInfo:
    name: str
    color: RGB
    stats: Callable[[BenchStats], IRStats]


__PATH_INFO: Mapping[IRType, PathInfo] = {
    IRType.LLVM: PathInfo(__LLVM, __LLVM_COLOR, lambda b: b.llvm),
    IRType.MLIR: PathInfo(__MLIR, __MLIR_COLOR, lambda b: b.mlir),
    IRType.MLIR_OPT: PathInfo(__MLIR_OPT, __MLIR_OPT_COLOR, lambda b: b.mlir_opt),
}
