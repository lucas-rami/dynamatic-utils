# Libraries
import numpy as np
import pandas as pd
from sys import argv
from math import floor, pi
from dataclasses import dataclass, field
from scipy.stats import distributions  # type: ignore
from bokeh.plotting import figure
from bokeh.layouts import column, layout, row
from bokeh.io import save, output_file
from bokeh.colors import RGB
from bokeh.transform import factor_cmap
from bokeh.models.annotations import ColorBar, LabelSet
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.tickers import BasicTicker
from bokeh.models.layouts import LayoutDOM
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from bokeh.core.has_props import HasProps

# Typing
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


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

SPRING_COLOR: Final[Mapping[str, str]] = {
    "yellow": "gold",
    "green": "forestgreen",
    "blue": "royalblue",
    "brown": "saddlebrown",
    "red": "crimson",
    "purple": "purple",
}

GOAT_COLOR: Final[Mapping[str, str]] = {
    "baby": "navajowhite",
    "mama": "lightgray",
    "papa": "dimgray",
}

SCREEN_W: Final[int] = 2500  # 2100
SCREEN_H: Final[int] = 1100

_Params = Optional[Dict[str, Any]]


def barchart(
    data: np.ndarray,
    best_norm_fit: bool = False,
    bin_size: Optional[int] = None,
    fig_args: _Params = None,
    vbar_args: _Params = None,
    line_args: _Params = None,
) -> figure:
    # Arguments dictionaries
    if fig_args is None:
        fig_args = {}
    if vbar_args is None:
        vbar_args = {}
    if line_args is None:
        line_args = {}

    if len(data) == 0:
        return missing_data(fig_args)

    # Very basic binning strategy
    if bin_size is not None:
        data_int = np.empty(data.shape, dtype=int)
        for i in range(len(data)):
            data_int[i] = floor(data[i] / bin_size)
        data = data_int

    # Determine extent of X axis
    min_val = np.min(data)
    max_val = np.max(data)
    x_range = Range1d(min_val - 0.5, max_val + 0.5)

    # Determine bar heights
    uniques, counts = np.unique(data, return_counts=True)
    x: np.ndarray
    if data.dtype.type is np.float64:
        x = uniques
        full_counts = counts
    else:
        x = np.arange(min_val, max_val + 1)
        full_counts = np.zeros(len(x), dtype=int)
        full_counts[uniques - min_val] = counts

    # Draw figure
    fig_base_args = {
        "x_range": x_range,
        "y_range": Range1d(0, np.max(counts)),
        "tools": "save",
        "toolbar_location": "above",
    }
    fig = figure(**{**fig_base_args, **fig_args})

    # Draw vertical bars
    vbar_base_args = {"x": x, "top": full_counts, "color": COLORS[0]}
    fig.vbar(**{**vbar_base_args, **vbar_args})

    # Draw line for best normal fit if requested
    if best_norm_fit:
        # Generate data for normal
        mean, var = distributions.norm.fit(data)  # type:ignore
        fitted_data = distributions.norm.pdf(x, mean, var)  # type:ignore

        # Add extra y range
        y_max = np.max(fitted_data)
        fig.extra_y_ranges = {"fit": Range1d(start=0.0, end=y_max * 1.1)}  # type: ignore

        # Draw the line
        line_base_args = {
            "x": x,
            "y": fitted_data,
            "line_color": COLORS[0],
            "line_width": 4,
            "legend_label": "Best normal fit",
            "y_range_name": "fit",
        }
        fig.line(**{**line_base_args, **line_args})

    return fig


def cat_barchart(
    data: Dict[str, int],
    fig_args: _Params = None,
    vbar_args: _Params = None,
) -> figure:
    # Arguments dictionaries
    if fig_args is None:
        fig_args = {}
    if vbar_args is None:
        vbar_args = {}

    if len(data) == 0:
        return missing_data(fig_args)
    categories = list(data)
    counts = list(data.values())
    src = ColumnDataSource(data=dict(categories=categories, counts=counts))

    # Draw figure
    fig_base_args = {
        "x_range": categories,
        "tools": "save",
        "toolbar_location": "above",
        "y_range": Range1d(0, np.max(counts)),
    }
    fig = figure(**{**fig_base_args, **fig_args})

    # Draw vertical bars
    vbar_base_args = {
        "x": "categories",
        "top": "counts",
        "source": src,
        "width": 0.9,
        "line_color": "white",
        "fill_color": factor_cmap("categories", palette=COLORS, factors=list(data)),
    }
    fig.vbar(**{**vbar_base_args, **vbar_args})
    return fig


def nested_barchart(
    categories: Sequence[str],
    data: Dict[str, np.ndarray],
    fig_args: _Params = None,
    vbar_args: _Params = None,
) -> figure:
    # Arguments dictionaries
    if fig_args is None:
        fig_args = {}
    if vbar_args is None:
        vbar_args = {}

    if len(categories) == 0 or len(data) == 0:
        return missing_data(fig_args)

    x = [(s, d) for s in categories for d in data]
    counts = sum(zip(*list(data.values())), ())
    src = ColumnDataSource(data=dict(x=x, counts=counts))

    # Draw figure
    fig_base_args = {
        "x_range": FactorRange(*x),
        "tools": "save",
        "toolbar_location": "above",
        "y_range": Range1d(0, np.max(counts)),
    }
    fig = figure(**{**fig_base_args, **fig_args})

    # Draw vertical bars
    vbar_base_args = {
        "x": "x",
        "top": "counts",
        "width": 0.9,
        "source": src,
        "line_color": "white",
        "fill_color": factor_cmap(
            "x", palette=COLORS, factors=list(data), start=1, end=2
        ),
    }
    fig.vbar(**{**vbar_base_args, **vbar_args})
    return fig


def missing_data(fig_args: _Params = None) -> figure:
    if fig_args is None:
        fig_args = {}
    assert fig_args is not None

    fig_base_args = {
        "x_range": Range1d(0.0, 1.0),
        "y_range": Range1d(0.0, 1.0),
        "title": "Missing data",
        "tools": "",
        "toolbar_location": "above",
    }
    fig = figure(**{**fig_base_args, **fig_args})
    disable_movement(fig)

    fig.x(x=0.5, y=0.5, line_width=5, size=100, color="red")
    return fig


def linechart(
    data: Dict[str, Union[Sequence[float], Sequence[Sequence[float]]]],
    fig_args: _Params = None,
    line_args: _Params = None,
    varea_args: _Params = None,
) -> figure:
    if fig_args is None:
        fig_args = {}
    if line_args is None:
        line_args = {}
    if varea_args is None:
        varea_args = {}
    assert fig_args is not None
    assert line_args is not None
    assert varea_args is not None

    # Determine if we have variance data
    x_len: int = 0
    has_var_data: bool = False
    for _, v in data.items():
        if isinstance(v[0], (list, np.ndarray)):
            has_var_data = True
            x_len = len(v[0])
        else:
            has_var_data = False
            x_len = len(v)
        break
    x = np.arange(x_len)

    # Create the figure
    fig = figure(x_range=Range1d(0, x_len - 1), tools="save", **fig_args)

    # Iterate over all lines
    for i, (k, v) in enumerate(data.items()):
        # Create the line
        line = np.mean(v, axis=0, dtype=float) if has_var_data else v
        line_base_args = {
            "x": x,
            "y": line,
            "line_width": 3,
            "legend_label": k,
            "color": COLORS[i % len(COLORS)],
        }
        fig.line(**{**line_base_args, **line_args})

        # If we have variance data, create area around line
        if has_var_data:
            std = np.std(v, axis=0, dtype=np.float64)

            df = ColumnDataSource(
                pd.DataFrame(
                    data={
                        "x": x,
                        "lower_std": line - std,
                        "upper_std": line + std,
                    }
                )
            )

            varea_base_args = {
                "fill_alpha": 0.2,
                "legend_label": k,
                "color": COLORS[i % len(COLORS)],
            }

            fig.varea(
                **{**varea_base_args, **varea_args},
                x="x",
                y1="lower_std",
                y2="upper_std",
                source=df,
            )

    # Make legend clickable
    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    return fig


@dataclass
class BoxPlotParams:
    upper_box: Dict[str, Any] = field(default_factory=dict)
    lower_box: Dict[str, Any] = field(default_factory=dict)
    stems: Dict[str, Any] = field(default_factory=dict)
    whiskers: Dict[str, Any] = field(default_factory=dict)
    outliers: Dict[str, Any] = field(default_factory=dict)


def boxplot(
    data: Dict[str, np.ndarray],
    fig_args: _Params = None,
    boxplot_args: Optional[BoxPlotParams] = None,
) -> figure:
    if fig_args is None:
        fig_args = {}
    if boxplot_args is None:
        boxplot_args = BoxPlotParams()
    assert fig_args is not None
    assert boxplot_args is not None

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
        "fill_color": COLORS[0],
        **box_common,
    }
    fig.vbar(**{**upper_box_base_args, **boxplot_args.upper_box})
    lower_box_base_args = {
        "bottom": "q1",
        "top": "q2",
        "fill_color": COLORS[1],
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
        "color": COLORS[-1],
        "fill_alpha": 0.6,
    }
    fig.circle(**{**outliers_base_args, **boxplot_args.outliers})

    return fig


_ROW: Final[str] = "row"
_COL: Final[str] = "col"
_HM_VAL: Final[str] = "val"
_VAL_TXT: Final[str] = "val"
_TXT_COLOR: Final[str] = "color"


@dataclass
class HeatMapParams:
    mapper: Dict[str, Any] = field(default_factory=dict)
    rect: Dict[str, Any] = field(default_factory=dict)
    colorbar: Dict[str, Any] = field(default_factory=dict)
    display_colorbar: bool = True


def heatmap(
    df: Union[pd.DataFrame, Sequence[Tuple[str, str, float]]],
    font_size: str = "16px",
    fig_args: _Params = None,
    heatmap_args: Optional[HeatMapParams] = None,
) -> figure:
    if fig_args is None:
        fig_args = {}
    if heatmap_args is None:
        heatmap_args = HeatMapParams()

    # Transform into dataframe if necessary
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=[_ROW, _COL, _HM_VAL])

    rows = df[_ROW].unique()
    cols = df[_COL].unique()

    # Add columns for displaying text on cells
    df[_TXT_COLOR] = list(map(lambda r: "white" if r < 0.5 else "black", df[_HM_VAL]))
    df[_VAL_TXT] = list(map(lambda p: f"{p:.3f}", df[_HM_VAL]))

    # Create figure
    fig_base_args = {
        "x_range": cols,
        "y_range": list(reversed(rows)),
        "x_axis_location": "below",
        "toolbar_location": "above",
        "tools": "save",
    }
    fig = figure(**{**fig_base_args, **fig_args})
    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_text_font_size = font_size
    fig.axis.major_label_standoff = 0
    fig.xaxis.major_label_orientation = pi / 4

    src = ColumnDataSource(df)

    # Create color mapper and heatmap
    mapper_def_args = {
        "palette": "Magma256",
        "low": 0.0,
        "high": 1.0,
        "nan_color": "lightgray",
    }
    mapper = LinearColorMapper(**{**mapper_def_args, **heatmap_args.mapper})
    rect_def_args = {
        "x": _COL,
        "y": _ROW,
        "width": 1,
        "height": 1,
        "source": src,
        "fill_color": {"field": _HM_VAL, "transform": mapper},
        "line_color": None,
    }
    fig.rect(**{**rect_def_args, **heatmap_args.rect})

    # Add labels on heatmap
    labels_def_args = {
        "x": _COL,
        "y": _ROW,
        "text": _VAL_TXT,
        "text_align": "center",
        "text_font_size": font_size,
        "text_font_style": "bold",
        "text_color": _TXT_COLOR,
        "source": src,
        "render_mode": "canvas",
    }
    labels_corr = LabelSet(**labels_def_args)
    fig.add_layout(labels_corr)

    # Add color bar on the side
    if heatmap_args.display_colorbar:
        colorbar_def_args = {
            "color_mapper": mapper,
            "major_label_text_font_size": font_size,
            "ticker": BasicTicker(desired_num_ticks=11),
            "formatter": PrintfTickFormatter(format="%.2f"),
            "label_standoff": 6,
            "border_line_color": None,
            "location": (0, 0),
        }
        color_bar = ColorBar(**{**colorbar_def_args, **heatmap_args.colorbar})
        fig.add_layout(color_bar, "right")

    return fig


_Range = Union[DataRange1d, FactorRange, HasProps, Range1d]


def fig_align(
    figs: Sequence[figure],
    x_range: Union[_Range, str] = "none",
    y_range: Union[_Range, str] = "none",
) -> Sequence[figure]:
    # Argument validation
    err = '{} must be either "auto", "none", or a Range1d instance.'
    if isinstance(x_range, str) and x_range not in ("auto", "none"):
        raise ValueError(err.format("x_range"))
    if isinstance(y_range, str) and y_range not in ("auto", "none"):
        raise ValueError(err.format("y_range"))

    def range_align(ranges: Sequence[_Range]) -> Optional[_Range]:
        if len(ranges) == 0:
            return None
        elif isinstance(ranges[0], Range1d):
            start: float = min(r.start for r in ranges)  # type: ignore
            end: float = max(r.end for r in ranges)  # type: ignore
            return Range1d(start, end)
        raise ValueError(f"Unsupported range type {type(ranges[0])}.")

    def get_range(
        range: Union[_Range, str], f_range: Callable[[figure], _Range]
    ) -> Optional[_Range]:
        if isinstance(range, str):
            if range == "none":
                # "none" means figures won't be aligned on that axis
                return None

            # range must be "auto"
            # Determine range automatically based on figure
            return range_align([f_range(f) for f in figs])
        return range

    # Align figures by tweaking their axes ranges
    if (x := get_range(x_range, lambda f: f.x_range)) is not None:
        for f in figs:
            f.x_range = x  # type: ignore
    if (y := get_range(y_range, lambda f: f.y_range)) is not None:
        for f in figs:
            f.y_range = y  # type: ignore
    return figs


def disable_movement(fig: figure) -> None:
    fig.toolbar.active_drag = None  # type: ignore
    fig.toolbar.active_scroll = None  # type: ignore
    fig.toolbar.active_tap = None  # type: ignore


def simple_grid(figures: list[list[figure]], filepath: str, title: str) -> None:
    rows = [row(*figs) for figs in figures]
    save_to_disk(layout(*rows), filepath, title)


def simple_column(figures: Sequence[figure], filepath: str, title: str) -> None:
    save_to_disk(column(*figures), filepath, title)


def simple_row(figures: Sequence[figure], filepath: str, title: str) -> None:
    save_to_disk(
        row(*figures),
        filepath,
        title,
    )


def save_to_disk(layout: LayoutDOM, filepath: str, title: str) -> None:
    output_file(filename=filepath, title=title)
    save(layout)
    print(f'Saved statistics to "{filepath}"')
