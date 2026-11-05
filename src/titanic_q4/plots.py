"""Reusable count and distribution visualisations for Titanic analysis."""

from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _hue_palette(df: pd.DataFrame, hue: str | None, palette: list | None):
    """Build a palette that matches the observed grouping categories.

    Args:
        df: Input data frame used by the graph.
        hue: Optional grouping column.
        palette: Optional ordered list of colours.

    Returns:
        The observed hue categories and their colour mapping, or (None, None).
    """
    if hue is None:
        return None, None
    levels = list(df[hue].dropna().unique())
    if not levels:
        return None, None
    if palette is None:
        colors = sns.color_palette("Blues", n_colors=len(levels))
    else:
        colors = [palette[i % len(palette)] for i in range(len(levels))]
    return levels, dict(zip(levels, colors))


def plot_count_grid(
    df: pd.DataFrame,
    columns: list[str],
    hue: str | None = None,
    palette: list | None = None,
    columns_per_row: int = 3,
) -> None:
    """Draw categorical count plots for several columns.

    Args:
        df: Data to visualise.
        columns: Categorical columns to show on separate subplots.
        hue: Optional grouping column, such as set, Survived or Pclass.
        palette: Optional list of plot colours.
        columns_per_row: Number of subplots in each row.

    Returns:
        None. Displays a figure with the requested count plots.
    """
    ncols = min(columns_per_row, len(columns))
    nrows = ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    hue_order, colours = _hue_palette(df, hue, palette)
    for ax, column in zip(axes.flat, columns):
        if hue is None:
            sns.countplot(data=df, x=column, color="#576CBC", ax=ax)
        else:
            sns.countplot(
                data=df, x=column, hue=hue, hue_order=hue_order,
                palette=colours, ax=ax,
            )
        ax.set_title(f"Number of passengers / {column}")
        ax.grid(axis="y", color="gray", linestyle="-.", linewidth=0.5)
        ax.set_axisbelow(True)
        if ax.get_legend() is not None and len(columns) > 1:
            ax.get_legend().remove()
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    if hue is not None and len(columns) > 1:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, title=hue, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=max(1, len(labels)))
    fig.tight_layout()
    plt.show()


def plot_numeric_distributions(
    df: pd.DataFrame,
    columns: list[str],
    hue: str | None = None,
    palette: list | None = None,
    bins: int = 30,
) -> None:
    """Plot histogram distributions of numeric Titanic variables.

    Args:
        df: Data to visualise.
        columns: Numeric columns to plot, such as Age and Fare.
        hue: Optional group column, such as set or Survived.
        palette: Optional list of colours.
        bins: Number of histogram bins.

    Returns:
        None. Displays a figure with a histogram for each variable.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4), squeeze=False)
    hue_order, colours = _hue_palette(df, hue, palette)
    for ax, column in zip(axes.flat, columns):
        args = {"data": df, "x": column, "bins": bins, "ax": ax,
                "element": "step", "alpha": 0.5}
        if hue is not None:
            args.update({"hue": hue, "hue_order": hue_order, "palette": colours})
        sns.histplot(**args)
        ax.set_title(f"Distribution of {column}" + (f" by {hue}" if hue else ""))
        ax.grid(axis="y", linestyle="-.", linewidth=0.5, color="gray")
        ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()
