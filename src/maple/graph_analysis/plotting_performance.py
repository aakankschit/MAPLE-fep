import itertools
import os
from typing import Optional, Union, Tuple

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy import stats
import seaborn as sns

from .performance_stats import bootstrap_statistic

# Optional IPython support for notebook display
try:
    from IPython.display import IFrame, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def _is_notebook():
    """Check if code is running in a Jupyter notebook environment.

    Returns
    -------
    bool
        True if running in a Jupyter notebook, False otherwise
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:  # Jupyter notebook or qtconsole
            return True
    except (ImportError, AttributeError):
        pass
    return False


def _master_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    xerr: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,
    method_name: str = "",
    target_name: str = "",
    quantity: str = "$\\Delta \\Delta$ G",
    xlabel: str = "Experimental",
    ylabel: str = "Calculated",
    units: str = "$\\mathrm{kcal\\,mol^{-1}}$",
    guidelines: bool = True,
    origins: bool = True,
    color: Optional[str] = None,
    statistics: list = ["RMSE", "MUE"],
    filename: Optional[str] = None,
    centralizing: bool = True,
    shift: float = 0.0,
    figsize: float = 10,
    dpi: Union[float, str] = "figure",
    data_labels: list = [],
    axis_padding: float = 0.5,
    xy_lim: list = [],
    font_sizes: dict = {"title": 20, "labels": 14, "other": 14},
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
    scatter_kwargs: dict = {"s": 10, "marker": "o"},
    results_dir: str = "results",
    nbootstrap: int = 1000,
):
    """Create a scatter plot with summary statistics.

    This function creates a scatter plot comparing experimental vs calculated values
    with error bars, guidelines, and summary statistics displayed in the title.

    Parameters
    ----------
    x : np.ndarray
        Experimental values (x-axis)
    y : np.ndarray
        Calculated values (y-axis)
    title : str, default = ""
        Title for the plot
    xerr : np.ndarray, optional
        Error bars for x values
    yerr : np.ndarray, optional
        Error bars for y values
    method_name : str, optional
        Name of method associated with results, e.g. 'perses'
    target_name : str, optional
        Name of system for results, e.g. 'Thrombin'
    quantity : str, default = "$\\Delta \\Delta$ G"
        Metric that is being plotted
    xlabel : str, default = "Experimental"
        Label for x-axis
    ylabel : str, default = "Calculated"
        Label for y-axis
    units : str, default = "$\\mathrm{kcal\\,mol^{-1}}$"
        Units to label axis
    guidelines : bool, default = True
        Whether to plot grey 0.5 and 1 kcal/mol error zones
    origins : bool, default = True
        Whether to plot x and y axis lines
    color : str, optional
        Color for points. If None, colored by distance from unity line
    statistics : list, default = ["RMSE", "MUE"]
        List of statistics to calculate and report on the plot
    filename : str, optional
        Filename to save plot
    centralizing : bool, default = True
        Whether to offset the free energies
    shift : float, default = 0.0
        Shift both x and y axis by a constant
    figsize : float, default = 5
        Size of figure for matplotlib
    dpi : float or 'figure', default 'figure'
        Resolution in dots per inch
    data_labels : list, default = []
        List of labels for each data point
    axis_padding : float, default = 0.5
        Padding to add to maximum axis value and subtract from minimum
    xy_lim : list, default = []
        Contains minimum and maximum values for x and y axes
    font_sizes : dict, default {"title": 12, "labels": 9, "other": 12}
        Font sizes for title, data labels, and rest of plot
    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        Type of statistic to use: 'mle' (sample statistic) or 'mean' (bootstrapped mean)
    scatter_kwargs : dict, default {"s": 10, "marker": "o"}
        Arguments to control plt.scatter()
    nbootstrap : int, default = 1000
        Number of bootstrap samples to draw for computing statistics and confidence intervals

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    nsamples = len(x)

    # Set up aesthetics
    plt.rcParams["xtick.labelsize"] = font_sizes["other"]
    plt.rcParams["ytick.labelsize"] = font_sizes["other"]
    plt.rcParams["font.size"] = font_sizes["other"]

    fig = plt.figure(figsize=(figsize, figsize))
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

    plt.xlabel(f"{xlabel} {quantity} ({units})")
    plt.ylabel(f"{ylabel} {quantity} ({units})")

    # Set axis limits
    if xy_lim:
        ax_min, ax_max = xy_lim
        scale = xy_lim
    else:
        ax_min = min(min(x), min(y)) - axis_padding
        ax_max = max(max(x), max(y)) + axis_padding
        scale = [ax_min, ax_max]

    plt.xlim(scale)
    plt.ylim(scale)

    # Plot x-axis and y-axis lines
    if origins:
        plt.plot([0, 0], scale, "gray")
        plt.plot(scale, [0, 0], "gray")

    # Plot x=y line
    plt.plot(scale, scale, "k:")

    # Plot guidelines (error zones)
    if guidelines:
        small_dist = 0.5
        # Plot grey region around x=y line
        plt.fill_between(
            scale,
            [ax_min - small_dist, ax_max - small_dist],
            [ax_min + small_dist, ax_max + small_dist],
            color="grey",
            alpha=0.2,
        )
        plt.fill_between(
            scale,
            [ax_min - small_dist * 2, ax_max - small_dist * 2],
            [ax_min + small_dist * 2, ax_max + small_dist * 2],
            color="grey",
            alpha=0.2,
        )

    # Set up color mapping
    cm = plt.get_cmap("coolwarm")
    if color is None:
        color = np.abs(x - y)
        # 2.372 kcal / mol = 4 RT
        color = cm(color / 2.372)

    # Plot error bars and scatter points
    plt.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        color="gray",
        linewidth=0.0,
        elinewidth=2.0,
        zorder=1,
    )
    plt.scatter(x, y, color=color, zorder=2, **scatter_kwargs)

    # Add data labels if provided
    if len(data_labels) > 0:
        texts = []
        for i, label in enumerate(data_labels):
            texts.append(
                plt.text(x[i] + 0.03, y[i] + 0.03, label, fontsize=font_sizes["labels"])
            )
        adjust_text(texts)

    # Calculate and display statistics
    statistics_string = ""
    if statistic_type not in ["mle", "mean"]:
        raise ValueError(f"Unknown statistic type {statistic_type}")

    for statistic in statistics:
        s = bootstrap_statistic(
            x,
            y,
            xerr,
            yerr,
            statistic=statistic,
            nbootstrap=nbootstrap,
            include_true_uncertainty=bootstrap_x_uncertainty,
            include_pred_uncertainty=bootstrap_y_uncertainty,
        )
        string = (
            f"{statistic}:   {s[statistic_type]:.2f} [95%: {s['low']:.2f}, {s['high']:.2f}] "
            + "\n"
        )
        statistics_string += string

    # Create title with statistics
    long_title = f"{title} \n {target_name} (N = {nsamples}) \n {statistics_string}"

    plt.title(
        long_title,
        fontsize=font_sizes["title"],
        loc="right",
        horizontalalignment="right",
        family="monospace",
    )

    # Save or show plot
    if filename is None:
        # Check if running in notebook
        if _is_notebook():
            # In Jupyter, just return figure - it will auto-display
            return fig
        else:
            # In scripts, show the plot
            plt.show()
            return fig
    else:
        # Ensure filename has .pdf extension
        if not filename.endswith('.pdf'):
            filename = filename.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(filename, bbox_inches="tight", dpi=dpi, format='pdf')
        plt.close(fig)
        return fig


def plot_dataset_DDGs(
    dataset,
    experimental_column: str = "Experimental DeltaDeltaG",
    predicted_column: str = None,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    map_positive: bool = False,
    filename: Optional[str] = None,
    symmetrise: bool = False,
    data_label_type: str = None,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
    statistics: list = None,
    figsize: float = 5,
    nbootstrap: int = 1000,
    **kwargs,
):
    """Plot relative free energies (ΔΔG) from a dataset with node model estimates.

    This function extracts relative free energy data from a dataset and creates
    a scatter plot comparing experimental vs calculated values with statistics.

    Parameters
    ----------
    dataset : FEPDataset
        Dataset object containing edge data with experimental and predicted values
    experimental_column : str, default = "DeltaG"
        Name of column containing experimental relative free energies
    predicted_column : str, optional
        Name of column containing predicted relative free energies.
        If None, will try to find common prediction column names like "MAP", "VI", etc.
    method_name : str, optional
        Name of method associated with results, e.g. 'perses'
    target_name : str, optional
        Name of system for results, e.g. 'Thrombin'
    title : str, default = ""
        Title for the plot
    map_positive : bool, default=False
        Whether to map all DDGs to positive x values (aesthetic choice)
    filename : str, optional
        Filename to save plot
    symmetrise : bool, default = False
        Whether to plot each datapoint twice, both positive and negative
    data_label_type : str or None, default = None
        Type of data label to add to each edge:
        - None: No labels
        - 'small-molecule': Edge labels as "A→B" format
        - 'protein-mutation': Edge labels as "A29B" format
    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        Type of statistic to use: 'mle' (maximum likelihood estimate) or 'mean' (bootstrap mean)
    statistics : list, optional
        List of statistics to compute and display in plot title. Default: ['RMSE', 'MUE']
        Available options: 'RMSE', 'MUE', 'R2', 'rho' (Pearson correlation),
        'KTAU' (Kendall's Tau), 'RAE' (Relative Absolute Error)
    figsize : float, default = 5
        Size of the figure (creates a square plot of figsize x figsize inches)
    nbootstrap : int, default = 1000
        Number of bootstrap samples to draw for computing statistics and confidence intervals
    **kwargs
        Additional arguments passed to _master_plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object

    Examples
    --------
    >>> # Basic usage with default statistics (RMSE, MUE)
    >>> fig = plot_dataset_DDGs(dataset, predicted_column='MAP')
    >>>
    >>> # Show correlation statistics
    >>> fig = plot_dataset_DDGs(
    ...     dataset,
    ...     predicted_column='VI',
    ...     statistics=['RMSE', 'R2', 'rho', 'KTAU']
    ... )
    >>>
    >>> # Custom target and method names
    >>> fig = plot_dataset_DDGs(
    ...     dataset,
    ...     predicted_column='GMVI',
    ...     target_name='CDK8',
    ...     method_name='FEP+',
    ...     statistics=['RMSE', 'MUE', 'rho']
    ... )
    """
    assert (
        int(symmetrise) + int(map_positive) != 2
    ), "Symmetrise and map_positive cannot both be True in the same plot"

    # Get edge data - only handle MAPLE dataset structure
    if not hasattr(dataset, "dataset_edges"):
        raise ValueError("Dataset must have 'dataset_edges' attribute")
    edges_df = dataset.dataset_edges

    # Extract experimental values
    if experimental_column not in edges_df.columns:
        raise ValueError(
            f"Experimental column '{experimental_column}' not found in dataset edges"
        )
    x = edges_df[experimental_column].values

    # Find predicted column if not specified
    common_prediction_columns = ["MAP", "VI", "GMVI", "MLE", "WCC", "Pred. DeltaG"]
    if predicted_column is None:
        # Try common MAPLE prediction column names
        pred_columns_found = []
        for col in common_prediction_columns:
            if col in edges_df.columns:
                pred_columns_found.append(col)
        print(f"Found predicted columns: {pred_columns_found}")
        # Randomly select a predicted column
        predicted_column = pred_columns_found[0]
        print(f"Using predicted column: {predicted_column}")

    elif predicted_column not in common_prediction_columns:
        raise ValueError(
            f"Predicted column '{predicted_column}' not found in common prediction columns"
            f"Common prediction columns must be one of: {common_prediction_columns}"
        )
    elif predicted_column in common_prediction_columns:
        print(f"Using predicted column: {predicted_column}")

    y = edges_df[predicted_column].values

    # Filter out NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    # Handle error columns (optional)
    xerr = None
    yerr = None

    # Look for error columns
    if f"{experimental_column} Error" in edges_df.columns:
        xerr = edges_df[f"{experimental_column} Error"].values[valid_mask]

    if f"{predicted_column}_uncertainty" in edges_df.columns:
        yerr = edges_df[f"{predicted_column}_uncertainty"].values[valid_mask]
    elif f"{predicted_column} Error" in edges_df.columns:
        yerr = edges_df[f"{predicted_column} Error"].values[valid_mask]

    # Create data labels if requested
    data_labels = []
    if data_label_type is not None:
        sources = edges_df["Source"].values[valid_mask]
        destinations = edges_df["Destination"].values[valid_mask]

        if data_label_type == "small-molecule":
            data_labels = [f"{src}→{dst}" for src, dst in zip(sources, destinations)]
        elif data_label_type == "protein-mutation":
            data_labels = [f"{src}{dst}" for src, dst in zip(sources, destinations)]

    # Create title if not provided
    if not title:
        if method_name and target_name:
            title = f"{method_name} - {target_name}"
        elif method_name:
            title = method_name
        elif target_name:
            title = target_name
        else:
            title = "Experimental vs Calculated ΔΔG"

    # Call the master plotting function
    return _master_plot(
        x=x,
        y=y,
        title=title,
        xerr=xerr,
        yerr=yerr,
        method_name=method_name,
        target_name=target_name,
        quantity="$\\Delta \\Delta$ G",
        xlabel="Experimental",
        ylabel="Calculated",
        units="$\\mathrm{kcal\\,mol^{-1}}$",
        guidelines=True,
        origins=True,
        statistics=statistics if statistics is not None else ["RMSE", "MUE"],
        filename=filename,
        centralizing=True,
        shift=0.0,
        figsize=figsize,
        dpi="figure",
        data_labels=data_labels,
        axis_padding=0.5,
        xy_lim=[],
        font_sizes={"title": 12, "labels": 9, "other": 12},
        bootstrap_x_uncertainty=bootstrap_x_uncertainty,
        bootstrap_y_uncertainty=bootstrap_y_uncertainty,
        statistic_type=statistic_type,
        scatter_kwargs={"s": 10, "marker": "o"},
        nbootstrap=nbootstrap,
        **kwargs,
    )


def plot_dataset_DGs(
    dataset,
    experimental_column: str = "Exp. DeltaG",
    predicted_column: str = None,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: Optional[str] = None,
    centralizing: bool = True,
    shift: float = 0.0,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
    statistics: list = None,
    figsize: float = 5,
    nbootstrap: int = 1000,
    **kwargs,
):
    """Plot absolute free energies (ΔG) from a dataset with node model estimates.

    This function extracts absolute free energy data from a dataset and creates
    a scatter plot comparing experimental vs calculated values with statistics.

    Parameters
    ----------
    dataset : FEPDataset
        Dataset object containing node data with experimental and predicted values
    experimental_column : str, default = "Exp. DeltaG"
        Name of column containing experimental absolute free energies
    predicted_column : str, optional
        Name of column containing predicted absolute free energies.
        If None, will try to find common prediction column names like "MAP", "VI", etc.
    method_name : str, optional
        Name of method associated with results, e.g. 'perses'
    target_name : str, optional
        Name of system for results, e.g. 'Thrombin'
    title : str, default = ""
        Title for the plot
    filename : str, optional
        Filename to save plot
    centralizing : bool, default = True
        Whether to offset the free energies (Note: Node ΔG values are always mean-centered regardless of this parameter)
    shift : float, default = 0.0
        Additional shift to apply after mean-centering (both x and y axis)
    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        Type of statistic to use: 'mle' (maximum likelihood estimate) or 'mean' (bootstrap mean)
    statistics : list, optional
        List of statistics to compute and display in plot title. Default: ['RMSE', 'MUE']
        Available options: 'RMSE', 'MUE', 'R2', 'rho' (Pearson correlation),
        'KTAU' (Kendall's Tau), 'RAE' (Relative Absolute Error)
    figsize : float, default = 5
        Size of the figure (creates a square plot of figsize x figsize inches)
    nbootstrap : int, default = 1000
        Number of bootstrap samples to draw for computing statistics and confidence intervals
    **kwargs
        Additional arguments passed to _master_plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object

    Examples
    --------
    >>> # Basic usage with default statistics (RMSE, MUE)
    >>> fig = plot_dataset_DGs(dataset, predicted_column='MAP')
    >>>
    >>> # Show correlation statistics
    >>> fig = plot_dataset_DGs(
    ...     dataset,
    ...     predicted_column='VI',
    ...     statistics=['RMSE', 'R2', 'rho']
    ... )
    >>>
    >>> # Comprehensive statistics
    >>> fig = plot_dataset_DGs(
    ...     dataset,
    ...     predicted_column='WCC',
    ...     target_name='CDK8',
    ...     statistics=['RMSE', 'MUE', 'R2', 'rho', 'KTAU']
    ... )
    """
    # Get node data - only handle MAPLE dataset structure
    if not hasattr(dataset, "dataset_nodes"):
        raise ValueError("Dataset must have 'dataset_nodes' attribute")
    nodes_df = dataset.dataset_nodes

    # Extract experimental values
    if experimental_column not in nodes_df.columns:
        raise ValueError(
            f"Experimental column '{experimental_column}' not found in dataset nodes"
        )
    x = nodes_df[experimental_column].values

    # Find predicted column if not specified
    common_prediction_columns = ["MAP", "VI", "GMVI", "MLE", "WCC", "Pred. DeltaG"]
    if predicted_column is None:
        # Try common MAPLE prediction column names
        pred_columns_found = []
        for col in common_prediction_columns:
            if col in nodes_df.columns:
                pred_columns_found.append(col)
        print(f"Found predicted columns: {pred_columns_found}")
        # Randomly select a predicted column
        predicted_column = pred_columns_found[0]
        print(f"Using predicted column: {predicted_column}")

    elif predicted_column not in common_prediction_columns:
        raise ValueError(
            f"Predicted column '{predicted_column}' not found in common prediction columns"
            f"Common prediction columns must be one of: {common_prediction_columns}"
        )
    elif predicted_column in common_prediction_columns:
        print(f"Using predicted column: {predicted_column}")

    y = nodes_df[predicted_column].values

    # Filter out NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    # Handle error columns (optional)
    xerr = None
    yerr = None

    # Look for error columns
    if f"{experimental_column.replace('Exp. ', '')} Error" in nodes_df.columns:
        xerr = nodes_df[f"{experimental_column.replace('Exp. ', '')} Error"].values[
            valid_mask
        ]
    elif "Exp. Error" in nodes_df.columns:
        xerr = nodes_df["Exp. Error"].values[valid_mask]

    if f"{predicted_column}_uncertainty" in nodes_df.columns:
        yerr = nodes_df[f"{predicted_column}_uncertainty"].values[valid_mask]
    elif f"{predicted_column} Error" in nodes_df.columns:
        yerr = nodes_df[f"{predicted_column} Error"].values[valid_mask]

    # For ΔG plots, we don't use data labels (node indices are not meaningful for visualization)
    data_labels = []

    # Always centralize node values (ΔG) for meaningful comparison
    # Mean centering is essential for node value analysis
    x = x - np.mean(x) + shift
    y = y - np.mean(y) + shift

    # Create title if not provided
    if not title:
        if method_name and target_name:
            title = f"{method_name} - {target_name}"
        elif method_name:
            title = method_name
        elif target_name:
            title = target_name
        else:
            title = "Experimental vs Calculated ΔG"

    # Call the master plotting function
    return _master_plot(
        x=x,
        y=y,
        title=title,
        xerr=xerr,
        yerr=yerr,
        method_name=method_name,
        target_name=target_name,
        quantity="$\\Delta$ G",
        xlabel="Experimental",
        ylabel="Calculated",
        units="$\\mathrm{kcal\\,mol^{-1}}$",
        guidelines=True,
        origins=True,
        statistics=statistics if statistics is not None else ["RMSE", "MUE"],
        filename=filename,
        centralizing=False,  # Already centralized above if requested
        shift=0.0,
        figsize=figsize,
        dpi="figure",
        data_labels=data_labels,
        axis_padding=0.5,
        xy_lim=[],
        font_sizes={"title": 12, "labels": 9, "other": 12},
        bootstrap_x_uncertainty=bootstrap_x_uncertainty,
        bootstrap_y_uncertainty=bootstrap_y_uncertainty,
        statistic_type=statistic_type,
        scatter_kwargs={"s": 10, "marker": "o"},
        nbootstrap=nbootstrap,
        **kwargs,
    )


def plot_model_comparison_bars(
    dataset,
    performance_metrics: list = None,
    models: list = None,
    data_type: str = "nodes",
    model_colors: dict = None,
    filename: Optional[str] = None,
    stats_filename: Optional[str] = None,
    figsize: tuple = None,
    grid_layout: Tuple[int, int] = None,
    nrows: int = None,
    ncols: int = None,
    title: str = None,
    target_name: str = None,
    return_stats: bool = True,
    nbootstrap: int = 1000,
    metrics: list = None,  # Deprecated, use performance_metrics
    **kwargs,
) -> Tuple[Optional[plt.Figure], Optional[pd.DataFrame]]:
    """Create bar plots comparing performance metrics across multiple models.

    This function generates side-by-side bar plots for multiple performance metrics,
    with each model represented by a consistent color. Useful for comparing the
    performance of different prediction models.

    Parameters
    ----------
    dataset : FEPDataset
        Dataset object containing model predictions and metrics
    performance_metrics : list, optional
        List of performance metrics/statistics to evaluate and plot.
        Available options: 'RMSE', 'MUE', 'R2', 'rho' (Pearson), 'KTAU' (Kendall's tau), 'RAE'
        Default: ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']
    models : list, optional
        List of model column names to compare (e.g., ['MAP', 'GMVI', 'WCC']).
        This gives control over which models in the dataset should be evaluated.
        If None, automatically detects common model columns (MAP, VI, GMVI, MLE, WCC, etc.)
    data_type : str, default = "nodes"
        Type of data to plot: "nodes" for ΔG or "edges" for ΔΔG
    model_colors : dict, optional
        Dictionary mapping model names to colors. Default:
        {'MAP': 'blue', 'VI': 'green', 'GMVI': 'orange', 'MLE': 'purple', 'WCC': 'red', ...}
    filename : str, optional
        Filename to save plot. If None, displays in notebook
    stats_filename : str, optional
        Filename to save statistics DataFrame as CSV. If None, doesn't save
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, calculated based on grid layout
    grid_layout : tuple of (int, int), optional
        Grid arrangement as (nrows, ncols). For example:
        - (1, 2) for 2 metrics in a single row
        - (2, 2) for 4 metrics in a 2x2 grid
        - (4, 1) for 4 metrics in a single column
        Takes precedence over nrows/ncols if provided.
    nrows : int, optional
        Number of rows in subplot grid. If None, uses single row layout.
        Ignored if grid_layout is provided.
    ncols : int, optional
        Number of columns in subplot grid. If None, calculated based on number of metrics.
        Ignored if grid_layout is provided.
    title : str, optional
        Overall title for the figure. If None, generates automatically
    target_name : str, optional
        Name of the target/system (e.g., 'CDK8', 'Thrombin'). If provided,
        will be added to the auto-generated title
    return_stats : bool, default = True
        Whether to return the statistics DataFrame
    nbootstrap : int, default = 1000
        Number of bootstrap samples to draw for computing statistics and confidence intervals
    metrics : list, optional
        Deprecated. Use performance_metrics instead.
    **kwargs
        Additional arguments passed to matplotlib

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, pandas.DataFrame or None)
        - Figure object (None if saved to file)
        - DataFrame with statistics for each model/metric combination (None if return_stats=False)

    Examples
    --------
    >>> # Get both plot and statistics - default single row
    >>> fig, stats_df = plot_model_comparison_bars(dataset, data_type="nodes")
    >>> print(stats_df)
    >>>
    >>> # Select specific models and metrics with 2x1 layout
    >>> fig, stats_df = plot_model_comparison_bars(
    ...     dataset,
    ...     models=['MAP', 'GMVI', 'WCC'],
    ...     performance_metrics=['RMSE', 'R2'],
    ...     grid_layout=(2, 1),
    ...     data_type="nodes"
    ... )
    >>>
    >>> # 4 metrics in a 2x2 grid
    >>> fig, stats_df = plot_model_comparison_bars(
    ...     dataset,
    ...     models=['MAP', 'VI', 'GMVI', 'WCC'],
    ...     performance_metrics=['RMSE', 'MUE', 'R2', 'rho'],
    ...     grid_layout=(2, 2),
    ...     data_type="nodes"
    ... )
    >>>
    >>> # Custom 2x3 grid layout for 5 metrics (legacy nrows/ncols)
    >>> fig, stats_df = plot_model_comparison_bars(
    ...     dataset,
    ...     models=['MAP', 'VI', 'GMVI', 'WCC'],
    ...     performance_metrics=['RMSE', 'MUE', 'R2', 'rho', 'KTAU'],
    ...     nrows=2,
    ...     ncols=3,
    ...     stats_filename='model_comparison_stats.csv'
    ... )
    """
    # Handle deprecated 'metrics' parameter
    if metrics is not None and performance_metrics is None:
        import warnings
        warnings.warn(
            "The 'metrics' parameter is deprecated. Use 'performance_metrics' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        performance_metrics = metrics

    # Set up default model colors
    default_colors = {
        'MAP': 'blue',
        'VI': 'green',
        'GMVI': 'orange',
        'MLE': 'purple',
        'WCC': 'red',
        'Pred.DeltaG': 'violet',
        'DeltaDeltaG': 'violet',
    }
    if model_colors is None:
        model_colors = default_colors
    else:
        # Merge with defaults
        model_colors = {**default_colors, **model_colors}

    # Set default performance metrics
    available_metrics = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU', 'RAE']
    if performance_metrics is None:
        performance_metrics = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']
    else:
        # Validate provided metrics
        invalid_metrics = [m for m in performance_metrics if m not in available_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid performance metrics: {invalid_metrics}. "
                f"Available options: {available_metrics}"
            )

    # Determine data source
    if data_type == "nodes":
        data_df = dataset.dataset_nodes
        exp_column = 'Exp. DeltaG'
        default_models = ['MAP', 'VI', 'GMVI', 'MLE', 'WCC', 'Pred.DeltaG']
    elif data_type == "edges":
        data_df = dataset.dataset_edges
        exp_column = 'Experimental DeltaDeltaG'
        default_models = ['MAP', 'VI', 'GMVI', 'MLE', 'WCC', 'DeltaDeltaG']
    else:
        raise ValueError(f"data_type must be 'nodes' or 'edges', got {data_type}")

    # Auto-detect models if not specified
    if models is None:
        models = [m for m in default_models if m in data_df.columns]
        if not models:
            raise ValueError(f"No model columns found in dataset {data_type}")
    else:
        # Validate provided models exist in dataset
        missing_models = [m for m in models if m not in data_df.columns]
        if missing_models:
            available_in_dataset = [m for m in default_models if m in data_df.columns]
            raise ValueError(
                f"Models not found in dataset: {missing_models}. "
                f"Available models: {available_in_dataset}"
            )

    # Get experimental values
    if exp_column not in data_df.columns:
        raise ValueError(f"Experimental column '{exp_column}' not found in dataset {data_type}")
    y_true = data_df[exp_column].values

    # Mean-center node values if needed
    if data_type == "nodes":
        y_true = y_true - np.mean(y_true)

    # Compute metrics for each model
    all_metrics = {}
    for model_name in models:
        if model_name not in data_df.columns:
            continue

        y_pred = data_df[model_name].values
        if data_type == "nodes":
            y_pred = y_pred - np.mean(y_pred)

        # Filter out NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]

        # Get uncertainties if available
        dy_true = None
        dy_pred = None
        if f'{model_name}_uncertainty' in data_df.columns:
            dy_pred = data_df[f'{model_name}_uncertainty'].values[valid_mask]

        # Compute all metrics
        all_metrics[model_name] = {}
        for metric in performance_metrics:
            try:
                stats = bootstrap_statistic(
                    y_true_clean,
                    y_pred_clean,
                    dy_true,
                    dy_pred,
                    statistic=metric,
                    nbootstrap=nbootstrap,
                    include_true_uncertainty=False,
                    include_pred_uncertainty=False,
                )
                all_metrics[model_name][metric] = {
                    'mean': stats['mean'],
                    'mle': stats['mle'],
                    'ci_lower': stats['low'],
                    'ci_upper': stats['high'],
                }
            except Exception:
                all_metrics[model_name][metric] = None

    # Determine grid layout
    n_metrics = len(performance_metrics)

    # Handle grid_layout parameter (takes precedence over nrows/ncols)
    if grid_layout is not None:
        if not isinstance(grid_layout, (tuple, list)) or len(grid_layout) != 2:
            raise ValueError(
                f"grid_layout must be a tuple of (nrows, ncols), got {grid_layout}"
            )
        nrows, ncols = grid_layout

    # Calculate grid dimensions if not fully specified
    if nrows is None and ncols is None:
        # Default: single row layout
        nrows = 1
        ncols = n_metrics
    elif nrows is None:
        # ncols specified, calculate nrows
        nrows = int(np.ceil(n_metrics / ncols))
    elif ncols is None:
        # nrows specified, calculate ncols
        ncols = int(np.ceil(n_metrics / nrows))

    # Validate grid can fit all metrics
    if nrows * ncols < n_metrics:
        raise ValueError(
            f"Grid layout {nrows}x{ncols} ({nrows * ncols} subplots) is too small "
            f"for {n_metrics} metrics. Need at least {n_metrics} subplots."
        )

    # Set up figure size
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    # Create the plot
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes array for easy indexing
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, metric in enumerate(performance_metrics):
        ax = axes[idx]

        plot_models = []
        plot_means = []
        plot_errors = []
        plot_colors = []

        for model_name in models:
            if model_name in all_metrics and metric in all_metrics[model_name]:
                if all_metrics[model_name][metric] is not None:
                    plot_models.append(model_name)
                    plot_means.append(all_metrics[model_name][metric]['mean'])
                    # Use CI for error bars
                    ci_lower = all_metrics[model_name][metric]['ci_lower']
                    ci_upper = all_metrics[model_name][metric]['ci_upper']
                    mean_val = all_metrics[model_name][metric]['mean']
                    plot_errors.append([
                        mean_val - ci_lower,
                        ci_upper - mean_val
                    ])
                    plot_colors.append(model_colors.get(model_name, 'gray'))

        if plot_models:
            x_pos = np.arange(len(plot_models))
            bars = ax.bar(x_pos, plot_means, color=plot_colors, alpha=0.7, edgecolor='black')

            # Add error bars
            ax.errorbar(x_pos, plot_means, yerr=np.array(plot_errors).T,
                       fmt='none', ecolor='black', capsize=5, capthick=2)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_models, rotation=45, ha='right', fontsize=12)
            ax.set_ylabel(metric, fontsize=14)
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, mean in zip(bars, plot_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=12)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Set overall title
    if title is None:
        data_label = "Node (ΔG)" if data_type == "nodes" else "Edge (ΔΔG)"
        title = f'{data_label} Performance Metrics with 95% Confidence Intervals'
        if target_name:
            title = f'{target_name}: {title}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Create statistics DataFrame if requested
    stats_df = None
    if return_stats:
        stats_data = []
        for model_name in models:
            if model_name in all_metrics:
                for metric in performance_metrics:
                    if metric in all_metrics[model_name] and all_metrics[model_name][metric] is not None:
                        stats_data.append({
                            'Model': model_name,
                            'Metric': metric,
                            'Value': all_metrics[model_name][metric]['mle'],
                            'Mean': all_metrics[model_name][metric]['mean'],
                            'CI_Lower': all_metrics[model_name][metric]['ci_lower'],
                            'CI_Upper': all_metrics[model_name][metric]['ci_upper'],
                            'Data_Type': data_type
                        })

        stats_df = pd.DataFrame(stats_data)

        # Save to CSV if requested
        if stats_filename is not None:
            if not stats_filename.endswith('.csv'):
                stats_filename = stats_filename.rsplit('.', 1)[0] + '.csv'
            stats_df.to_csv(stats_filename, index=False)
            print(f"Statistics saved to '{stats_filename}'")

    # Save or display plot
    fig_result = None
    if filename is not None:
        if not filename.endswith('.pdf'):
            filename = filename.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        # Check if running in notebook
        if _is_notebook():
            # In Jupyter, return figure without plt.show() to avoid double display
            fig_result = fig
        else:
            # In scripts, need explicit show
            plt.show()
            fig_result = fig

    return fig_result, stats_df


def plot_model_comparison_correlation(
    dataset,
    models: list = None,
    data_type: str = "nodes",
    model_colors: dict = None,
    performance_metrics: list = None,
    filename: Optional[str] = None,
    stats_filename: Optional[str] = None,
    figsize: tuple = None,
    grid_layout: Tuple[int, int] = None,
    nrows: int = None,
    ncols: int = None,
    title: str = None,
    target_name: str = None,
    show_statistics: bool = True,
    return_stats: bool = True,
    nbootstrap: int = 1000,
    metrics: list = None,  # Deprecated, use performance_metrics
    **kwargs,
) -> Tuple[Optional[plt.Figure], Optional[pd.DataFrame]]:
    """Create correlation plots comparing multiple models side-by-side.

    This function generates a grid of scatter plots, one for each model, allowing
    direct visual comparison of model performance. Each model uses a consistent
    color scheme and displays bootstrap statistics.

    Parameters
    ----------
    dataset : FEPDataset
        Dataset object containing model predictions
    models : list, optional
        List of model column names to compare (e.g., ['MAP', 'GMVI', 'WCC']).
        This gives control over which models in the dataset should be evaluated.
        If None, automatically detects common model columns (MAP, VI, GMVI, MLE, WCC, etc.)
    data_type : str, default = "nodes"
        Type of data to plot: "nodes" for ΔG or "edges" for ΔΔG
    model_colors : dict, optional
        Dictionary mapping model names to colors. Default:
        {'MAP': 'blue', 'VI': 'green', 'GMVI': 'orange', 'MLE': 'purple', 'WCC': 'red', ...}
    performance_metrics : list, optional
        List of performance metrics/statistics to display in statistics box.
        Available options: 'RMSE', 'MUE', 'R2', 'rho' (Pearson), 'KTAU' (Kendall's tau), 'RAE'
        Default: ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']
    filename : str, optional
        Filename to save plot. If None, displays in notebook
    stats_filename : str, optional
        Filename to save statistics DataFrame as CSV. If None, doesn't save
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, calculated based on grid layout
    grid_layout : tuple of (int, int), optional
        Grid arrangement as (nrows, ncols). For example:
        - (1, 2) for 2 models in a single row
        - (2, 2) for 4 models in a 2x2 grid
        - (4, 1) for 4 models in a single column
        Takes precedence over nrows/ncols if provided.
    nrows : int, optional
        Number of rows in subplot grid. If None, uses single row layout.
        Ignored if grid_layout is provided.
    ncols : int, optional
        Number of columns in subplot grid. If None, calculated based on number of models.
        Ignored if grid_layout is provided.
    title : str, optional
        Overall title for the figure. If None, generates automatically
    target_name : str, optional
        Name of the target/system being analyzed. If provided, prepended to the title
    show_statistics : bool, default = True
        Whether to display statistics text box on each subplot
    return_stats : bool, default = True
        Whether to return the statistics DataFrame
    nbootstrap : int, default = 1000
        Number of bootstrap samples to draw for computing statistics and confidence intervals
    metrics : list, optional
        Deprecated. Use performance_metrics instead.
    **kwargs
        Additional arguments passed to matplotlib

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, pandas.DataFrame or None)
        - Figure object (None if saved to file)
        - DataFrame with statistics for each model/metric combination (None if return_stats=False)

    Examples
    --------
    >>> # Get both plot and statistics - default single row
    >>> fig, stats_df = plot_model_comparison_correlation(dataset, data_type="nodes")
    >>>
    >>> # Select specific models to compare with 2x1 layout
    >>> fig, stats_df = plot_model_comparison_correlation(
    ...     dataset,
    ...     models=['MAP', 'GMVI'],
    ...     grid_layout=(2, 1),
    ...     data_type="nodes"
    ... )
    >>>
    >>> # Custom 2x2 grid layout for 4 models with specific metrics
    >>> fig, stats_df = plot_model_comparison_correlation(
    ...     dataset,
    ...     models=['MAP', 'VI', 'GMVI', 'WCC'],
    ...     performance_metrics=['RMSE', 'R2', 'rho'],
    ...     grid_layout=(2, 2),
    ...     stats_filename='correlation_stats.csv'
    ... )
    >>>
    >>> # Single column layout for 3 models
    >>> fig, stats_df = plot_model_comparison_correlation(
    ...     dataset,
    ...     models=['MAP', 'GMVI', 'WCC'],
    ...     grid_layout=(3, 1),
    ...     data_type="edges"
    ... )
    """
    # Handle deprecated 'metrics' parameter
    if metrics is not None and performance_metrics is None:
        import warnings
        warnings.warn(
            "The 'metrics' parameter is deprecated. Use 'performance_metrics' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        performance_metrics = metrics

    # Set up default model colors
    default_colors = {
        'MAP': 'blue',
        'VI': 'green',
        'GMVI': 'orange',
        'MLE': 'purple',
        'WCC': 'red',
        'Pred.DeltaG': 'violet',
        'DeltaDeltaG': 'violet',
    }
    if model_colors is None:
        model_colors = default_colors
    else:
        model_colors = {**default_colors, **model_colors}

    # Set default performance metrics for statistics display
    available_metrics = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU', 'RAE']
    if performance_metrics is None:
        performance_metrics = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']
    else:
        # Validate provided metrics
        invalid_metrics = [m for m in performance_metrics if m not in available_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid performance metrics: {invalid_metrics}. "
                f"Available options: {available_metrics}"
            )

    # Determine data source
    if data_type == "nodes":
        data_df = dataset.dataset_nodes
        exp_column = 'Exp. DeltaG'
        default_models = ['MAP', 'VI', 'GMVI', 'MLE', 'WCC']
        quantity_label = "$\\Delta$ G"
    elif data_type == "edges":
        data_df = dataset.dataset_edges
        exp_column = 'Experimental DeltaDeltaG'
        default_models = ['MAP', 'VI', 'GMVI', 'MLE', 'WCC', 'DeltaDeltaG']
        quantity_label = "$\\Delta \\Delta$ G"
    else:
        raise ValueError(f"data_type must be 'nodes' or 'edges', got {data_type}")

    # Auto-detect models if not specified
    if models is None:
        models = [m for m in default_models if m in data_df.columns]
        if not models:
            raise ValueError(f"No model columns found in dataset {data_type}")
    else:
        # Validate provided models exist in dataset
        missing_models = [m for m in models if m not in data_df.columns]
        if missing_models:
            available_in_dataset = [m for m in default_models if m in data_df.columns]
            raise ValueError(
                f"Models not found in dataset: {missing_models}. "
                f"Available models: {available_in_dataset}"
            )

    # Get experimental values
    if exp_column not in data_df.columns:
        raise ValueError(f"Experimental column '{exp_column}' not found in dataset {data_type}")
    exp_values = data_df[exp_column].values

    # Mean-center node values if needed
    if data_type == "nodes":
        exp_mean = np.mean(exp_values)
        exp_values = exp_values - exp_mean

    # Determine grid layout
    n_models = len(models)

    # Handle grid_layout parameter (takes precedence over nrows/ncols)
    if grid_layout is not None:
        if not isinstance(grid_layout, (tuple, list)) or len(grid_layout) != 2:
            raise ValueError(
                f"grid_layout must be a tuple of (nrows, ncols), got {grid_layout}"
            )
        nrows, ncols = grid_layout

    # Calculate grid dimensions if not fully specified
    if nrows is None and ncols is None:
        # Default: single row layout
        nrows = 1
        ncols = n_models
    elif nrows is None:
        # ncols specified, calculate nrows
        nrows = int(np.ceil(n_models / ncols))
    elif ncols is None:
        # nrows specified, calculate ncols
        ncols = int(np.ceil(n_models / nrows))

    # Validate grid can fit all models
    if nrows * ncols < n_models:
        raise ValueError(
            f"Grid layout {nrows}x{ncols} ({nrows * ncols} subplots) is too small "
            f"for {n_models} models. Need at least {n_models} subplots."
        )

    # Set up figure size
    if figsize is None:
        figsize = (6 * ncols, 6 * nrows)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes array for easy indexing
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Dictionary to store statistics for DataFrame
    all_stats = {} if return_stats else None

    # Plot each model
    for idx, model_name in enumerate(models):
        if model_name not in data_df.columns:
            continue

        ax = axes[idx]
        color = model_colors.get(model_name, 'gray')

        # Get predicted values
        pred = data_df[model_name].values

        # Mean-center node predictions
        if data_type == "nodes":
            pred = pred - np.mean(pred)

        # Filter out NaN values
        valid_mask = ~(np.isnan(exp_values) | np.isnan(pred))
        exp_clean = exp_values[valid_mask]
        pred_clean = pred[valid_mask]

        # Check for uncertainties
        pred_errors = None
        if f'{model_name}_uncertainty' in data_df.columns:
            pred_errors = data_df[f'{model_name}_uncertainty'].values[valid_mask]

        # Create scatter plot
        if pred_errors is not None:
            ax.errorbar(exp_clean, pred_clean, yerr=pred_errors, fmt='o',
                       alpha=0.6, color=color, markersize=6, elinewidth=1,
                       capsize=3, zorder=5)
        else:
            ax.scatter(exp_clean, pred_clean, alpha=0.6, color=color,
                      s=50, zorder=5)

        # Perfect correlation line
        min_val = min(exp_clean.min(), pred_clean.min()) - 1
        max_val = max(exp_clean.max(), pred_clean.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
               label='Perfect correlation', zorder=4, linewidth=2)

        # Add error bands
        x_band = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_band, x_band - 1, x_band + 1, alpha=0.15,
                       color='gray', label='±1 kcal/mol', zorder=1)
        ax.fill_between(x_band, x_band - 0.5, x_band + 0.5, alpha=0.25,
                       color='gray', label='±0.5 kcal/mol', zorder=2)

        # Set axis properties
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel(f'Experimental {quantity_label} (kcal/mol)', fontsize=14)
        ax.set_ylabel(f'Predicted {quantity_label} (kcal/mol)', fontsize=14)
        ax.set_title(f'{model_name} Model', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Add legend (only first subplot to avoid clutter)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=12)

        # Compute and display statistics
        metrics_text = ""
        if return_stats:
            all_stats[model_name] = {}

        for metric in performance_metrics:
            try:
                stats = bootstrap_statistic(
                    exp_clean,
                    pred_clean,
                    None,
                    pred_errors if pred_errors is not None else None,
                    statistic=metric,
                    nbootstrap=nbootstrap,
                    include_true_uncertainty=False,
                    include_pred_uncertainty=False,
                )
                mle_val = stats['mle']
                ci_low = stats['low']
                ci_high = stats['high']
                mean_val = stats['mean']

                # Store for DataFrame
                if return_stats:
                    all_stats[model_name][metric] = {
                        'mle': mle_val,
                        'mean': mean_val,
                        'ci_lower': ci_low,
                        'ci_upper': ci_high
                    }

                # Add to display text
                if show_statistics:
                    metrics_text += f"{metric}: {mle_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]\n"
            except Exception:
                pass

        if show_statistics and metrics_text:
            ax.text(0.95, 0.05, metrics_text.strip(),
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   fontsize=12, zorder=10, family='monospace')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Set overall title
    if title is None:
        data_label = "Node (ΔG)" if data_type == "nodes" else "Edge (ΔΔG)"
        center_info = " (Mean-Centered)" if data_type == "nodes" else ""
        title = f'{data_label} Predictions: Model Comparison with Bootstrap Statistics{center_info}'
        if target_name:
            title = f'{target_name}: {title}'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    # Create statistics DataFrame if requested
    stats_df = None
    if return_stats and all_stats:
        stats_data = []
        for model_name in models:
            if model_name in all_stats:
                for metric in performance_metrics:
                    if metric in all_stats[model_name]:
                        stats_data.append({
                            'Model': model_name,
                            'Metric': metric,
                            'Value': all_stats[model_name][metric]['mle'],
                            'Mean': all_stats[model_name][metric]['mean'],
                            'CI_Lower': all_stats[model_name][metric]['ci_lower'],
                            'CI_Upper': all_stats[model_name][metric]['ci_upper'],
                            'Data_Type': data_type
                        })

        stats_df = pd.DataFrame(stats_data)

        # Save to CSV if requested
        if stats_filename is not None:
            if not stats_filename.endswith('.csv'):
                stats_filename = stats_filename.rsplit('.', 1)[0] + '.csv'
            stats_df.to_csv(stats_filename, index=False)
            print(f"Statistics saved to '{stats_filename}'")

    # Save or display plot
    fig_result = None
    if filename is not None:
        if not filename.endswith('.pdf'):
            filename = filename.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        # Check if running in notebook
        if _is_notebook():
            # In Jupyter, return figure without plt.show() to avoid double display
            fig_result = fig
        else:
            # In scripts, need explicit show
            plt.show()
            fig_result = fig

    return fig_result, stats_df


def plot_error_distribution(
    dataset,
    predicted_column: str = None,
    experimental_column: str = None,
    data_type: str = "edges",
    fit_distribution: str = "normal",
    filename: Optional[str] = None,
    stats_filename: Optional[str] = None,
    figsize: tuple = (12, 8),
    title: str = None,
    bins: int = 20,
    show_statistics: bool = True,
    return_stats: bool = True,
    **kwargs,
) -> Tuple[Optional[plt.Figure], Optional[pd.DataFrame]]:
    """Plot error distribution between predicted and experimental values with KDE.

    This function creates a kernel density estimate (KDE) plot of the errors
    between predicted and experimental values, overlaid with a fitted distribution.
    Useful for assessing the error characteristics of FEP predictions.

    Parameters
    ----------
    dataset : FEPDataset
        Dataset object containing predictions and experimental values
    predicted_column : str, optional
        Name of column containing predicted values. If None, auto-detects
        common prediction columns (MAP, VI, GMVI, MLE, WCC, etc.)
    experimental_column : str, optional
        Name of column containing experimental values. If None, uses default
        experimental column for the data type
    data_type : str, default = "edges"
        Type of data to analyze: "edges" for ΔΔG or "nodes" for ΔG
    fit_distribution : str, default = "normal"
        Distribution to fit to errors. Options: "normal", "none"
    filename : str, optional
        Filename to save plot. If None, displays in notebook
    stats_filename : str, optional
        Filename to save error statistics DataFrame as CSV. If None, doesn't save
    figsize : tuple, default = (12, 8)
        Figure size (width, height) in inches
    title : str, optional
        Title for the plot. If None, generates automatically
    bins : int, default = 20
        Number of bins for histogram overlay
    show_statistics : bool, default = True
        Whether to display statistics box on plot
    return_stats : bool, default = True
        Whether to return the error statistics DataFrame
    **kwargs
        Additional arguments passed to matplotlib

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, pandas.DataFrame or None)
        - Figure object (None if saved to file)
        - DataFrame with error statistics (None if return_stats=False)

    Examples
    --------
    >>> # Get plot and error statistics
    >>> fig, stats_df = plot_error_distribution(dataset, predicted_column='MAP')
    >>> print(stats_df)
    >>>
    >>> # Save statistics to CSV
    >>> fig, stats_df = plot_error_distribution(
    ...     dataset,
    ...     predicted_column='WCC',
    ...     stats_filename='wcc_error_stats.csv'
    ... )
    """
    # Determine data source and default columns
    if data_type == "edges":
        data_df = dataset.dataset_edges
        default_exp_column = 'Experimental DeltaDeltaG'
        default_pred_columns = ['MAP', 'VI', 'GMVI', 'MLE', 'WCC', 'DeltaDeltaG']
        quantity_label = "$\\Delta \\Delta$ G"
    elif data_type == "nodes":
        data_df = dataset.dataset_nodes
        default_exp_column = 'Exp. DeltaG'
        default_pred_columns = ['MAP', 'VI', 'GMVI', 'MLE', 'WCC', 'Pred. DeltaG']
        quantity_label = "$\\Delta$ G"
    else:
        raise ValueError(f"data_type must be 'nodes' or 'edges', got {data_type}")

    # Set experimental column
    if experimental_column is None:
        experimental_column = default_exp_column

    if experimental_column not in data_df.columns:
        raise ValueError(
            f"Experimental column '{experimental_column}' not found in dataset {data_type}"
        )

    # Auto-detect predicted column if not specified
    if predicted_column is None:
        pred_columns_found = [col for col in default_pred_columns if col in data_df.columns]
        if not pred_columns_found:
            raise ValueError(f"No predicted columns found in dataset {data_type}")
        predicted_column = pred_columns_found[0]
        print(f"Using predicted column: {predicted_column}")

    if predicted_column not in data_df.columns:
        raise ValueError(
            f"Predicted column '{predicted_column}' not found in dataset {data_type}"
        )

    # Get values and calculate errors
    exp_values = data_df[experimental_column].values
    pred_values = data_df[predicted_column].values

    # Filter out NaN values
    valid_mask = ~(np.isnan(exp_values) | np.isnan(pred_values))
    exp_values = exp_values[valid_mask]
    pred_values = pred_values[valid_mask]

    # Calculate errors (predicted - experimental)
    errors = pred_values - exp_values

    print(f"\n{'='*60}")
    print(f"Error Distribution Analysis: {predicted_column}")
    print(f"{'='*60}")
    print(f"Number of points: {len(errors)}")
    print(f"Error range: [{errors.min():.4f}, {errors.max():.4f}] kcal/mol")

    # Fit distribution if requested
    fit_params = None
    if fit_distribution == "normal":
        mu, sigma = stats.norm.fit(errors)
        fit_params = {'mu': mu, 'sigma': sigma}
        print(f"\nFitted Normal Distribution:")
        print(f"  Mean (μ): {mu:.4f} kcal/mol")
        print(f"  Std (σ): {sigma:.4f} kcal/mol")
        print(f"  Variance (σ²): {sigma**2:.4f} kcal²/mol²")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # KDE plot
    sns.kdeplot(errors, ax=ax, color='steelblue', fill=True, alpha=0.3,
               linewidth=3, label='KDE of Errors')

    # Overlay fitted distribution
    if fit_distribution == "normal" and fit_params:
        x = np.linspace(errors.min() - 1, errors.max() + 1, 200)
        fitted_normal = stats.norm.pdf(x, fit_params['mu'], fit_params['sigma'])
        ax.plot(x, fitted_normal, 'r--', linewidth=3,
               label=f'Fitted N({fit_params["mu"]:.3f}, {fit_params["sigma"]:.3f}²)')

    # Add histogram for reference
    ax.hist(errors, bins=bins, density=True, alpha=0.2, color='steelblue',
           edgecolor='black', label='Histogram')

    # Add vertical lines
    if fit_params:
        ax.axvline(fit_params['mu'], color='red', linestyle=':', linewidth=2.5,
                  label=f'Mean = {fit_params["mu"]:.3f}')
    else:
        ax.axvline(np.mean(errors), color='red', linestyle=':', linewidth=2.5,
                  label=f'Mean = {np.mean(errors):.3f}')

    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero error')

    # Labels and title
    ax.set_xlabel(f'Error ({predicted_column} - Experimental) [kcal/mol]', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)

    if title is None:
        title = f'Error Distribution: {predicted_column} vs Experimental {quantity_label}\n'
        if fit_distribution == "normal":
            title += 'with Fitted Normal Distribution'
    ax.set_title(title, fontsize=18, fontweight='bold')

    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)

    # Add statistics box
    if show_statistics:
        stats_text = f"Statistics:\n"
        stats_text += f"N = {len(errors)}\n"
        if fit_params:
            stats_text += f"μ = {fit_params['mu']:.4f} kcal/mol\n"
            stats_text += f"σ = {fit_params['sigma']:.4f} kcal/mol\n"
            stats_text += f"σ² = {fit_params['sigma']**2:.4f} kcal²/mol²\n"
        else:
            stats_text += f"Mean = {np.mean(errors):.4f} kcal/mol\n"
            stats_text += f"Std = {np.std(errors, ddof=1):.4f} kcal/mol\n"
        stats_text += f"Min = {errors.min():.4f} kcal/mol\n"
        stats_text += f"Max = {errors.max():.4f} kcal/mol\n"
        stats_text += f"Median = {np.median(errors):.4f} kcal/mol\n"
        stats_text += f"MAE = {np.abs(errors).mean():.4f} kcal/mol\n"
        stats_text += f"RMSE = {np.sqrt(np.mean(errors**2)):.4f} kcal/mol"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                        edgecolor='black', linewidth=1.5),
               fontsize=13, family='monospace')

    plt.tight_layout()

    # Print summary first
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    if fit_params:
        print(f"Mean error: {fit_params['mu']:.4f} kcal/mol")
        print(f"Std error: {fit_params['sigma']:.4f} kcal/mol")
    else:
        print(f"Mean error: {np.mean(errors):.4f} kcal/mol")
        print(f"Std error: {np.std(errors, ddof=1):.4f} kcal/mol")
    print(f"Median error: {np.median(errors):.4f} kcal/mol")
    print(f"MAE: {np.abs(errors).mean():.4f} kcal/mol")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f} kcal/mol")
    print(f"{'='*60}\n")

    # Create statistics DataFrame if requested
    stats_df = None
    if return_stats:
        stats_dict = {
            'Model': predicted_column,
            'Data_Type': data_type,
            'N': len(errors),
            'Mean_Error': fit_params['mu'] if fit_params else np.mean(errors),
            'Std_Error': fit_params['sigma'] if fit_params else np.std(errors, ddof=1),
            'Variance': fit_params['sigma']**2 if fit_params else np.var(errors, ddof=1),
            'Min_Error': errors.min(),
            'Max_Error': errors.max(),
            'Median_Error': np.median(errors),
            'MAE': np.abs(errors).mean(),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'Distribution_Fit': fit_distribution
        }
        stats_df = pd.DataFrame([stats_dict])

        # Save to CSV if requested
        if stats_filename is not None:
            if not stats_filename.endswith('.csv'):
                stats_filename = stats_filename.rsplit('.', 1)[0] + '.csv'
            stats_df.to_csv(stats_filename, index=False)
            print(f"✅ Error statistics saved to '{stats_filename}'")

    # Save or display plot
    fig_result = None
    if filename is not None:
        if not filename.endswith('.pdf'):
            filename = filename.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✅ Error distribution plot saved as '{filename}'")
        plt.close(fig)
    else:
        # Check if running in notebook
        if _is_notebook():
            # In Jupyter, just return figure - it will auto-display once
            # Don't call plt.show() to avoid double display
            fig_result = fig
        else:
            # In scripts, need explicit show
            plt.show()
            fig_result = fig

    return fig_result, stats_df


def plot_dataset_all_DDGs(
    dataset,
    experimental_column: str = "Experimental DeltaDeltaG",
    predicted_column: str = None,
    method_name: str = "",
    target_name: str = "",
    title: str = "",
    filename: Optional[str] = None,
    shift: float = 0.0,
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    statistic_type: str = "mle",
    **kwargs,
):
    """Plot all possible relative free energies calculated from absolute free energies in a dataset.

    This function computes all pairwise differences between absolute free energies
    from a dataset and creates a scatter plot comparing experimental vs calculated values.

    Parameters
    ----------
    dataset : BaseDataset
        Dataset object containing node data with experimental and predicted values
    experimental_column : str, default = "Exp. DeltaG"
        Name of column containing experimental absolute free energies
    predicted_column : str, optional
        Name of column containing predicted absolute free energies.
        If None, will try to find "Estimated DeltaG" column
    method_name : str, optional
        Name of method associated with results, e.g. 'perses'
    target_name : str, optional
        Name of system for results, e.g. 'Thrombin'
    title : str, default = ""
        Title for the plot
    filename : str, optional
        Filename to save plot
    shift : float, default = 0.0
        Shift both x and y axis by a constant
    bootstrap_x_uncertainty : bool, default False
        Whether to account for uncertainty in x when bootstrapping
    bootstrap_y_uncertainty : bool, default False
        Whether to account for uncertainty in y when bootstrapping
    statistic_type : str, default 'mle'
        Type of statistic to use: 'mle' or 'mean'
    **kwargs
        Additional arguments passed to _master_plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Get node data
    nodes_df = dataset.dataset_nodes

    # Extract experimental values
    if experimental_column not in nodes_df.columns:
        raise ValueError(
            f"Experimental column '{experimental_column}' not found in dataset nodes"
        )
    x_abs = nodes_df[experimental_column].values

    # Find predicted column if not specified
    if predicted_column is None:
        # Try standardized column name first
        if "Estimated DeltaG" in nodes_df.columns:
            predicted_column = "Estimated DeltaG"
        else:
            # Fallback to old naming convention
            predicted_columns = [
                col for col in nodes_df.columns if col.startswith("Predicted_Node_")
            ]
            if len(predicted_columns) == 0:
                raise ValueError(
                    "No predicted node columns found. Please specify predicted_column parameter."
                )
            elif len(predicted_columns) == 1:
                predicted_column = predicted_columns[0]
            else:
                raise ValueError(
                    f"Multiple predicted node columns found: {predicted_columns}. Please specify predicted_column parameter."
                )
    elif predicted_column not in nodes_df.columns:
        raise ValueError(
            f"Predicted column '{predicted_column}' not found in dataset nodes"
        )

    y_abs = nodes_df[predicted_column].values

    # Filter out NaN values
    valid_mask = ~(np.isnan(x_abs) | np.isnan(y_abs))
    x_abs = x_abs[valid_mask]
    y_abs = y_abs[valid_mask]

    # Handle error columns (optional)
    xabserr = None
    yabserr = None

    # Look for error columns
    xerr_column = (
        f"{experimental_column}_error"
        if f"{experimental_column}_error" in nodes_df.columns
        else None
    )
    yerr_column = (
        f"{predicted_column}_error"
        if f"{predicted_column}_error" in nodes_df.columns
        else None
    )

    if xerr_column:
        xabserr = nodes_df[xerr_column].values[valid_mask]
    if yerr_column:
        yabserr = nodes_df[yerr_column].values[valid_mask]

    # Use default errors if not provided
    if xabserr is None:
        xabserr = np.repeat(0.1, len(x_abs))
    if yabserr is None:
        yabserr = np.repeat(0.1, len(y_abs))

    # Compute all pairwise differences
    x_data = []
    y_data = []
    xerr = []
    yerr = []
    for a, b in itertools.combinations(range(len(x_abs)), 2):
        x = x_abs[a] - x_abs[b]
        x_data.append(x)
        x_data.append(-x)
        err = (xabserr[a] ** 2 + xabserr[b] ** 2) ** 0.5
        xerr.append(err)
        xerr.append(err)
        y = y_abs[a] - y_abs[b]
        y_data.append(y)
        y_data.append(-y)
        err = (yabserr[a] ** 2 + yabserr[b] ** 2) ** 0.5
        yerr.append(err)
        yerr.append(err)

    x_data_ = np.array(x_data)
    y_data_ = np.array(y_data)
    xerr_ = np.array(xerr)
    yerr_ = np.array(yerr)

    # Create the plot
    return _master_plot(
        x_data_,
        y_data_,
        xerr=xerr_,
        yerr=yerr_,
        title=title,
        method_name=method_name,
        filename=filename,
        target_name=target_name,
        shift=shift,
        bootstrap_x_uncertainty=bootstrap_x_uncertainty,
        bootstrap_y_uncertainty=bootstrap_y_uncertainty,
        statistic_type=statistic_type,
        **kwargs,
    )
