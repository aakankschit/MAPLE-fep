# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "maple-fep",
#     "marimo",
#     "matplotlib",
#     "pandas",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Getting Started with MAPLE-FEP

        **MAPLE (Maximum A Posteriori Learning of Energies)** is a Python package for analyzing Free Energy Perturbation (FEP) data using probabilistic node models and Bayesian inference.

        This tutorial will walk you through:

        1. Loading a benchmark FEP dataset
        2. Understanding the data structure
        3. Training a Bayesian inference model (NodeModel)
        4. Analyzing the results and comparing to experimental values
        5. Visualizing model performance

        Let's get started!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Setting Up the Environment

        First, let's import the necessary modules from MAPLE and set up our environment.
        """
    )
    return


@app.cell
def _():
    import os

    # Handle OpenMP issue on some systems
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyro

    # MAPLE imports
    from maple.dataset import FEPDataset, FEPBenchmarkDataset
    from maple.models import NodeModel, NodeModelConfig
    from maple.models import PriorType, GuideType
    from maple.graph_analysis import calculate_rmse, calculate_mae
    from maple.graph_analysis import plot_dataset_DGs, plot_dataset_DDGs

    # Clear any previous Pyro state
    pyro.clear_param_store()

    print("All imports successful!")
    return (
        FEPBenchmarkDataset,
        FEPDataset,
        GuideType,
        NodeModel,
        NodeModelConfig,
        PriorType,
        calculate_mae,
        calculate_rmse,
        np,
        os,
        pd,
        plt,
        plot_dataset_DDGs,
        plot_dataset_DGs,
        pyro,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Loading Benchmark Data

        MAPLE provides easy access to several FEP benchmark datasets from the Schindler et al. repository.

        Available datasets include:
        - **cdk8** - Cyclin-dependent kinase 8
        - **cmet** - c-MET kinase
        - **eg5** - Kinesin Eg5
        - **hif2a** - Hypoxia-inducible factor 2α
        - **pfkfb3** - 6-phosphofructo-2-kinase
        - **shp2** - SHP2 phosphatase
        - **syk** - Spleen tyrosine kinase
        - **tnks2** - Tankyrase 2

        Let's load the **CDK8** dataset with 5ns sampling time:
        """
    )
    return


@app.cell
def _(FEPDataset):
    # Create an FEPDataset from the benchmark collection
    # Using CDK8 (Cyclin-dependent kinase 8) with 5ns sampling time
    _dataset_name = "cdk8"
    _sampling_time = "5ns"

    dataset = FEPDataset(dataset_name=_dataset_name, sampling_time=_sampling_time)

    print(f"Dataset: {_dataset_name}")
    print(f"Sampling time: {_sampling_time}")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Exploring the Data Structure

        The FEPDataset contains two main DataFrames:

        - **dataset_nodes**: Information about each ligand (node in the FEP graph)
        - **dataset_edges**: Information about each perturbation (edge in the FEP graph)

        Let's explore both:
        """
    )
    return


@app.cell
def _(dataset, mo):
    # Get the node data
    nodes_df = dataset.dataset_nodes.copy()
    nodes_df_display = nodes_df.head(10)

    mo.md(f"""
    ### Node Data (Ligands)

    The node DataFrame contains {len(nodes_df)} ligands with experimental binding free energies:

    {mo.as_html(nodes_df_display)}
    """)
    return nodes_df, nodes_df_display


@app.cell
def _(dataset, mo):
    # Get the edge data
    edges_df = dataset.dataset_edges.copy()
    edges_df_display = edges_df.head(10)

    mo.md(f"""
    ### Edge Data (Perturbations)

    The edge DataFrame contains {len(edges_df)} FEP perturbations connecting ligand pairs:

    {mo.as_html(edges_df_display)}

    Key columns:
    - **Source/Destination**: Ligand indices being compared
    - **DeltaDeltaG**: Computed relative binding free energy difference (ΔΔG)
    - **DeltaDeltaG Error**: Uncertainty in the computed value
    - **CCC**: Cycle closure corrected value
    """)
    return edges_df, edges_df_display


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Understanding the Graph Structure

        FEP calculations form a network where:
        - **Nodes** represent ligands
        - **Edges** represent perturbations with computed ΔΔG values

        MAPLE converts this to a graph format for inference:
        """
    )
    return


@app.cell
def _(dataset, mo):
    # Get the graph data structure
    graph_data = dataset.get_graph_data()

    mo.md(f"""
    ### Graph Statistics

    | Property | Value |
    |----------|-------|
    | Number of nodes (ligands) | {graph_data['N']} |
    | Number of edges (perturbations) | {graph_data['M']} |
    | Average connectivity | {2 * graph_data['M'] / graph_data['N']:.2f} edges per node |
    """)
    return (graph_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Training a NodeModel

        The **NodeModel** is MAPLE's primary inference engine. It uses Bayesian inference to estimate absolute binding free energies (ΔG) from relative measurements (ΔΔG).

        We'll configure the model with:
        - **MAP inference** (Maximum A Posteriori) using `GuideType.AUTO_DELTA`
        - **Normal prior** on node values
        - Reasonable hyperparameters for FEP data
        """
    )
    return


@app.cell
def _(GuideType, NodeModel, NodeModelConfig, PriorType, dataset, pyro):
    # Clear previous Pyro state
    pyro.clear_param_store()

    # Configure the model
    config = NodeModelConfig(
        learning_rate=0.01,
        num_steps=2000,
        prior_type=PriorType.NORMAL,
        prior_scale=10.0,  # Wide prior (in kcal/mol)
        guide_type=GuideType.AUTO_DELTA,  # MAP inference
        verbose=True,
    )

    # Create and train the model
    model = NodeModel(config=config, dataset=dataset)
    print("Training NodeModel...")
    model.train()
    print("Training complete!")
    return config, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Analyzing Model Results

        After training, we can:
        1. Extract the predicted node values (absolute ΔG estimates)
        2. Add predictions to the dataset
        3. Compare with experimental values
        """
    )
    return


@app.cell
def _(dataset, model, mo, np):
    # Add predictions to the dataset
    model.add_predictions_to_dataset()

    # Get the updated node data
    results_df = dataset.dataset_nodes.copy()

    # The predictions are relative - we need to align with experimental values
    # by matching the mean (gauge freedom in FEP)
    if "MAP" in results_df.columns and "Exp. DeltaG" in results_df.columns:
        exp_mean = results_df["Exp. DeltaG"].mean()
        pred_mean = results_df["MAP"].mean()
        results_df["MAP_aligned"] = results_df["MAP"] - pred_mean + exp_mean

    mo.md(f"""
    ### Predictions Added to Dataset

    The model predictions have been added as the 'MAP' column:

    {mo.as_html(results_df[['Name', 'Exp. DeltaG', 'MAP']].head(10).round(3))}

    **Note**: FEP values have gauge freedom - absolute values are arbitrary but relative differences are meaningful. We align predictions with experimental mean for comparison.
    """)
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7. Performance Metrics

        Let's calculate key performance metrics comparing our predictions to experimental values:

        - **RMSE**: Root Mean Square Error
        - **MAE**: Mean Absolute Error
        - **Correlation**: Pearson and Spearman coefficients
        """
    )
    return


@app.cell
def _(calculate_mae, calculate_rmse, mo, np, results_df):
    from scipy import stats

    # Get aligned predictions and experimental values
    if "MAP_aligned" in results_df.columns:
        y_pred = results_df["MAP_aligned"].values
        y_true = results_df["Exp. DeltaG"].values

        # Calculate metrics
        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)

        # Correlation coefficients
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)

        mo.md(f"""
        ### Performance Summary

        | Metric | Value |
        |--------|-------|
        | RMSE | {rmse:.3f} kcal/mol |
        | MAE | {mae:.3f} kcal/mol |
        | Pearson r | {pearson_r:.3f} (p={pearson_p:.2e}) |
        | Spearman ρ | {spearman_r:.3f} (p={spearman_p:.2e}) |

        **Interpretation**:
        - RMSE < 1.0 kcal/mol is typically considered good for FEP
        - High correlation (> 0.7) indicates good ranking ability
        """)
    else:
        mo.md("**Note**: Could not calculate metrics - check data alignment.")
    return (
        mae,
        pearson_p,
        pearson_r,
        rmse,
        spearman_p,
        spearman_r,
        stats,
        y_pred,
        y_true,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8. Visualization with MAPLE's Built-in Plotting

        MAPLE provides specialized plotting functions for FEP analysis. Let's use `plot_dataset_DGs` to visualize our results with bootstrap confidence intervals:
        """
    )
    return


@app.cell
def _(dataset, plot_dataset_DGs):
    # Use MAPLE's built-in plotting function for node (ΔG) data
    # This automatically handles mean-centering and computes bootstrap statistics
    fig_dg = plot_dataset_DGs(
        dataset,
        predicted_column="MAP",
        target_name="CDK8",
        title="MAP Inference Results",
        statistics=["RMSE", "MUE", "R2", "rho"],  # Show key metrics
        figsize=8,
        nbootstrap=500,  # Number of bootstrap samples for CI
    )
    fig_dg
    return (fig_dg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Edge-Level Analysis (ΔΔG)

        We can also visualize edge-level predictions (relative free energies) using `plot_dataset_DDGs`:
        """
    )
    return


@app.cell
def _(dataset, plot_dataset_DDGs):
    # Plot edge-level (ΔΔG) predictions
    fig_ddg = plot_dataset_DDGs(
        dataset,
        predicted_column="MAP",
        target_name="CDK8",
        title="Edge Predictions (ΔΔG)",
        statistics=["RMSE", "MUE"],
        figsize=8,
        nbootstrap=500,
    )
    fig_ddg
    return (fig_ddg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The plots show:
        - **Grey shaded regions**: ±0.5 and ±1.0 kcal/mol error zones
        - **Color gradient**: Points colored by distance from perfect prediction
        - **Bootstrap statistics**: RMSE, MUE, R², and correlation with 95% confidence intervals

        These visualizations help assess both accuracy (RMSE, MUE) and ranking ability (R², ρ).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 9. Exploring the Dataset-Centric Design

        One of MAPLE's key features is its **dataset-centric architecture**. Multiple models can add their predictions to the same dataset, making comparison easy:
        """
    )
    return


@app.cell
def _(dataset, mo):
    # Check what estimators have been applied
    estimators = dataset.get_estimators()

    mo.md(f"""
    ### Estimators Applied to Dataset

    The following models have added predictions to this dataset:

    **Estimators**: {estimators}

    You can now access all predictions directly from `dataset.dataset_nodes`:

    ```python
    # Access predictions
    dataset.dataset_nodes[['Name', 'Exp. DeltaG', 'MAP']]
    ```

    This design allows you to:
    - Train multiple models on the same dataset
    - Compare predictions side-by-side
    - Track which models have been applied
    """)
    return (estimators,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 10. Next Steps

        You've now learned the basics of MAPLE! Here are some things to explore next:

        ### Try Different Models

        ```python
        # Variational Inference (full uncertainty)
        vi_config = NodeModelConfig(
            guide_type=GuideType.AUTO_NORMAL,
            num_steps=3000,
        )

        # Gaussian Mixture VI (outlier detection)
        from maple.models import GMVI_model, GMVIConfig
        gmvi_config = GMVIConfig(prior_std=5.0, outlier_prob=0.2)
        ```

        ### Try Different Datasets

        ```python
        # Available datasets
        datasets = ["cdk8", "cmet", "eg5", "hif2a", "pfkfb3", "shp2", "syk", "tnks2"]

        # Different sampling times
        dataset = FEPDataset(dataset_name="cmet", sampling_time="20ns")
        ```

        ### Performance Tracking

        ```python
        from maple.utils import PerformanceTracker

        tracker = PerformanceTracker("./results")
        tracker.record_run(
            run_id="experiment_1",
            y_true=y_true,
            y_pred=y_pred,
            model_config={"model": "MAP", "steps": 2000},
        )
        ```

        For more details, check out the full [MAPLE documentation](https://maple-fep.readthedocs.io/).
        """
    )
    return


if __name__ == "__main__":
    app.run()
