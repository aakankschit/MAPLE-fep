Graph Analysis Package (``maple.graph_analysis``)
==================================================

The ``maple.graph_analysis`` package provides performance statistics, plotting functionality, and graph-based analysis tools for FEP data.

Overview
--------

The graph analysis package includes:

* **Performance Statistics**: Statistical metrics (RMSE, MAE, R², Pearson ρ, Kendall τ) with bootstrap confidence intervals
* **Single Model Plotting**: Scatter plots for individual model predictions (ΔG and ΔΔG)
* **Multi-Model Comparison**: Bar plots and correlation plots for comparing multiple models
* **Error Analysis**: KDE plots of prediction errors with fitted distributions
* **Graph Setup**: Tools for graph construction and analysis
* **Cycle Analysis**: Graph cycle detection and analysis

The plotting functions support all MAPLE models (MAP, VI, GMVI, MLE, WCC) with automatic detection and consistent color schemes.

Quick Examples
--------------

Performance Statistics
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.graph_analysis import calculate_rmse, calculate_r2, bootstrap_statistic
   
   # Calculate basic metrics
   rmse = calculate_rmse(y_true, y_pred)
   r2 = calculate_r2(y_true, y_pred)
   
   # Bootstrap confidence intervals
   ci_low, ci_high = bootstrap_statistic(
       y_true, y_pred, 
       statistic_func=calculate_rmse,
       n_bootstrap=1000
   )

Plotting
~~~~~~~~

.. code-block:: python

   from maple.graph_analysis import (
       plot_dataset_DDGs,
       plot_model_comparison_bars,
       plot_model_comparison_correlation,
       plot_error_distribution
   )

   # Single model scatter plot
   fig = plot_dataset_DDGs(dataset, predicted_column='MAP')

   # Compare multiple models with bar plots
   fig = plot_model_comparison_bars(dataset, data_type='edges')

   # Side-by-side correlation plots
   fig = plot_model_comparison_correlation(
       dataset,
       models=['MAP', 'VI', 'GMVI', 'WCC'],
       data_type='edges'
   )

   # Analyze error distribution
   fig = plot_error_distribution(dataset, predicted_column='WCC')

API Reference
-------------

Performance Statistics
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: maple.graph_analysis.calculate_mae

.. autofunction:: maple.graph_analysis.calculate_rmse

.. autofunction:: maple.graph_analysis.calculate_r2

.. autofunction:: maple.graph_analysis.calculate_correlation

.. autofunction:: maple.graph_analysis.bootstrap_statistic

.. autofunction:: maple.graph_analysis.compute_simple_statistics

Plotting Functions
~~~~~~~~~~~~~~~~~~

Single Model Plotting
^^^^^^^^^^^^^^^^^^^^^

These functions create scatter plots for a single model's predictions.

.. autofunction:: maple.graph_analysis.plot_dataset_DDGs

.. autofunction:: maple.graph_analysis.plot_dataset_DGs

.. autofunction:: maple.graph_analysis.plot_dataset_all_DDGs

Multi-Model Comparison
^^^^^^^^^^^^^^^^^^^^^^

These functions compare multiple models side-by-side with consistent colors.

**Supported Models:**

* MAP (blue) - Maximum A Posteriori
* VI (green) - Variational Inference
* GMVI (orange) - Graph-Modified Variational Inference
* MLE (purple) - Maximum Likelihood Estimation
* WCC (red) - Weighted Cycle Closure

.. autofunction:: maple.graph_analysis.plot_model_comparison_bars

.. autofunction:: maple.graph_analysis.plot_model_comparison_correlation

Error Analysis
^^^^^^^^^^^^^^

Analyze prediction error distributions to assess model quality and determine appropriate error models.

.. autofunction:: maple.graph_analysis.plot_error_distribution
