Utils Package (``maple.utils``)
===============================

The ``maple.utils`` package provides utility functions for performance tracking, parameter optimization, and model comparison.

Overview
--------

The utils package includes:

* **PerformanceTracker**: Comprehensive performance tracking and storage
* **ParameterSweep**: Systematic parameter exploration and optimization with failure tracking
* **ModelRun**: Data structure for storing individual model run results
* **Convenience Functions**: High-level functions for common workflows

Quick Examples
--------------

Performance Tracking
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.utils import PerformanceTracker
   
   # Initialize tracker
   tracker = PerformanceTracker("./results")
   
   # Record a model run
   tracker.record_run(
       run_id="experiment_1",
       y_true=experimental_data,
       y_pred=model_predictions,
       model_config={"learning_rate": 0.01},
       dataset_info={"name": "dataset_A"}
   )
   
   # Compare multiple runs
   comparison = tracker.compare_runs(["experiment_1", "experiment_2"])

Parameter Sweeps
~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.utils import ParameterSweep
   
   # Initialize sweep
   sweep = ParameterSweep(tracker, base_config, datasets)
   
   # Sweep a parameter
   results = sweep.sweep_parameter(
       'learning_rate',
       values=[0.001, 0.01, 0.1],
       metrics=['RMSE', 'R2']
   )
   
   # Find optimal parameters
   optimal = sweep.find_optimal_parameters(results, metric='RMSE')
   
   # Check for failed parameter combinations
   failed_experiments = sweep.get_failed_experiments()
   if len(failed_experiments) > 0:
       print(f"Warning: {len(failed_experiments)} parameter combinations failed")
       print("Failed experiments:", failed_experiments)

Failure Tracking
~~~~~~~~~~~~~~~~~

The parameter sweep system automatically tracks and reports failed parameter combinations, enabling robust parameter exploration even when some configurations cause errors:

.. code-block:: python

   # Run parameter sweep with potential failures
   results = sweep.sweep_parameter(
       'num_steps', 
       values=[-1, 10, 100, 1000],  # -1 will fail validation
       metrics=['RMSE', 'R2']
   )
   
   # Results will only contain successful experiments
   print(f"Successful experiments: {len(results)}")
   
   # View details of failed experiments
   failed_df = sweep.get_failed_experiments()
   print(f"Failed experiments: {len(failed_df)}")
   
   # Analyze failure patterns
   for _, failure in failed_df.iterrows():
       print(f"Dataset: {failure['dataset']}")
       print(f"Parameter: {failure['parameter']} = {failure['parameter_value']}")
       print(f"Error: {failure['error_type']}: {failure['error_message']}")

Key features of failure tracking:

* **Graceful Handling**: Parameter sweeps continue running even when individual configurations fail
* **Detailed Recording**: Failed experiments include parameter values, error types, and error messages
* **Easy Access**: Use ``get_failed_experiments()`` to retrieve failure details as a pandas DataFrame
* **User Notifications**: Automatic warnings about failed parameter combinations with guidance to check details

API Reference
-------------

Performance Tracking
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: maple.utils.PerformanceTracker
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: maple.utils.ModelRun
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: maple.utils.load_performance_history

.. autofunction:: maple.utils.compare_model_runs

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: maple.utils.ParameterSweep
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: maple.utils.create_prior_sweep_experiment

.. autofunction:: maple.utils.create_comprehensive_parameter_study
