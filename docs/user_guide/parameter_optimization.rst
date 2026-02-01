Parameter Optimization Guide
============================

This guide provides comprehensive approaches for investigating parameter effects on model performance in MAPLE, including systematic parameter sweeps, optimization, and visualization techniques.

Overview
--------

The MAPLE parameter investigation system offers multiple approaches for exploring how different parameters affect model performance:

1. **Direct Replication** - Exact recreation of existing analysis workflows
2. **Parameter Sweeps** - Systematic exploration of individual parameters  
3. **Grid Search** - Multi-parameter optimization
4. **Advanced Analysis** - Parameter interactions and cross-dataset studies

Quick Start
-----------

Simple Parameter Sweep
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.utils import ParameterSweep, PerformanceTracker
   
   tracker = PerformanceTracker("./results")
   sweep = ParameterSweep(tracker, base_config, datasets)
   
   # Investigate prior standard deviation effect
   results = sweep.sweep_parameter(
       'prior_std',
       values=[0.01, 0.1, 0.5, 1, 2, 4, 6], 
       metrics=['RMSE', 'MUE', 'R2', 'rho']
   )
   
   # Create visualization
   fig = sweep.plot_parameter_effects(results, 'prior_std')

Direct Notebook Replication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maple.utils import create_prior_sweep_experiment
   
   # Replicate existing notebook analysis
   results = create_prior_sweep_experiment(
       tracker=tracker,
       datasets=datasets,
       prior_std_values=[0.01, 0.1, 0.5, 1, 2, 4, 6]
   )

Core Components
---------------

ParameterSweep Class
~~~~~~~~~~~~~~~~~~~~

The main class for systematic parameter exploration:

.. code-block:: python

   sweep = ParameterSweep(
       tracker=tracker,        # PerformanceTracker instance
       base_config=config,     # Base ModelConfig to modify
       datasets=datasets       # Dict of datasets to test on
   )

Key methods:

* ``sweep_parameter()``: Sweep a single parameter with automatic failure tracking
* ``grid_search()``: Multi-parameter grid search with error handling
* ``find_optimal_parameters()``: Identify best parameter values
* ``plot_parameter_effects()``: Visualize parameter effects
* ``get_failed_experiments()``: Retrieve details of failed parameter combinations

Single Parameter Sweeps
------------------------

Learning Rate Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sweep learning rate
   lr_results = sweep.sweep_parameter(
       parameter_name='learning_rate',
       values=[0.0001, 0.001, 0.01, 0.1],
       metrics=['RMSE', 'R2'],
       n_runs=3  # Multiple runs for statistical reliability
   )
   
   # Find optimal value
   optimal_lr = sweep.find_optimal_parameters(lr_results, metric='RMSE')
   print(f"Optimal learning rate: {optimal_lr['optimal_value'].iloc[0]}")
   
   # Check for any parameter combinations that failed
   failed_experiments = sweep.get_failed_experiments()
   if len(failed_experiments) > 0:
       print(f"Note: {len(failed_experiments)} parameter combinations failed during optimization")

Prior Parameter Investigation  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Investigate prior standard deviation
   prior_results = sweep.sweep_parameter(
       parameter_name='prior_parameters', 
       values=[[0.0, std] for std in [0.1, 0.5, 1.0, 2.0, 5.0]],
       metrics=['RMSE', 'MUE', 'R2', 'correlation'],
       center_data=True,         # Center FEP data
       bootstrap_stats=True      # Include confidence intervals
   )
   
   # Plot effects
   fig = sweep.plot_parameter_effects(
       results_df=prior_results,
       parameter_name='prior_std',
       save_path='prior_effects.png'
   )

Training Steps Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize number of training steps
   steps_results = sweep.sweep_parameter(
       'num_steps',
       values=[500, 1000, 2000, 5000],
       metrics=['RMSE', 'R2'],
       track_convergence=True  # Monitor training progress
   )

Multi-Parameter Optimization
----------------------------

Grid Search
~~~~~~~~~~~

.. code-block:: python

   # Define parameter grid
   parameter_grid = {
       'learning_rate': [0.001, 0.01, 0.1],
       'prior_std': [0.5, 1.0, 2.0],
       'error_std': [0.1, 0.5, 1.0]
   }
   
   # Run grid search
   grid_results = sweep.grid_search(
       parameter_grid,
       metrics=['RMSE', 'R2'],
       n_runs=2  # Multiple runs per combination
   )
   
   # Find best combination
   best_params = sweep.find_optimal_parameters(grid_results, metric='RMSE')
   print("Best parameter combination:")
   print(best_params)

Sequential Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize parameters sequentially
   base_config = ModelConfig()
   
   # Step 1: Optimize learning rate
   lr_results = sweep.sweep_parameter('learning_rate', [0.001, 0.01, 0.1])
   optimal_lr = sweep.find_optimal_parameters(lr_results, 'RMSE')
   base_config.learning_rate = optimal_lr['optimal_value'].iloc[0]
   
   # Step 2: Optimize prior with optimal learning rate
   sweep.base_config = base_config
   prior_results = sweep.sweep_parameter(
       'prior_parameters', 
       [[0.0, std] for std in [0.5, 1.0, 2.0]]
   )
   optimal_prior = sweep.find_optimal_parameters(prior_results, 'RMSE')
   base_config.prior_parameters = optimal_prior['optimal_value'].iloc[0]

Cross-Dataset Analysis
----------------------

Multi-Dataset Parameter Sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load multiple datasets
   datasets = {
       "dataset_A": FEPDataset("data_A.csv"),
       "dataset_B": FEPDataset("data_B.csv"), 
       "dataset_C": FEPDataset("data_C.csv")
   }
   
   # Create sweep across all datasets
   sweep = ParameterSweep(tracker, base_config, datasets)
   
   # Sweep parameter across all datasets
   cross_results = sweep.sweep_parameter(
       'learning_rate',
       values=[0.001, 0.01, 0.1],
       metrics=['RMSE', 'R2']
   )
   
   # Find universally optimal parameters
   universal_optimal = sweep.find_optimal_parameters(
       cross_results,
       metric='RMSE', 
       aggregation='mean'  # Average across datasets
   )

Dataset-Specific Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find optimal parameters for each dataset
   dataset_optima = {}
   
   for dataset_name in datasets.keys():
       dataset_results = cross_results[cross_results['dataset'] == dataset_name]
       optimal = sweep.find_optimal_parameters(dataset_results, 'RMSE')
       dataset_optima[dataset_name] = optimal['optimal_value'].iloc[0]
   
   print("Dataset-specific optimal learning rates:")
   for dataset, lr in dataset_optima.items():
       print(f"  {dataset}: {lr}")

Advanced Analysis Techniques
----------------------------

Parameter Interaction Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze parameter interactions
   interaction_results = sweep.analyze_parameter_interactions(
       grid_results,
       parameters=['learning_rate', 'prior_std'],
       metric='RMSE'
   )
   
   # Visualize interactions
   fig = sweep.plot_parameter_interactions(
       interaction_results,
       save_path='parameter_interactions.png'
   )

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute parameter sensitivity
   sensitivity = sweep.compute_parameter_sensitivity(
       parameter_results=grid_results,
       metric='RMSE',
       baseline_config=base_config
   )
   
   print("Parameter sensitivity (higher = more sensitive):")
   for param, sens in sensitivity.items():
       print(f"  {param}: {sens:.4f}")

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Advanced Bayesian optimization (requires additional packages)
   from maple.utils.optimization import BayesianOptimizer
   
   optimizer = BayesianOptimizer(
       objective_function=sweep.evaluate_config,
       parameter_bounds={
           'learning_rate': (0.0001, 0.1),
           'prior_std': (0.1, 5.0),
           'error_std': (0.1, 2.0)
       }
   )
   
   # Run optimization
   best_config, best_score = optimizer.optimize(n_calls=50)

Visualization and Reporting
---------------------------

Parameter Effect Plots
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create comprehensive parameter effect plots
   fig = sweep.plot_parameter_effects(
       results_df=results,
       parameter_name='learning_rate',
       metrics=['RMSE', 'R2'],
       groupby='dataset',
       confidence_intervals=True,
       save_path='lr_effects.png'
   )

Heatmaps for Multi-Parameter Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create parameter interaction heatmap
   fig = sweep.plot_parameter_heatmap(
       grid_results,
       x_param='learning_rate',
       y_param='prior_std', 
       metric='RMSE',
       save_path='parameter_heatmap.png'
   )

Performance Summary Reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate comprehensive report
   report = sweep.generate_optimization_report(
       parameter_results=grid_results,
       output_path='optimization_report.html',
       include_plots=True
   )

Best Practices
--------------

Experimental Design
~~~~~~~~~~~~~~~~~~~

1. **Start Simple**: Begin with single-parameter sweeps
2. **Use Multiple Runs**: Include multiple runs for statistical reliability
3. **Include Baselines**: Always test default/literature values
4. **Document Everything**: Use descriptive run IDs and metadata

Parameter Selection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Recommended parameter ranges for common sweeps
   
   # Learning rate (log scale)
   learning_rates = [10**i for i in range(-4, 0)]  # 0.0001 to 0.1
   
   # Prior standard deviation  
   prior_stds = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
   
   # Number of training steps
   training_steps = [500, 1000, 2000, 5000, 10000]
   
   # Error standard deviation
   error_stds = [0.1, 0.2, 0.5, 1.0, 2.0]

Statistical Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Include proper statistical analysis
   results = sweep.sweep_parameter(
       'learning_rate',
       values=[0.001, 0.01, 0.1],
       metrics=['RMSE', 'R2'],
       n_runs=5,                    # Multiple runs
       bootstrap_stats=True,        # Bootstrap CIs
       statistical_tests=True       # Significance tests
   )
   
   # Check for significant differences
   significance = sweep.test_parameter_significance(
       results, 
       parameter='learning_rate',
       metric='RMSE'
   )

Convenience Functions
--------------------

For common workflows, MAPLE provides high-level convenience functions:

Prior Parameter Study
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Replicate common prior parameter analysis
   results = create_prior_sweep_experiment(
       tracker=tracker,
       datasets=datasets,
       prior_std_values=[0.01, 0.1, 0.5, 1, 2, 4, 6],
       metrics=['RMSE', 'MUE', 'R2', 'rho'],
       n_runs=3
   )

Comprehensive Parameter Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Full parameter exploration study
   results = create_comprehensive_parameter_study(
       tracker=tracker,
       datasets=datasets,
       parameter_ranges={
           'learning_rate': [0.001, 0.01, 0.1],
           'prior_std': [0.5, 1.0, 2.0],
           'num_steps': [1000, 2000]
       }
   )

Error Handling and Failure Tracking
-----------------------------------

MAPLE's parameter sweep system includes robust error handling that allows exploration to continue even when individual parameter combinations fail:

Automatic Failure Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Parameter sweep automatically handles failures gracefully  
   results = sweep.sweep_parameter(
       'num_steps',
       values=[1, 10, 100, 1000],  # Some values may fail validation
       metrics=['RMSE', 'R2']
   )
   
   # Results contain only successful experiments
   print(f"Successful experiments: {len(results)}")
   
   # Access failure details
   failed_experiments = sweep.get_failed_experiments()
   print(f"Failed experiments: {len(failed_experiments)}")

Analyzing Failed Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get detailed failure information
   failures = sweep.get_failed_experiments()
   
   if len(failures) > 0:
       # Group by error type
       error_counts = failures.groupby('error_type').size()
       print("Error types encountered:")
       for error_type, count in error_counts.items():
           print(f"  {error_type}: {count} experiments")
       
       # Show specific failures
       for _, failure in failures.iterrows():
           print(f"Failed: {failure['parameter']}={failure['parameter_value']}")
           print(f"  Error: {failure['error_type']} - {failure['error_message']}")
           print(f"  Dataset: {failure['dataset']}")

Robust Parameter Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_parameter_study(sweep, param_ranges):
       """Conduct parameter study with comprehensive error handling."""
       all_results = []
       all_failures = []
       
       for param_name, values in param_ranges.items():
           print(f"Exploring {param_name}...")
           
           # Run sweep
           results = sweep.sweep_parameter(param_name, values)
           all_results.append(results)
           
           # Track failures
           failures = sweep.get_failed_experiments()
           param_failures = failures[failures['parameter'] == param_name]
           all_failures.append(param_failures)
           
           # Report status
           success_rate = len(results) / len(values) if len(values) > 0 else 0
           print(f"  Success rate: {success_rate:.1%}")
           
           if len(param_failures) > 0:
               print(f"  Failed values: {param_failures['parameter_value'].tolist()}")
       
       return all_results, all_failures

Best Practices for Robust Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Always check for failures after parameter sweeps
   results = sweep.sweep_parameter('learning_rate', [0.001, 0.01, 0.1])
   failures = sweep.get_failed_experiments()
   
   if len(failures) > 0:
       print("⚠️ Some parameter combinations failed - check failure details")
       print("Use sweep.get_failed_experiments() for details")
   
   # 2. Validate parameter ranges before running expensive sweeps
   def validate_param_range(param_name, values):
       if param_name == 'learning_rate':
           return [v for v in values if 0 < v <= 1.0]
       elif param_name == 'num_steps': 
           return [v for v in values if v >= 10]
       return values
   
   # 3. Use conservative parameter ranges for initial exploration
   conservative_ranges = {
       'learning_rate': [0.001, 0.01, 0.1],          # Known good range
       'num_steps': [100, 500, 1000, 2000],          # Reasonable training lengths
       'error_std': [0.1, 0.5, 1.0]                  # Moderate uncertainty levels
   }

Common Failure Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter sweep system tracks several common failure types:

* **ValidationError**: Invalid parameter values (e.g., negative learning rates)
* **RuntimeError**: Training convergence issues or numerical instability  
* **AttributeError**: Configuration object errors
* **ValueError**: Data format or shape mismatches
* **TimeoutError**: Experiments that exceed time limits

Recovery and Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Continue optimization after failures
   def continue_after_failures(sweep, original_results, failed_params):
       """Continue parameter search with modified ranges."""
       
       # Analyze successful parameter ranges
       if len(original_results) > 0:
           successful_values = original_results['parameter_value'].unique()
           print(f"Previously successful values: {successful_values}")
           
           # Expand around successful values
           expanded_range = []
           for val in successful_values:
               expanded_range.extend([val * 0.8, val, val * 1.2])
           
           # Test expanded range
           new_results = sweep.sweep_parameter(
               parameter_name='learning_rate',
               values=expanded_range,
               experiment_name='recovery_sweep'
           )
           
           return new_results
       
       return pd.DataFrame()

This comprehensive parameter optimization system enables systematic investigation of parameter effects, identification of optimal configurations, and creation of publication-quality visualizations while maintaining statistical rigor throughout the process. The robust error handling ensures that parameter exploration can continue productively even when individual configurations encounter issues.
