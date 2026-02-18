Parameter Sweep Examples
========================

This page provides detailed examples of using MAPLE's parameter sweep functionality for systematic optimization and analysis.

Example 1: Learning Rate Optimization
--------------------------------------

This example demonstrates how to systematically optimize the learning rate parameter.

Setup
~~~~~

.. code-block:: python

   from maple.utils import PerformanceTracker, ParameterSweep
   from maple.models import ModelConfig, PriorType
   from maple.dataset import FEPDataset
   import numpy as np
   
   # Initialize tracking and data
   tracker = PerformanceTracker("./learning_rate_study")
   dataset = FEPDataset("example_data.csv")
   
   # Base configuration
   base_config = ModelConfig(
       num_steps=1000,
       prior_type=PriorType.NORMAL,
       prior_parameters=[0.0, 1.0],
       error_std=0.5
   )

Execute Parameter Sweep
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create parameter sweep
   sweep = ParameterSweep(tracker, base_config, {"example": dataset})
   
   # Define learning rate values to test
   learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
   
   # Run the sweep
   print("Running learning rate optimization...")
   results = sweep.sweep_parameter(
       parameter_name='learning_rate',
       values=learning_rates,
       metrics=['RMSE', 'R2', 'MUE', 'correlation'],
       n_runs=3,  # Multiple runs for reliability
       random_seeds=[42, 123, 456]  # Reproducible results
   )
   
   print(f"Completed {len(results)} experiments")
   
   # Check for any failures
   failed_experiments = sweep.get_failed_experiments()
   if len(failed_experiments) > 0:
       print(f"Warning: {len(failed_experiments)} parameter combinations failed")
   else:
       print("All parameter combinations completed successfully")

Analyze Results
~~~~~~~~~~~~~~~

.. code-block:: python

   # Find optimal learning rate
   optimal = sweep.find_optimal_parameters(results, metric='RMSE')
   optimal_lr = optimal['optimal_value'].iloc[0]
   optimal_rmse = optimal['optimal_metric_value'].iloc[0]
   
   print(f"Optimal learning rate: {optimal_lr}")
   print(f"Best RMSE: {optimal_rmse:.4f}")
   
   # View detailed results
   summary = results.groupby('learning_rate').agg({
       'RMSE': ['mean', 'std'],
       'R2': ['mean', 'std']
   }).round(4)
   
   print("\nDetailed Results:")
   print(summary)

Visualize Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Plot parameter effects
   fig = sweep.plot_parameter_effects(
       results_df=results,
       parameter_name='learning_rate',
       metrics=['RMSE', 'R2'],
       save_path='learning_rate_effects.png'
   )
   
   # Custom detailed plot
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # RMSE vs Learning Rate
   sns.lineplot(data=results, x='learning_rate', y='RMSE', 
                marker='o', ax=axes[0,0])
   axes[0,0].set_xscale('log')
   axes[0,0].set_title('RMSE vs Learning Rate')
   axes[0,0].grid(True, alpha=0.3)
   
   # R² vs Learning Rate
   sns.lineplot(data=results, x='learning_rate', y='R2', 
                marker='o', ax=axes[0,1], color='orange')
   axes[0,1].set_xscale('log')
   axes[0,1].set_title('R² vs Learning Rate')
   axes[0,1].grid(True, alpha=0.3)
   
   # Box plot of RMSE distribution
   sns.boxplot(data=results, x='learning_rate', y='RMSE', ax=axes[1,0])
   axes[1,0].tick_params(axis='x', rotation=45)
   axes[1,0].set_title('RMSE Distribution by Learning Rate')
   
   # Convergence analysis (if tracked)
   if 'final_loss' in results.columns:
       sns.scatterplot(data=results, x='learning_rate', y='final_loss', 
                      ax=axes[1,1])
       axes[1,1].set_xscale('log')
       axes[1,1].set_title('Final Training Loss')
   
   plt.tight_layout()
   plt.savefig('learning_rate_detailed_analysis.png', dpi=300)
   plt.show()

Example 2: Prior Parameter Investigation
-----------------------------------------

This example replicates a typical prior standard deviation study similar to notebook analyses.

Setup and Execution
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prior standard deviation values to investigate
   prior_std_values = [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
   
   # Convert to prior_parameters format
   prior_params = [[0.0, std] for std in prior_std_values]
   
   print("Running prior standard deviation study...")
   prior_results = sweep.sweep_parameter(
       parameter_name='prior_parameters',
       values=prior_params,
       metrics=['RMSE', 'MUE', 'R2', 'correlation'],
       center_data=True,  # Center FEP data as in typical analyses
       bootstrap_stats=True,  # Include confidence intervals
       n_bootstrap=1000
   )

Create Publication-Quality Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add prior_std column for easier plotting
   prior_results['prior_std'] = [params[1] for params in prior_results['prior_parameters']]
   
   # Create faceted plot similar to notebook style
   import seaborn as sns
   
   # Prepare data for plotting
   plot_data = prior_results[prior_results['prior_std'] < 6].copy()
   
   # Create FacetGrid
   g = sns.FacetGrid(
       plot_data, 
       col='metric_name',  # Assumes melted format
       hue='dataset',
       col_wrap=2,
       height=4,
       aspect=1.2
   )
   
   # Plot lines with confidence intervals
   g.map(sns.lineplot, 'prior_std', 'metric_value', marker='o')
   
   # Add confidence intervals if available
   if 'ci_lower' in plot_data.columns:
       g.map(plt.fill_between, 'prior_std', 'ci_lower', 'ci_upper', alpha=0.3)
   
   g.add_legend()
   g.fig.suptitle('Prior Standard Deviation Effects', y=1.02)
   
   plt.tight_layout()
   plt.savefig('prior_std_effects_publication.png', dpi=300, bbox_inches='tight')
   plt.show()

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find optimal prior standard deviation
   optimal_prior = sweep.find_optimal_parameters(prior_results, metric='RMSE')
   print(f"Optimal prior std: {optimal_prior['optimal_value'].iloc[0][1]}")
   
   # Test for significant differences between prior values
   from scipy.stats import f_oneway
   
   rmse_by_prior = {}
   for std in prior_std_values:
       mask = prior_results['prior_std'] == std
       rmse_by_prior[std] = prior_results[mask]['RMSE'].values
   
   # ANOVA test
   f_stat, p_value = f_oneway(*rmse_by_prior.values())
   print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")

Example 3: Multi-Parameter Grid Search
---------------------------------------

This example demonstrates optimization across multiple parameters simultaneously.

Define Parameter Grid
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define comprehensive parameter grid
   parameter_grid = {
       'learning_rate': [0.001, 0.01, 0.1],
       'prior_parameters': [[0.0, 0.5], [0.0, 1.0], [0.0, 2.0]],
       'error_std': [0.1, 0.5, 1.0],
       'num_steps': [500, 1000, 2000]
   }
   
   # Calculate total combinations
   total_combinations = np.prod([len(values) for values in parameter_grid.values()])
   print(f"Total parameter combinations: {total_combinations}")

Execute Grid Search
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run grid search (this may take a while)
   print("Starting grid search...")
   grid_results = sweep.grid_search(
       parameter_grid=parameter_grid,
       metrics=['RMSE', 'R2'],
       n_runs=2,  # Reduce for faster execution
       parallel=True,  # Use parallel execution if available
       max_workers=4
   )
   
   print(f"Grid search completed: {len(grid_results)} total runs")

Analyze Grid Search Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find best overall combination
   best_combo = sweep.find_optimal_parameters(grid_results, metric='RMSE')
   print("Best parameter combination:")
   for param, value in best_combo.iloc[0].items():
       if param.endswith('_value'):
           print(f"  {param.replace('_optimal_value', '')}: {value}")
   
   # Analyze parameter importance
   param_importance = sweep.compute_parameter_importance(
       grid_results, 
       metric='RMSE'
   )
   
   print("\nParameter importance (variance explained):")
   for param, importance in sorted(param_importance.items(), 
                                 key=lambda x: x[1], reverse=True):
       print(f"  {param}: {importance:.3f}")

Visualize Grid Search Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create parameter interaction heatmaps
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # Learning rate vs Prior std
   heatmap_data = grid_results.pivot_table(
       index='learning_rate', 
       columns='prior_std',  # Extract from prior_parameters
       values='RMSE',
       aggfunc='mean'
   )
   
   sns.heatmap(heatmap_data, annot=True, fmt='.3f', ax=axes[0,0])
   axes[0,0].set_title('RMSE: Learning Rate vs Prior Std')
   
   # Learning rate vs Error std
   heatmap_data2 = grid_results.pivot_table(
       index='learning_rate',
       columns='error_std', 
       values='RMSE',
       aggfunc='mean'
   )
   
   sns.heatmap(heatmap_data2, annot=True, fmt='.3f', ax=axes[0,1])
   axes[0,1].set_title('RMSE: Learning Rate vs Error Std')
   
   # Prior std vs Error std
   heatmap_data3 = grid_results.pivot_table(
       index='prior_std',
       columns='error_std',
       values='RMSE', 
       aggfunc='mean'
   )
   
   sns.heatmap(heatmap_data3, annot=True, fmt='.3f', ax=axes[1,0])
   axes[1,0].set_title('RMSE: Prior Std vs Error Std')
   
   # Parameter sensitivity bar plot
   param_names = list(param_importance.keys())
   importances = list(param_importance.values())
   
   axes[1,1].bar(param_names, importances)
   axes[1,1].set_title('Parameter Importance')
   axes[1,1].tick_params(axis='x', rotation=45)
   
   plt.tight_layout()
   plt.savefig('grid_search_analysis.png', dpi=300, bbox_inches='tight')
   plt.show()

Example 4: Cross-Dataset Parameter Study
-----------------------------------------

This example shows how to find parameters that work well across multiple datasets.

Setup Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load multiple datasets
   datasets = {
       "CDK2": FEPDataset("cdk2_data.csv"),
       "FXA": FEPDataset("fxa_data.csv"), 
       "THROMBIN": FEPDataset("thrombin_data.csv")
   }
   
   # Inspect dataset characteristics
   print("Dataset characteristics:")
   for name, dataset in datasets.items():
       graph_data = dataset.get_graph_data()
       print(f"  {name}: {graph_data['N']} nodes, {graph_data['M']} edges")

Cross-Dataset Parameter Sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create cross-dataset sweep
   cross_sweep = ParameterSweep(tracker, base_config, datasets)
   
   # Test learning rates across all datasets
   cross_results = cross_sweep.sweep_parameter(
       'learning_rate',
       values=[0.001, 0.005, 0.01, 0.05, 0.1],
       metrics=['RMSE', 'R2']
   )
   
   print(f"Cross-dataset sweep completed: {len(cross_results)} runs")

Find Universal Optimal Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find learning rate that works best on average
   universal_optimal = cross_sweep.find_optimal_parameters(
       cross_results,
       metric='RMSE',
       aggregation='mean'  # Average across datasets
   )
   
   universal_lr = universal_optimal['optimal_value'].iloc[0]
   print(f"Universal optimal learning rate: {universal_lr}")
   
   # Find parameters that minimize worst-case performance
   robust_optimal = cross_sweep.find_optimal_parameters(
       cross_results,
       metric='RMSE', 
       aggregation='max'  # Minimize maximum RMSE
   )
   
   robust_lr = robust_optimal['optimal_value'].iloc[0]
   print(f"Robust optimal learning rate: {robust_lr}")

Cross-Dataset Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot cross-dataset performance
   fig, axes = plt.subplots(1, 2, figsize=(14, 6))
   
   # Line plot showing all datasets
   sns.lineplot(
       data=cross_results,
       x='learning_rate', 
       y='RMSE',
       hue='dataset',
       marker='o',
       ax=axes[0]
   )
   axes[0].set_xscale('log')
   axes[0].set_title('RMSE Across Datasets')
   axes[0].grid(True, alpha=0.3)
   
   # Heatmap of performance by dataset
   heatmap_data = cross_results.pivot_table(
       index='dataset',
       columns='learning_rate', 
       values='RMSE',
       aggfunc='mean'
   )
   
   sns.heatmap(heatmap_data, annot=True, fmt='.3f', ax=axes[1])
   axes[1].set_title('RMSE Heatmap: Dataset vs Learning Rate')
   
   plt.tight_layout()
   plt.savefig('cross_dataset_analysis.png', dpi=300, bbox_inches='tight')
   plt.show()

Example 5: Error Handling and Failure Tracking
-----------------------------------------------

This example demonstrates how MAPLE's parameter sweep system gracefully handles parameter configurations that cause errors during training or evaluation.

Robust Parameter Sweep
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test a range including potentially problematic values
   test_values = {
       'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],  # Very high values may fail
       'num_steps': [1, 10, 100, 1000, 10000],                   # Very low values may fail
       'error_std': [0.001, 0.01, 0.1, 1.0, 100.0]              # Extreme values may cause issues
   }
   
   print("Testing potentially problematic parameter ranges...")
   for param_name, values in test_values.items():
       print(f"\nTesting {param_name} with values: {values}")
       
       # Run parameter sweep
       results = sweep.sweep_parameter(
           parameter_name=param_name,
           values=values,
           metrics=['RMSE', 'R2'],
           experiment_name=f'{param_name}_robustness_test'
       )
       
       # Check results and failures
       print(f"Successful experiments: {len(results)}")
       
       # Analyze failures
       failed_experiments = sweep.get_failed_experiments()
       param_failures = failed_experiments[
           failed_experiments['parameter'] == param_name
       ]
       
       if len(param_failures) > 0:
           print(f"Failed parameter values: {len(param_failures)}")
           
           # Show failure details
           for _, failure in param_failures.iterrows():
               print(f"  {param_name}={failure['parameter_value']}: "
                    f"{failure['error_type']} - {failure['error_message'][:50]}...")
       else:
           print("All parameter values succeeded!")

Failure Analysis and Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Comprehensive failure analysis
   all_failures = sweep.get_failed_experiments()
   
   if len(all_failures) > 0:
       print(f"\nTotal failed experiments: {len(all_failures)}")
       
       # Group failures by error type
       error_summary = all_failures.groupby('error_type').size().sort_values(ascending=False)
       print("\nFailure types:")
       for error_type, count in error_summary.items():
           print(f"  {error_type}: {count} experiments")
       
       # Group failures by parameter
       param_summary = all_failures.groupby('parameter').size().sort_values(ascending=False)
       print("\nMost problematic parameters:")
       for param, count in param_summary.items():
           print(f"  {param}: {count} failures")
       
       # Show most common failure scenarios
       print("\nMost common failure scenarios:")
       failure_scenarios = all_failures.groupby(['parameter', 'parameter_value']).size()
       for (param, value), count in failure_scenarios.head(5).items():
           print(f"  {param}={value}: {count} failures")

Recovery Strategies
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Strategy 1: Retry with modified parameters
   def retry_failed_experiments(sweep, max_retries=2):
       """Retry failed experiments with slightly modified parameters."""
       failures = sweep.get_failed_experiments()
       recovery_results = []
       
       for _, failure in failures.iterrows():
           param_name = failure['parameter']
           original_value = failure['parameter_value']
           
           # Generate alternative values to try
           alternatives = []
           if isinstance(original_value, (int, float)):
               if original_value > 0:
                   alternatives = [original_value * 0.8, original_value * 1.2]
               else:
                   alternatives = [0.01, 0.1]  # Safe defaults
           
           # Try alternatives
           for alt_value in alternatives[:max_retries]:
               print(f"Retrying {param_name}={original_value} with {alt_value}")
               try:
                   retry_results = sweep.sweep_parameter(
                       parameter_name=param_name,
                       values=[alt_value],
                       metrics=['RMSE'],
                       experiment_name=f'retry_{param_name}'
                   )
                   if len(retry_results) > 0:
                       recovery_results.append({
                           'original_value': original_value,
                           'recovery_value': alt_value,
                           'success': True
                       })
                       break
               except Exception as e:
                   print(f"Retry with {alt_value} also failed: {e}")
       
       return recovery_results
   
   # Execute recovery
   recovered = retry_failed_experiments(sweep)
   print(f"Successfully recovered {len(recovered)} failed experiments")

Parameter Validation and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_parameters(param_name, values):
       """Validate and filter parameter values before running sweep."""
       valid_values = []
       warnings = []
       
       for value in values:
           valid = True
           warning = None
           
           # Parameter-specific validation rules
           if param_name == 'learning_rate':
               if value <= 0 or value > 1:
                   valid = False
                   warning = f"Learning rate {value} outside reasonable range (0, 1]"
           elif param_name == 'num_steps':
               if value < 10:
                   valid = False
                   warning = f"num_steps {value} too small (minimum 10)"
           elif param_name == 'error_std':
               if value <= 0:
                   valid = False
                   warning = f"error_std {value} must be positive"
           
           if valid:
               valid_values.append(value)
           else:
               warnings.append(warning)
       
       return valid_values, warnings
   
   # Use validation before running sweeps
   param_name = 'learning_rate'
   proposed_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
   
   valid_values, validation_warnings = validate_parameters(param_name, proposed_values)
   
   if validation_warnings:
       print("Parameter validation warnings:")
       for warning in validation_warnings:
           print(f"  {warning}")
   
   print(f"Using {len(valid_values)} valid values: {valid_values}")
   
   # Run sweep with validated parameters
   results = sweep.sweep_parameter(
       parameter_name=param_name,
       values=valid_values,
       metrics=['RMSE', 'R2']
   )

Best Practices for Robust Parameter Sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Best practice: Always check for failures after parameter sweeps
   def robust_parameter_sweep(sweep, param_name, values, **kwargs):
       """Run parameter sweep with comprehensive error handling."""
       
       print(f"🔍 Starting robust sweep of {param_name}")
       print(f"   Testing {len(values)} parameter values")
       
       # Run the sweep
       results = sweep.sweep_parameter(
           parameter_name=param_name,
           values=values,
           **kwargs
       )
       
       # Check results
       failures = sweep.get_failed_experiments()
       recent_failures = failures[failures['parameter'] == param_name]
       
       success_rate = (len(values) - len(recent_failures)) / len(values) * 100
       
       print(f"✅ Sweep completed:")
       print(f"   Success rate: {success_rate:.1f}% ({len(results)//len(kwargs.get('metrics', [1]))} successful)")
       print(f"   Failures: {len(recent_failures)}")
       
       if len(recent_failures) > 0:
           print("⚠️  Failed parameter values:")
           for _, failure in recent_failures.iterrows():
               print(f"   {param_name}={failure['parameter_value']}: {failure['error_type']}")
       
       # Provide recommendations
       if success_rate < 80:
           print("💡 Recommendations:")
           print("   - Consider narrowing parameter ranges")
           print("   - Check parameter validation rules")
           print("   - Review error messages for patterns")
       
       return results, recent_failures
   
   # Example usage
   results, failures = robust_parameter_sweep(
       sweep=sweep,
       param_name='learning_rate',
       values=[0.0001, 0.001, 0.01, 0.1],
       metrics=['RMSE', 'R2'],
       experiment_name='robust_lr_sweep'
   )

These examples demonstrate the comprehensive failure tracking and error handling capabilities in MAPLE's parameter sweep system. The robust design ensures that parameter exploration can continue even when individual configurations fail, while providing detailed information about failures to help users understand and resolve issues.

Summary
-------

These examples demonstrate the full range of parameter sweep capabilities in MAPLE, from simple single-parameter optimization to complex cross-dataset studies with comprehensive error handling. The systematic approach ensures reproducible results and enables identification of robust parameter settings even in the presence of challenging parameter configurations.
