#!/usr/bin/env python
"""
Benchmark Script for FEP Datasets
This script trains MAP, VI, GMVI, and MLE models on multiple FEP benchmark datasets
and generates correlation and performance plots for both edges and nodes.
For edges, CCC values from the benchmark dataset are also included in the analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from maple.dataset import FEPBenchmarkDataset, FEPDataset
from maple.models import (
    GMVI_model, GMVIConfig, NodeModel, NodeModelConfig,
    GuideType, PriorType, ErrorDistributionType
)
from maple.graph_analysis import bootstrap_statistic

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def train_models(dataset, dataset_name):
    """Train MAP, VI, GMVI, and MLE models on the given dataset."""
    
    print(f"\n{'='*60}")
    print(f"Training models for {dataset_name}")
    print('='*60)
    
    # 1. MAP Model (AutoDelta guide)
    print("\n1. Training MAP Model...")
    map_config = NodeModelConfig(
        prior_type=PriorType.NORMAL,
        prior_parameters=[0.0, 1.0],
        guide_type=GuideType.AUTO_DELTA,
        num_steps=10000,
        learning_rate=0.001,
        error_std=1.0
    )
    map_model = NodeModel(config=map_config, dataset=dataset)
    map_results = map_model.train()
    map_model.add_predictions_to_dataset()
    print(f"   MAP final loss: {map_results['final_loss']:.4f}")
    
    # 2. VI Model (AutoNormal guide)
    print("\n2. Training VI Model...")
    vi_config = NodeModelConfig(
        prior_type=PriorType.NORMAL,
        prior_parameters=[0.0, 1.0],
        guide_type=GuideType.AUTO_NORMAL,
        num_steps=10000,
        learning_rate=0.001,
        error_std=1.0
    )
    vi_model = NodeModel(config=vi_config, dataset=dataset)
    vi_results = vi_model.train()
    vi_model.add_predictions_to_dataset()
    print(f"   VI final loss: {vi_results['final_loss']:.4f}")
    
    # 3. GMVI Model
    print("\n3. Training GMVI Model...")
    gmvi_config = GMVIConfig(
        prior_std=1.0,          # Prior standard deviation for node values
        normal_std=1.0,         # Standard deviation for normal edges
        outlier_std=3.0,        # Standard deviation for outlier edges
        outlier_prob=0.2,       # Probability of an edge being an outlier
        kl_weight=0.1,          # Weight for KL divergence in ELBO
        learning_rate=0.001,     # Learning rate for ADAM optimizer
        n_epochs=5000,           # Number of training epochs
        n_samples=1000,          # Monte Carlo samples for ELBO
        patience=500             # Early stopping patience
    )
    gmvi_model = GMVI_model(dataset=dataset, config=gmvi_config)
    gmvi_model.initialize_parameters()
    gmvi_model.fit()
    gmvi_predictions = gmvi_model.get_posterior_estimates()
    # Add GMVI predictions to dataset
    gmvi_model.add_predictions_to_dataset()
    print("   GMVI training complete")

    # 4. MLE Model (uses skewed normal error distribution)
    print("\n4. Training MLE Model...")
    mle_config = NodeModelConfig(
        prior_type=PriorType.UNIFORM,
        prior_parameters=[-10, 10],
        guide_type=GuideType.AUTO_DELTA,
        num_steps=10000,
        learning_rate=0.001,
        error_std=1.0,
        error_distribution=ErrorDistributionType.SKEWED_NORMAL
    )
    mle_model = NodeModel(config=mle_config, dataset=dataset)
    mle_results = mle_model.train()
    mle_model.add_predictions_to_dataset()
    print(f"   MLE final loss: {mle_results['final_loss']:.4f}")

    # Debug: Print available columns
    print(f"\n   Debug - Node columns: {list(dataset.dataset_nodes.columns)}")
    print(f"   Debug - Edge columns: {list(dataset.dataset_edges.columns)}")

    return dataset

def create_edge_correlation_plot(dataset, dataset_name, output_dir):
    """Create 1x5 grid of edge correlation plots for all models (mean-centered)."""

    print(f"\nCreating edge correlation plots for {dataset_name}...")
    print(f"   Available edge columns: {list(dataset.dataset_edges.columns)}")

    fig, axes = plt.subplots(1, 5, figsize=(30, 8))
    axes = axes.flatten()

    # Models to plot (CCC values come from dataset)
    models = [
        ('MAP', 'blue', 'MAP Model (AutoDelta)', 0),
        ('VI', 'green', 'VI Model (AutoNormal)', 1),
        ('GMVI', 'orange', 'GMVI Model', 2),
        ('MLE', 'purple', 'MLE Model', 3),
        ('CCC', 'red', 'CCC (from dataset)', 4)
    ]

    # Get experimental edge values and mean-center them
    exp_values = dataset.dataset_edges['Experimental DeltaDeltaG'].values
    exp_mean = np.mean(exp_values)
    exp_values_centered = exp_values - exp_mean

    print(f"   Mean-centering edge values (subtracting mean = {exp_mean:.3f})")

    for model_name, color, title, idx in models:
        ax = axes[idx]

        if model_name not in dataset.dataset_edges.columns:
            ax.text(0.5, 0.5, f'{model_name} not available',
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=18, fontweight='bold')
            continue

        pred_values = dataset.dataset_edges[model_name].values
        pred_mean = np.mean(pred_values)
        pred_values_centered = pred_values - pred_mean

        # Scatter plot
        ax.scatter(exp_values_centered, pred_values_centered, alpha=0.6, color=color, s=50, zorder=5)

        # Perfect correlation line
        min_val = min(exp_values_centered.min(), pred_values_centered.min()) - 1
        max_val = max(exp_values_centered.max(), pred_values_centered.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
               label='Perfect correlation', zorder=4)

        # Add error bands
        x_band = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_band, x_band - 1, x_band + 1, alpha=0.15,
                       color='gray', label='±1 kcal/mol', zorder=1)
        ax.fill_between(x_band, x_band - 0.5, x_band + 0.5, alpha=0.25,
                       color='gray', label='±0.5 kcal/mol', zorder=2)

        ax.set_xlabel('Experimental ΔΔG (mean-centered, kcal/mol)', fontsize=16)
        ax.set_ylabel(f'{model_name} Predicted ΔΔG (mean-centered, kcal/mol)', fontsize=16)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Calculate and add statistics box - positioned at bottom right
        metrics_text = ""
        for metric in ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']:
            try:
                result = bootstrap_statistic(
                    y_true=exp_values_centered,
                    y_pred=pred_values_centered,
                    statistic=metric,
                    nbootstrap=1000
                )
                metrics_text += f"{metric}: {result['mean']:.3f} [{result['low']:.3f}, {result['high']:.3f}]\n"
            except:
                pass

        if metrics_text:
            ax.text(0.95, 0.05, metrics_text.strip(), transform=ax.transAxes,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   fontsize=14, zorder=10)

    plt.suptitle(f'{dataset_name.upper()} - Edge Predictions: Model Comparison with Bootstrap Statistics (Mean-Centered)',
                fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Save plot
    filename = output_dir / f"{dataset_name}_edge_correlation.pdf"
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filename}")

def create_node_correlation_plot(dataset, dataset_name, output_dir):
    """Create 1x4 grid of node correlation plots for all models (mean-centered)."""

    print(f"Creating node correlation plots for {dataset_name}...")

    fig, axes = plt.subplots(1, 4, figsize=(24, 8))
    axes = axes.flatten()

    # Get experimental node values and mean-center them
    exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
    exp_mean = np.mean(exp_values)
    exp_values_centered = exp_values - exp_mean

    print(f"   Mean-centering node values (subtracting mean = {exp_mean:.3f})")

    # Models to plot (no CCC for nodes)
    models = [
        ('MAP', 'blue', 'MAP Model (AutoDelta)', 0),
        ('VI', 'green', 'VI Model (AutoNormal)', 1),
        ('GMVI', 'orange', 'GMVI Model', 2),
        ('MLE', 'purple', 'MLE Model', 3)
    ]
    
    for model_name, color, title, idx in models:
        ax = axes[idx]
        
        if model_name not in dataset.dataset_nodes.columns:
            ax.text(0.5, 0.5, f'{model_name} not available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=18, fontweight='bold')
            continue
        
        pred_values = dataset.dataset_nodes[model_name].values
        pred_mean = np.mean(pred_values)
        pred_values_centered = pred_values - pred_mean
        
        # Scatter plot with error bars if uncertainties are available
        if f'{model_name}_uncertainty' in dataset.dataset_nodes.columns:
            pred_errors = dataset.dataset_nodes[f'{model_name}_uncertainty'].values
            ax.errorbar(exp_values_centered, pred_values_centered, yerr=pred_errors, fmt='o', alpha=0.6,
                       color=color, markersize=6, elinewidth=1, capsize=3, zorder=5)
        else:
            ax.scatter(exp_values_centered, pred_values_centered, alpha=0.6, 
                      color=color, s=50, zorder=5)
        
        # Perfect correlation line
        min_val = min(exp_values_centered.min(), pred_values_centered.min()) - 1
        max_val = max(exp_values_centered.max(), pred_values_centered.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               label='Perfect correlation', zorder=4)
        
        # Add error bands
        x_band = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_band, x_band - 1, x_band + 1, alpha=0.15, 
                       color='gray', label='±1 kcal/mol', zorder=1)
        ax.fill_between(x_band, x_band - 0.5, x_band + 0.5, alpha=0.25, 
                       color='gray', label='±0.5 kcal/mol', zorder=2)
        
        ax.set_xlabel('Experimental ΔG (mean-centered, kcal/mol)', fontsize=16)
        ax.set_ylabel(f'{model_name} Predicted ΔG (mean-centered, kcal/mol)', fontsize=16)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.legend(loc='upper left', fontsize=14)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Calculate and add statistics box - positioned at bottom right
        metrics_text = ""
        for metric in ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']:
            try:
                result = bootstrap_statistic(
                    y_true=exp_values_centered,
                    y_pred=pred_values_centered,
                    statistic=metric,
                    nbootstrap=1000
                )
                metrics_text += f"{metric}: {result['mean']:.3f} [{result['low']:.3f}, {result['high']:.3f}]\n"
            except:
                pass
        
        if metrics_text:
            ax.text(0.95, 0.05, metrics_text.strip(), transform=ax.transAxes,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   fontsize=12, zorder=10)
    
    plt.suptitle(f'{dataset_name.upper()} - Node Predictions: Model Comparison with Bootstrap Statistics (Mean-Centered)', 
                fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Save plot
    filename = output_dir / f"{dataset_name}_node_correlation.pdf"
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filename}")

def create_edge_performance_plot(dataset, dataset_name, output_dir):
    """Create bar plot of edge performance metrics for all models (mean-centered)."""

    print(f"Creating edge performance bar plot for {dataset_name}...")

    # Calculate performance metrics for all models (CCC comes from dataset)
    models = ['MAP', 'VI', 'GMVI', 'MLE', 'CCC']
    statistics_to_calculate = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']

    exp_values = dataset.dataset_edges['Experimental DeltaDeltaG'].values
    exp_mean = np.mean(exp_values)
    exp_centered = exp_values - exp_mean

    all_metrics = {}

    for model in models:
        if model not in dataset.dataset_edges.columns:
            continue

        pred_values = dataset.dataset_edges[model].values
        pred_mean = np.mean(pred_values)
        pred_centered = pred_values - pred_mean

        all_metrics[model] = {}

        for statistic in statistics_to_calculate:
            result = bootstrap_statistic(
                y_true=exp_centered,
                y_pred=pred_centered,
                statistic=statistic,
                nbootstrap=1000
            )
            all_metrics[model][statistic] = {
                'mean': result['mean'],
                'ci_lower': result['low'],
                'ci_upper': result['high'],
                'std': result['stderr']
            }

    # Create bar plot in 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Arrange metrics: Top row: R2, rho, KTAU; Bottom row: RMSE, MUE, empty
    metric_positions = {
        'R2': (0, 0),
        'rho': (0, 1),
        'KTAU': (0, 2),
        'RMSE': (1, 0),
        'MUE': (1, 1)
    }

    # Hide the unused subplot
    axes[1, 2].axis('off')

    for statistic in statistics_to_calculate:
        if statistic not in metric_positions:
            continue
        ax = axes[metric_positions[statistic]]

        means = []
        errors = []
        labels = []
        colors_list = []

        for model, color in zip(models, ['blue', 'green', 'orange', 'purple', 'red']):
            if model in all_metrics:
                means.append(all_metrics[model][statistic]['mean'])
                ci_lower = all_metrics[model][statistic]['ci_lower']
                ci_upper = all_metrics[model][statistic]['ci_upper']
                errors.append([all_metrics[model][statistic]['mean'] - ci_lower,
                             ci_upper - all_metrics[model][statistic]['mean']])
                labels.append(model)
                colors_list.append(color)

        if means:
            x_pos = np.arange(len(means))
            bars = ax.bar(x_pos, means, color=colors_list, alpha=0.7)
            ax.errorbar(x_pos, means, yerr=np.array(errors).T, fmt='none',
                       capsize=5, color='black', alpha=0.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=16)
            ax.set_title(statistic, fontsize=22, fontweight='bold')
            ax.set_ylabel('Value', fontsize=20)
            ax.grid(True, axis='y', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=16)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=16)

    plt.suptitle(f'{dataset_name.upper()} - Edge Performance Metrics Comparison (Mean-Centered)',
                fontsize=24, fontweight='bold', y=1.05)
    plt.tight_layout()

    # Save plot
    filename = output_dir / f"{dataset_name}_edge_performance.pdf"
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filename}")

def create_node_performance_plot(dataset, dataset_name, output_dir):
    """Create bar plot of node performance metrics for all models (mean-centered)."""

    print(f"Creating node performance bar plot for {dataset_name}...")

    # Calculate performance metrics for all models (no CCC for nodes)
    models = ['MAP', 'VI', 'GMVI', 'MLE']
    statistics_to_calculate = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']

    exp_values = dataset.dataset_nodes['Exp. DeltaG'].values
    exp_mean = np.mean(exp_values)
    exp_centered = exp_values - exp_mean

    all_metrics = {}

    for model in models:
        if model not in dataset.dataset_nodes.columns:
            continue

        pred_values = dataset.dataset_nodes[model].values
        pred_mean = np.mean(pred_values)
        pred_centered = pred_values - pred_mean

        all_metrics[model] = {}

        for statistic in statistics_to_calculate:
            result = bootstrap_statistic(
                y_true=exp_centered,
                y_pred=pred_centered,
                statistic=statistic,
                nbootstrap=1000
            )
            all_metrics[model][statistic] = {
                'mean': result['mean'],
                'ci_lower': result['low'],
                'ci_upper': result['high'],
                'std': result['stderr']
            }

    # Create bar plot in 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Arrange metrics: Top row: R2, rho, KTAU; Bottom row: RMSE, MUE, empty
    metric_positions = {
        'R2': (0, 0),
        'rho': (0, 1),
        'KTAU': (0, 2),
        'RMSE': (1, 0),
        'MUE': (1, 1)
    }

    # Hide the unused subplot
    axes[1, 2].axis('off')

    for statistic in statistics_to_calculate:
        if statistic not in metric_positions:
            continue
        ax = axes[metric_positions[statistic]]

        means = []
        errors = []
        labels = []
        colors_list = []

        for model, color in zip(models, ['blue', 'green', 'orange', 'purple']):
            if model in all_metrics:
                means.append(all_metrics[model][statistic]['mean'])
                ci_lower = all_metrics[model][statistic]['ci_lower']
                ci_upper = all_metrics[model][statistic]['ci_upper']
                errors.append([all_metrics[model][statistic]['mean'] - ci_lower,
                             ci_upper - all_metrics[model][statistic]['mean']])
                labels.append(model)
                colors_list.append(color)

        if means:
            x_pos = np.arange(len(means))
            bars = ax.bar(x_pos, means, color=colors_list, alpha=0.7)
            ax.errorbar(x_pos, means, yerr=np.array(errors).T, fmt='none',
                       capsize=5, color='black', alpha=0.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=16)
            ax.set_title(statistic, fontsize=22, fontweight='bold')
            ax.set_ylabel('Value', fontsize=20)
            ax.grid(True, axis='y', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=16)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=16)

    plt.suptitle(f'{dataset_name.upper()} - Node Performance Metrics Comparison (Mean-Centered)',
                fontsize=24, fontweight='bold', y=1.05)
    plt.tight_layout()

    # Save plot
    filename = output_dir / f"{dataset_name}_node_performance.pdf"
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filename}")

def create_grouped_edge_barplots_by_dataset(all_results, output_dir):
    """
    Create grouped barplots for edge metrics, grouped by dataset.

    Parameters
    ----------
    all_results : dict
        Dictionary with dataset names as keys and metrics dictionaries as values
    output_dir : Path
        Directory to save the plots
    """
    print(f"\n{'='*60}")
    print(f"Creating grouped edge barplots by dataset")
    print('='*60)

    models = ['MAP', 'VI', 'GMVI', 'MLE', 'CCC']
    model_colors = {'MAP': 'blue', 'VI': 'green', 'GMVI': 'orange', 'MLE': 'purple', 'CCC': 'red'}
    statistics = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']

    # Create a figure with subplots in 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Arrange metrics: Top row: R2, rho, KTAU; Bottom row: RMSE, MUE, empty
    metric_positions = {
        'R2': (0, 0),
        'rho': (0, 1),
        'KTAU': (0, 2),
        'RMSE': (1, 0),
        'MUE': (1, 1)
    }

    # Keep track of handles and labels for legend
    legend_handles = None
    legend_labels = None

    for stat in statistics:
        if stat not in metric_positions:
            continue
        ax = axes[metric_positions[stat]]
        
        # Prepare data for grouped bars
        datasets_list = sorted(all_results.keys())
        n_datasets = len(datasets_list)
        n_models = len(models)
        bar_width = 0.2
        
        # Create positions for bars
        dataset_positions = np.arange(n_datasets)
        
        # Plot bars for each model
        for model_idx, model in enumerate(models):
            means = []
            errors_low = []
            errors_high = []

            for dataset in datasets_list:
                metrics_key = 'edge_metrics'
                if (dataset in all_results and
                    metrics_key in all_results[dataset] and
                    model in all_results[dataset][metrics_key] and
                    stat in all_results[dataset][metrics_key][model]):

                    metric = all_results[dataset][metrics_key][model][stat]
                    means.append(metric['mean'])
                    errors_low.append(metric['mean'] - metric['ci_lower'])
                    errors_high.append(metric['ci_upper'] - metric['mean'])
                else:
                    means.append(0)
                    errors_low.append(0)
                    errors_high.append(0)

            # Calculate position for this model's bars
            positions = dataset_positions + (model_idx - n_models/2 + 0.5) * bar_width

            # Plot bars with error bars
            bars = ax.bar(positions, means, bar_width,
                         label=model, color=model_colors[model], alpha=0.8)

            # Add error bars
            ax.errorbar(positions, means,
                       yerr=[errors_low, errors_high],
                       fmt='none', ecolor='black', capsize=3, capthick=1, alpha=0.5)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                if mean != 0:  # Only label non-zero bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.2f}', ha='center', va='bottom', fontsize=16, rotation=90)

        # Capture legend handles and labels after all models have been plotted
        if legend_handles is None and legend_labels is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Customize subplot
        ax.set_xlabel('Dataset', fontsize=20)
        ax.set_ylabel(f'{stat} Value', fontsize=20)
        ax.set_title(f'{stat}', fontsize=22, fontweight='bold')
        ax.set_xticks(dataset_positions)
        ax.set_xticklabels([d.upper() for d in datasets_list], fontsize=16)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Add legend to the empty subplot (bottom right)
    if legend_handles is not None:
        axes[1, 2].axis('off')
        axes[1, 2].legend(legend_handles, legend_labels, loc='center', fontsize=20,
                         frameon=True, title='Models', title_fontsize=22)
    else:
        axes[1, 2].axis('off')

    # Add overall title
    plt.suptitle(f'All Datasets - Edge Metrics Comparison by Dataset (Mean-Centered)',
                fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    filename = output_dir / f"all_datasets_edge_grouped_by_dataset.pdf"
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filename}")

def create_grouped_node_barplots_by_dataset(all_results, output_dir):
    """
    Create grouped barplots for node metrics, grouped by dataset.

    Parameters
    ----------
    all_results : dict
        Dictionary with dataset names as keys and metrics dictionaries as values
    output_dir : Path
        Directory to save the plots
    """
    print(f"\n{'='*60}")
    print(f"Creating grouped node barplots by dataset")
    print('='*60)

    models = ['MAP', 'VI', 'GMVI', 'MLE']
    model_colors = {'MAP': 'blue', 'VI': 'green', 'GMVI': 'orange', 'MLE': 'purple'}
    statistics = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']

    # Create a figure with subplots in 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Arrange metrics: Top row: R2, rho, KTAU; Bottom row: RMSE, MUE, empty
    metric_positions = {
        'R2': (0, 0),
        'rho': (0, 1),
        'KTAU': (0, 2),
        'RMSE': (1, 0),
        'MUE': (1, 1)
    }

    # Keep track of handles and labels for legend
    legend_handles = None
    legend_labels = None

    for stat in statistics:
        if stat not in metric_positions:
            continue
        ax = axes[metric_positions[stat]]

        # Prepare data for grouped bars
        datasets_list = sorted(all_results.keys())
        n_datasets = len(datasets_list)
        n_models = len(models)
        bar_width = 0.2

        # Create positions for bars
        dataset_positions = np.arange(n_datasets)

        # Plot bars for each model
        for model_idx, model in enumerate(models):
            means = []
            errors_low = []
            errors_high = []

            for dataset in datasets_list:
                metrics_key = 'node_metrics'
                if (dataset in all_results and
                    metrics_key in all_results[dataset] and
                    model in all_results[dataset][metrics_key] and
                    stat in all_results[dataset][metrics_key][model]):

                    metric = all_results[dataset][metrics_key][model][stat]
                    means.append(metric['mean'])
                    errors_low.append(metric['mean'] - metric['ci_lower'])
                    errors_high.append(metric['ci_upper'] - metric['mean'])
                else:
                    means.append(0)
                    errors_low.append(0)
                    errors_high.append(0)

            # Calculate position for this model's bars
            positions = dataset_positions + (model_idx - n_models/2 + 0.5) * bar_width

            # Plot bars with error bars
            bars = ax.bar(positions, means, bar_width,
                         label=model, color=model_colors[model], alpha=0.8)

            # Add error bars
            ax.errorbar(positions, means,
                       yerr=[errors_low, errors_high],
                       fmt='none', ecolor='black', capsize=3, capthick=1, alpha=0.5)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                if mean != 0:  # Only label non-zero bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.2f}', ha='center', va='bottom', fontsize=16, rotation=90)

        # Capture legend handles and labels after all models have been plotted
        if legend_handles is None and legend_labels is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Customize subplot
        ax.set_xlabel('Dataset', fontsize=20)
        ax.set_ylabel(f'{stat} Value', fontsize=20)
        ax.set_title(f'{stat}', fontsize=22, fontweight='bold')
        ax.set_xticks(dataset_positions)
        ax.set_xticklabels([d.upper() for d in datasets_list], fontsize=16)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Add legend to the empty subplot (bottom right)
    if legend_handles is not None:
        axes[1, 2].axis('off')
        axes[1, 2].legend(legend_handles, legend_labels, loc='center', fontsize=20,
                         frameon=True, title='Models', title_fontsize=22)
    else:
        axes[1, 2].axis('off')

    # Add overall title
    plt.suptitle(f'All Datasets - Node Metrics Comparison by Dataset (Mean-Centered)',
                fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    filename = output_dir / f"all_datasets_node_grouped_by_dataset.pdf"
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filename}")

def collect_all_metrics(datasets, sampling_time, output_dir):
    """
    Process all datasets and collect metrics for grouped barplots.
    
    Returns
    -------
    dict
        Dictionary with dataset names as keys and metrics as values
    """
    all_results = {}
    
    for dataset_name in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name.upper()} dataset")
            print('='*60)
            
            # Load dataset
            print(f"Loading {dataset_name} dataset with {sampling_time} sampling...")
            benchmark = FEPBenchmarkDataset(cache_dir="~/.maple_cache")
            edge_df, node_df = benchmark.get_dataset(dataset_name, sampling_time)
            dataset = FEPDataset(dataset_nodes=node_df, dataset_edges=edge_df)
            
            print(f"Dataset loaded: {len(node_df)} nodes, {len(edge_df)} edges")
            
            # Train models
            dataset = train_models(dataset, dataset_name)
            
            # Calculate metrics
            results = {
                'dataset': dataset_name,
                'num_nodes': len(dataset.dataset_nodes),
                'num_edges': len(dataset.dataset_edges),
                'node_metrics': {},
                'edge_metrics': {}
            }
            
            statistics = ['RMSE', 'MUE', 'R2', 'rho', 'KTAU']

            # Calculate NODE metrics (mean-centered) - no CCC for nodes
            node_models = ['MAP', 'VI', 'GMVI', 'MLE']
            exp_node_values = dataset.dataset_nodes['Exp. DeltaG'].values
            exp_mean = np.mean(exp_node_values)
            exp_node_centered = exp_node_values - exp_mean

            for model in node_models:
                if model not in dataset.dataset_nodes.columns:
                    continue
                    
                pred_values = dataset.dataset_nodes[model].values
                pred_mean = np.mean(pred_values)
                pred_centered = pred_values - pred_mean
                
                results['node_metrics'][model] = {}
                
                for statistic in statistics:
                    try:
                        result = bootstrap_statistic(
                            y_true=exp_node_centered,
                            y_pred=pred_centered,
                            statistic=statistic,
                            nbootstrap=1000
                        )
                        results['node_metrics'][model][statistic] = {
                            'mean': result['mean'],
                            'ci_lower': result['low'],
                            'ci_upper': result['high'],
                            'std': result['stderr']
                        }
                    except Exception as e:
                        print(f"     Error calculating {model} node {statistic}: {e}")
                        results['node_metrics'][model][statistic] = None
            
            # Calculate EDGE metrics (mean-centered) - includes CCC from dataset
            edge_models = ['MAP', 'VI', 'GMVI', 'MLE', 'CCC']
            exp_edge_values = dataset.dataset_edges['Experimental DeltaDeltaG'].values
            exp_edge_mean = np.mean(exp_edge_values)
            exp_edge_centered = exp_edge_values - exp_edge_mean

            for model in edge_models:
                if model not in dataset.dataset_edges.columns:
                    continue

                pred_values = dataset.dataset_edges[model].values
                pred_mean = np.mean(pred_values)
                pred_centered = pred_values - pred_mean

                results['edge_metrics'][model] = {}

                for statistic in statistics:
                    try:
                        result = bootstrap_statistic(
                            y_true=exp_edge_centered,
                            y_pred=pred_centered,
                            statistic=statistic,
                            nbootstrap=1000
                        )
                        results['edge_metrics'][model][statistic] = {
                            'mean': result['mean'],
                            'ci_lower': result['low'],
                            'ci_upper': result['high'],
                            'std': result['stderr']
                        }
                    except Exception as e:
                        print(f"     Error calculating {model} edge {statistic}: {e}")
                        results['edge_metrics'][model][statistic] = None
            
            all_results[dataset_name] = results
            
            # Also generate individual plots
            create_edge_correlation_plot(dataset, dataset_name, output_dir)
            create_edge_performance_plot(dataset, dataset_name, output_dir)
            create_node_correlation_plot(dataset, dataset_name, output_dir)
            create_node_performance_plot(dataset, dataset_name, output_dir)
            
            print(f"\n✅ {dataset_name.upper()} processing complete!")
            
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results

def process_dataset(dataset_name, sampling_time, output_dir):
    """Process a single dataset: load, train models, and generate plots."""
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} dataset")
        print('='*60)
        
        # Load dataset
        print(f"Loading {dataset_name} dataset with {sampling_time} sampling...")
        benchmark = FEPBenchmarkDataset(cache_dir="~/.maple_cache")
        edge_df, node_df = benchmark.get_dataset(dataset_name, sampling_time)
        dataset = FEPDataset(dataset_nodes=node_df, dataset_edges=edge_df)
        
        print(f"Dataset loaded: {len(node_df)} nodes, {len(edge_df)} edges")
        
        # Train models
        dataset = train_models(dataset, dataset_name)
        
        # Generate plots
        create_edge_correlation_plot(dataset, dataset_name, output_dir)
        create_edge_performance_plot(dataset, dataset_name, output_dir)
        create_node_correlation_plot(dataset, dataset_name, output_dir)
        create_node_performance_plot(dataset, dataset_name, output_dir)
        
        print(f"\n✅ {dataset_name.upper()} processing complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to process all datasets."""
    
    # Datasets to process
    datasets = ['cdk8', 'cmet', 'eg5', 'pfkfb3', 'syk', 'tnks2']
    sampling_time = "20ns"
    
    # Create output directory
    output_dir = Path("benchmark_plots")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Collect metrics from all datasets
    print("\n" + "="*60)
    print("PROCESSING ALL DATASETS")
    print("="*60)
    
    all_results = collect_all_metrics(datasets, sampling_time, output_dir)
    
    # Create grouped barplots
    if all_results:
        print("\n" + "="*60)
        print("CREATING GROUPED COMPARISON PLOTS")
        print("="*60)

        # Create grouped barplots for edge metrics
        create_grouped_edge_barplots_by_dataset(all_results, output_dir)

        # Create grouped barplots for node metrics
        create_grouped_node_barplots_by_dataset(all_results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for dataset_name in datasets:
        if dataset_name in all_results:
            print(f"{dataset_name.upper():10s}: ✅ Success")
        else:
            print(f"{dataset_name.upper():10s}: ❌ Failed")
    
    print(f"\nAll plots saved in: {output_dir.absolute()}")
    print("\nExpected output files:")
    print("\nPer dataset:")
    print("  - <dataset>_edge_correlation.pdf")
    print("  - <dataset>_edge_performance.pdf")
    print("  - <dataset>_node_correlation.pdf")
    print("  - <dataset>_node_performance.pdf")
    print("\nGrouped comparisons:")
    print("  - all_datasets_edge_grouped_by_dataset.pdf")
    print("  - all_datasets_node_grouped_by_dataset.pdf")

if __name__ == "__main__":
    main()