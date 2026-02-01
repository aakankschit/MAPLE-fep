from typing import Union

import numpy as np
import scipy
import sklearn.metrics

"""
This module was taken from Cinnabar and modified to fit the needs of MAPLE.

Cinnabar: 

"""


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values

    Returns
    -------
    float
        Mean Absolute Error
    """
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return sklearn.metrics.mean_absolute_error(y_true[mask], y_pred[mask])


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE).

    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values

    Returns
    -------
    float
        Root Mean Square Error
    """
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true[mask], y_pred[mask]))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).

    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values

    Returns
    -------
    float
        R² value
    """
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            y_true_clean, y_pred_clean
        )
        return r_value**2
    except ValueError:
        # Handle case where all values are identical
        if np.allclose(y_true_clean, y_pred_clean):
            return 1.0  # Perfect prediction
        elif np.var(y_pred_clean) == 0 or np.var(y_true_clean) == 0:
            return np.nan  # Undefined when either variable has no variance
        else:
            return np.nan  # Other undefined cases


def calculate_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient.

    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values

    Returns
    -------
    float
        Pearson correlation coefficient
    """
    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    try:
        return scipy.stats.pearsonr(y_true_clean, y_pred_clean)[0]
    except (ValueError, RuntimeWarning):
        # Handle case where correlation is not defined
        return np.nan


def bootstrap_statistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dy_true: Union[np.ndarray, None] = None,
    dy_pred: Union[np.ndarray, None] = None,
    ci: float = 0.95,
    statistic: str = "RMSE",
    nbootstrap: int = 1000,
    include_true_uncertainty: bool = False,
    include_pred_uncertainty: bool = False,
) -> dict:
    """Compute mean and confidence intervals of specified statistic.

    This function performs bootstrap resampling to compute robust statistics
    with confidence intervals, accounting for measurement uncertainties.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values (experimental data)
    y_pred : ndarray with shape (N,)
        Predicted values (calculated data)
    dy_true : ndarray with shape (N,) or None
        Errors of true values. If None, the values are assumed to have no errors
    dy_pred : ndarray with shape (N,) or None
        Errors of predicted values. If None, the values are assumed to have no errors
    ci : float, optional, default=0.95
        Confidence interval level (e.g., 0.95 for 95% CI)
    statistic : str
        Statistic to compute, one of ['RMSE', 'MUE', 'R2', 'rho', 'KTAU', 'RAE']
    nbootstrap : int, optional, default=1000
        Number of bootstrap samples
    include_true_uncertainty : bool, default False
        Whether to account for the uncertainty in y_true when bootstrapping
    include_pred_uncertainty : bool, default False
        Whether to account for the uncertainty in y_pred when bootstrapping

    Returns
    -------
    stats_dict : dict
        Dictionary containing:
        - 'mle': Maximum likelihood estimate of the statistic
        - 'mean': Mean of bootstrapped samples
        - 'stderr': Standard error
        - 'low': Lower bound of confidence interval
        - 'high': Upper bound of confidence interval

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    >>> stats = bootstrap_statistic(y_true, y_pred, statistic="RMSE")
    >>> print(f"RMSE: {stats['mle']:.3f} ± {stats['stderr']:.3f}")
    """

    def compute_statistic(
        y_true_sample: np.ndarray, y_pred_sample: np.ndarray, statistic: str
    ):
        """Compute requested statistic for a given sample.

        Parameters
        ----------
        y_true_sample : ndarray with shape (N,)
            True values for current bootstrap sample
        y_pred_sample : ndarray with shape (N,)
            Predicted values for current bootstrap sample
        statistic : str
            Statistic to compute, one of ['RMSE', 'MUE', 'R2', 'rho', 'RAE', 'KTAU']

        Returns
        -------
        float
            Computed statistic value
        """

        def calc_RAE(y_true_sample: np.ndarray, y_pred_sample: np.ndarray):
            """Calculate Relative Absolute Error (RAE)."""
            MAE = sklearn.metrics.mean_absolute_error(y_true_sample, y_pred_sample)
            mean = np.mean(y_true_sample)
            MAD = np.sum([np.abs(mean - i) for i in y_true_sample]) / float(
                len(y_true_sample)
            )
            return MAE / MAD

        if statistic == "RMSE":
            return np.sqrt(
                sklearn.metrics.mean_squared_error(y_true_sample, y_pred_sample)
            )
        elif statistic == "MUE":
            return sklearn.metrics.mean_absolute_error(y_true_sample, y_pred_sample)
        elif statistic == "R2":
            try:
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                    y_true_sample, y_pred_sample
                )
                return r_value**2
            except ValueError:
                # Handle case where all values are identical
                return 1.0 if np.allclose(y_true_sample, y_pred_sample) else 0.0
        elif statistic == "rho":
            try:
                return scipy.stats.pearsonr(y_true_sample, y_pred_sample)[0]
            except (ValueError, RuntimeWarning):
                return np.nan
        elif statistic == "RAE":
            return calc_RAE(y_true_sample, y_pred_sample)
        elif statistic == "KTAU":
            try:
                return scipy.stats.kendalltau(y_true_sample, y_pred_sample)[0]
            except (ValueError, RuntimeWarning):
                return np.nan
        else:
            raise ValueError(
                f"Unknown statistic '{statistic}'. Supported: ['RMSE', 'MUE', 'R2', 'rho', 'KTAU', 'RAE']"
            )

    # Handle None error arrays
    if dy_true is None:
        dy_true = np.zeros_like(y_true)
    if dy_pred is None:
        dy_pred = np.zeros_like(y_pred)

    # Input validation
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
    assert len(y_true) == len(dy_true), "y_true and dy_true must have same length"
    assert len(y_true) == len(dy_pred), "y_pred and dy_pred must have same length"

    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        # Return NaN results if no valid data
        return {
            "mle": np.nan,
            "mean": np.nan,
            "stderr": np.nan,
            "low": np.nan,
            "high": np.nan,
        }

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    dy_true_clean = dy_true[mask]
    dy_pred_clean = dy_pred[mask]

    sample_size = len(y_true_clean)
    s_n = np.zeros([nbootstrap], np.float64)  # Statistics for each bootstrap sample

    # Perform bootstrap resampling
    for replicate in range(nbootstrap):
        y_true_sample = np.zeros_like(y_true_clean)
        y_pred_sample = np.zeros_like(y_pred_clean)

        # Resample with replacement
        for i, j in enumerate(
            np.random.choice(np.arange(sample_size), size=[sample_size], replace=True)
        ):
            # Add uncertainty if requested
            stddev_true = np.fabs(dy_true_clean[j]) if include_true_uncertainty else 0
            stddev_pred = np.fabs(dy_pred_clean[j]) if include_pred_uncertainty else 0

            y_true_sample[i] = np.random.normal(
                loc=y_true_clean[j], scale=stddev_true, size=1
            )[0]
            y_pred_sample[i] = np.random.normal(
                loc=y_pred_clean[j], scale=stddev_pred, size=1
            )[0]

        s_n[replicate] = compute_statistic(y_true_sample, y_pred_sample, statistic)

    # Compute final statistics
    stats_dict = dict()
    stats_dict["mle"] = compute_statistic(
        y_true_clean, y_pred_clean, statistic
    )  # Maximum likelihood estimate
    stats_dict["stderr"] = np.std(s_n)
    stats_dict["mean"] = np.mean(s_n)

    # Compute confidence intervals
    s_n = np.sort(s_n)
    low_frac = (1.0 - ci) / 2.0
    high_frac = 1.0 - low_frac
    stats_dict["low"] = s_n[int(np.floor(nbootstrap * low_frac))]
    stats_dict["high"] = s_n[int(np.ceil(nbootstrap * high_frac))]

    return stats_dict


def compute_simple_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute basic statistics without bootstrapping.

    This function provides a quick way to compute common statistics
    without the computational overhead of bootstrapping.

    Parameters
    ----------
    y_true : ndarray with shape (N,)
        True values (experimental data)
    y_pred : ndarray with shape (N,)
        Predicted values (calculated data)

    Returns
    -------
    dict
        Dictionary containing basic statistics:
        - 'RMSE': Root Mean Square Error
        - 'MUE': Mean Absolute Error
        - 'R2': Coefficient of determination
        - 'rho': Pearson correlation coefficient
        - 'KTAU': Kendall's tau correlation coefficient
    """

    # Input validation
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"

    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return {
            "RMSE": np.nan,
            "MUE": np.nan,
            "R2": np.nan,
            "rho": np.nan,
            "KTAU": np.nan,
        }

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    stats = {}

    # RMSE
    stats["RMSE"] = np.sqrt(
        sklearn.metrics.mean_squared_error(y_true_clean, y_pred_clean)
    )

    # MUE (Mean Absolute Error)
    stats["MUE"] = sklearn.metrics.mean_absolute_error(y_true_clean, y_pred_clean)

    # R²
    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            y_true_clean, y_pred_clean
        )
        stats["R2"] = r_value**2
    except ValueError:
        stats["R2"] = 1.0 if np.allclose(y_true_clean, y_pred_clean) else 0.0

    # Pearson correlation
    try:
        stats["rho"] = scipy.stats.pearsonr(y_true_clean, y_pred_clean)[0]
    except (ValueError, RuntimeWarning):
        stats["rho"] = np.nan

    # Kendall's tau
    try:
        stats["KTAU"] = scipy.stats.kendalltau(y_true_clean, y_pred_clean)[0]
    except (ValueError, RuntimeWarning):
        stats["KTAU"] = np.nan

    return stats
