from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import scipy.stats
import sklearn.metrics
from sklearn.metrics import roc_curve
def sensivity_specifity_cutoff(y_true, y_score):
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def _binarize(x, variable_name='values'):
  """Casts a boolean vector to {0, 1} and validates its values."""
  binarized = np.array(x, dtype=np.int32)
  if set(binarized) - set([0, 1]):
    raise ValueError('%s must be in {0, 1}' % variable_name)
  return binarized

def _delong_covariance(y_true, y_scores, sample_weight=None):
  """Estimates the covariance matrix for a set of ROC-AUC scores."""

  y_true = _binarize(y_true, 'true labels')

  if sample_weight is None:
    sample_weight = np.ones_like(y_true, dtype=np.int32)
  else:
    sample_weight = _binarize(sample_weight, 'sample weight')
    if len(sample_weight) != len(y_true):
      raise ValueError()
    elif not sample_weight.sum():
      raise ValueError('No nonzero weights found.')

  y_scores = np.array(y_scores)
  if y_scores.ndim == 1:
    # If there's just one score, add a singleton dimension.
    y_scores = np.expand_dims(y_scores, 1)
  elif y_scores.ndim != 2:
    raise ValueError('Unexpected shape for y_scores: %r' % y_scores.shape)

  num_obs, num_scores = y_scores.shape
  if num_obs != len(y_true):
    raise ValueError('y_true and y_scores must have the same length!')

  y_scores_valid = y_scores[sample_weight == 1, :]
  y_true_valid = y_true[sample_weight == 1]
  num_positives = y_true_valid.sum()
  num_negatives = len(y_true_valid) - num_positives

  point_estimates = []
  d_01s = []
  d_10s = []
  for score_idx in range(num_scores):
    score_pos_neg_matrix = np.array((
        range(len(y_true_valid)),
        y_true_valid,
        1 - y_true_valid,
        y_scores_valid[:, score_idx],
    ),
                                    dtype=np.float64)
    # Positives and negatives are sorted by the score while avoiding bias.
    # Using two ways to break the tie to take the average later.
    # 1. Treat positive larger than negative for tie breaking.
    neg_first_order = score_pos_neg_matrix[:3,
                                           np.lexsort(score_pos_neg_matrix[
                                               1::2, :])]
    # 2. Treat negative larger than positive for tie breaking.
    pos_first_order = score_pos_neg_matrix[:3, np.lexsort(score_pos_neg_matrix)]

    # Up to each point, how many positives and negatives are there.
    cumsum_neg_first_order = neg_first_order[1:].cumsum(axis=1)
    cumsum_pos_first_order = pos_first_order[1:].cumsum(axis=1)

    # For each positive, how many negatives are equal or smaller .
    le_neg_count = cumsum_neg_first_order[1, neg_first_order[1, :] > 0]
    # For each positive, how many negatives are smaller.
    lt_neg_count = cumsum_pos_first_order[1, pos_first_order[1, :] > 0]
    # For each negative, how many positives are equal or greater.
    ge_pos_count = num_positives - cumsum_neg_first_order[
        0, neg_first_order[2, :] > 0]
    # For each negative, how many positives are greater.
    gt_pos_count = num_positives - cumsum_pos_first_order[
        0, pos_first_order[2, :] > 0]

    # Taking the average of the two count methods.
    d01 = (le_neg_count + lt_neg_count) / 2 / num_negatives
    d10 = (ge_pos_count + gt_pos_count) / 2 / num_positives

    # Sorting by index to restore original order.
    d01 = d01[np.argsort(neg_first_order[0, neg_first_order[1, :] > 0])]
    d10 = d10[np.argsort(neg_first_order[0, neg_first_order[2, :] > 0])]

    # Equivalent to sklearn.metrics.roc_auc_score(y_true, y_score).
    point_estimates.append(d01.mean())

    # The notation d_01 and d_10 comes from [2]; see docstring.
    # For each positive score, the fraction of negatives that are smaller.
    d_01s.append(d01)

    # For each negative score, the fraction of positives that are larger.
    d_10s.append(d10)

  s_01 = np.cov(d_01s, ddof=1)
  s_10 = np.cov(d_10s, ddof=1)
  covariance_matrix = s_01 / num_positives + s_10 / num_negatives

  return point_estimates, covariance_matrix

def delong_interval(y_true, y_score, sample_weight=None, coverage=0.95):
  """Computes a confidence interval on the AUC-ROC using DeLong's method.
  See [1] for the original formulation, and [2] for discussion/simulation.
  [1] DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or
  more correlated receiver operating characteristic cuvers: a nonparametric
  approach. Biometrics. 1988;44: 837-845.
  [2] Gensheng Qin, Hotilovac L. Comparison of non-parametric confidence
  intervals for the area under the ROC curve of a continuous-score diagnostic
  test. Stat Methods Med Res. 2008;17: 207-221.
  Args:
    y_true: An array of boolean outcomes.
    y_score: An array of continuous-valued scores. Ordinal values are
      acceptable.
    sample_weight: An optional mask of binary sample weights. Must have nonzero
      entries.
    coverage: The size of the confidence interval. Should be in (0, 1]. The
      default is 95%.
  Returns:
    (lower, upper) the endpoints of an equitailed confidence interval for
    the area under the ROC curve.
  Raises:
    ValueError: if inputs are invalid.
  """
  if coverage > 1.0 or coverage <= 0.0:
    raise ValueError('coverage level must be in (0, 1]')

  point_estimates, covariance_matrix = _delong_covariance(
      y_true, y_score, sample_weight=sample_weight)

  point_estimate = point_estimates[0]
  variance = float(covariance_matrix)

  standard_error = np.sqrt(variance)
  z = scipy.stats.norm.isf((1 - coverage) / 2.0)
  lower = max(point_estimate - z * standard_error, 0.0)
  upper = min(point_estimate + z * standard_error, 1.0)
  return (lower, upper)