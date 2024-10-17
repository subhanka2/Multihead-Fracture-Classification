#http://araw.mede.uic.edu/cgi-bin/testcalc.pl?DT=76&Dt=52&dT=1&dt=38&2x2=Compute
from __future__ import print_function, division
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
from math import sqrt
from scipy.special import ndtri

def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.
    
    Follows notation described on pages 46--47 of [1]. 
    
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    """

    A = 2*r + z**2
    B = z*sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)

def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (AUC, lower, upper)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)


def pred_formu(data_frame, threshold):
    pred_list = []
    for i in range(len(data_frame)):
        if data_frame[i]>= threshold:
            pred_list.append(1)
        else:
            pred_list.append(0)
    return pred_list


def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    """Compute confidence intervals for sensitivity and specificity using Wilson's method. 
    
    This method does not rely on a normal approximation and results in accurate 
    confidence intervals even for small sample sizes.
    
    Parameters
    ----------
    TP : int
        Number of true positives
    FP : int 
        Number of false positives
    FN : int
        Number of false negatives
    TN : int
        Number of true negatives
    alpha : float, optional
        Desired confidence. Defaults to 0.95, which yields a 95% confidence interval. 
    
    Returns
    -------
    sensitivity_point_estimate : float
        Numerical estimate of the test sensitivity
    specificity_point_estimate : float
        Numerical estimate of the test specificity
    sensitivity_confidence_interval : Tuple (float, float)
        Lower and upper bounds on the alpha confidence interval for sensitivity
    specificity_confidence_interval
        Lower and upper bounds on the alpha confidence interval for specificity 
        
    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    [2] E. B. Wilson, Probable inference, the law of succession, and statistical inference,
    J Am Stat Assoc 22:209-12, 1927. 
    """
    z = -ndtri((1.0-alpha)/2)
    sensitivity_point_estimate = TP/(TP + FN)
    ppv_point_estimate = TP/(TP + FP)
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)
    ppv_confidence_interval = _proportion_confidence_interval(TP, TP + FP, z)
    specificity_point_estimate = TN/(TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)
    npv_point_estimate = TN/(TN + FN)
    npv_confidence_interval = _proportion_confidence_interval(TN, TN + FN, z)
    return sensitivity_point_estimate, specificity_point_estimate, sensitivity_confidence_interval,specificity_confidence_interval, ppv_point_estimate, ppv_confidence_interval, npv_point_estimate, npv_confidence_interval