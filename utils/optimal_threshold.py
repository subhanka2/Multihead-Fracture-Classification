from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay as confusion_matrix
from statsmodels.stats import proportion
from sklearn import metrics
import numpy as np

def _threshold_finder(y_true, y_predict_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba)
    auc = roc_auc_score(y_true, y_predict_proba)
    precision, recall, thresholds2 = precision_recall_curve(y_true, y_predict_proba)
    
    class_names = [0, 1]
    youden_idx = np.argmax(np.abs(tpr - fpr))
    youden_threshold = thresholds[youden_idx]
    y_pred_youden = (y_predict_proba > youden_threshold).astype(int)
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred_youden)
    np.set_printoptions(precision=2)
    
    f1 = []
    for i in range(len(precision)):
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
        
    queue_rate = []
    for thr in thresholds2:
        queue_rate.append((y_predict_proba >= thr).mean()) 
    return youden_threshold

def print_conf(Y, Y_pred, thresh, verbose=True):
    conf = metrics.confusion_matrix(Y, Y_pred >= thresh, labels=[0, 1])
    tn, fp, fn, tp = conf.flatten()
    tpr = tp / (tp + fn)
    tpr_interval = proportion.proportion_confint(tp, (tp + fn), method='beta')
    spc = tn / (tn + fp)
    spc_interval = proportion.proportion_confint(tn, (tn + fp), method='beta')
    ppv = tp / (tp + fp)
    ppv_interval = proportion.proportion_confint(tp, (tp + fp), method='beta')

    acc = (tp + tn) / conf.sum()
    return thresh


def find_nearest(array, value):
    val = np.min(array[array >= value])
    idx = np.abs(array - val).argmin()
    return idx


def print_measures(Y, Y_pred):
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)

    Y_score = Y_pred
    fpr, tpr, thresh = metrics.roc_curve(Y, Y_score)
    #print("threshold:", thresh)
    tnr = 1 - fpr

    auc = metrics.roc_auc_score(Y, Y_score)
    ap = metrics.average_precision_score(Y, Y_score)
    auc_low, auc_hi = auc_ci(Y, Y_score)

    pt3 = find_nearest(tpr - tnr, 0)
    thresold_value = print_conf(Y, Y_pred, thresh[pt3])

    return thresold_value

def auc_ci(Y, Y_pred):
    A = metrics.roc_auc_score(Y, Y_pred)
    m = np.sum(Y == 1)
    n = np.sum(Y == 0)
    assert m + n == Y_pred.size

    P_xxy = A / (2 - A)
    P_xyy = 2 * (A ** 2) / (1 + A)
    SD = (A * (1 - A) + (m - 1) * (P_xxy - A**2) +
          (n - 1) * (P_xyy - A**2)) / (m * n)
    SE = np.sqrt(SD)

    A1 = A - 1.96 * SE
    A2 = A + 1.96 * SE
    return A1, A2