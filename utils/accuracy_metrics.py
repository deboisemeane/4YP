import numpy as np


def accuracy_metrics(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  # False Positives
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)  # False Negatives
    TP = np.diag(confusion_matrix)  # True Positives
    TN = confusion_matrix.sum() - (FP + FN + TP)  # True Negatives

    # Class-specific metrics
    TPR = TP / (TP + FN)  # True Positive Rate
    TNR = TN / (TN + FP)  # True Negative Rate
    PPV = TP / (TP + FP)  # Positive Predictive Value
    NPV = TN / (TN + FN)  # Negative Predictive Value
    FPR = FP / (FP + TN)  # False Positive Rate
    FNR = FN / (TP + FN)  # False Negative Rate
    FDR = FP / (TP + FP)  # False Discovery Rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # Accuracy

    # Total metrics (weighted averages)
    total_instances = confusion_matrix.sum()
    weights_for_ppv = confusion_matrix.sum(axis=0) / total_instances  # Weights for PPV based on predicted class sizes
    weights_for_npv = (confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)) / total_instances  # Weights for NPV based on negative predictions
    weights_for_other_metrics = confusion_matrix.sum(axis=1) / total_instances  # Weights for other metrics based on actual class sizes
    total_TPR = np.sum(TPR * weights_for_other_metrics)
    total_PPV = np.sum(PPV * weights_for_ppv)
    total_NPV = np.sum(NPV * weights_for_npv)
    total_FDR = np.sum(FDR * weights_for_ppv)
    total_ACC = np.sum(ACC * weights_for_other_metrics)

    # Total Cohen's Kappa
    Po = np.sum(TP) / total_instances
    Pe = np.sum(confusion_matrix.sum(axis=0) * confusion_matrix.sum(axis=1)) / total_instances**2
    total_Kappa = (Po - Pe) / (1 - Pe)

    return {"TPR": TPR, "TNR": TNR, "PPV": PPV, "NPV": NPV, "FPR": FPR, "FNR": FNR, "FDR": FDR, "ACC": ACC,
            "Total TPR": total_TPR, "Total PPV": total_PPV, "Total NPV": total_NPV, "Total FDR": total_FDR,
            "Total ACC": total_ACC,
            "Total Kappa": total_Kappa}
