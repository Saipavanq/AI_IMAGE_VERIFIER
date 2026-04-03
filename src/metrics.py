from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    tn: int
    fp: int
    fn: int
    tp: int


def compute_binary_metrics(y_true: np.ndarray, y_score_fake: np.ndarray, threshold: float = 0.5) -> BinaryMetrics:
    """Compute metrics for label 1 = FAKE (AI-generated), label 0 = REAL."""
    y_true = np.asarray(y_true).astype(int)
    y_score_fake = np.asarray(y_score_fake).astype(float)

    y_pred = (y_score_fake >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    roc_auc: float | None
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score_fake, pos_label=1)
        roc_auc = float(auc(fpr, tpr)) if len(np.unique(y_true)) > 1 else None
    except ValueError:
        roc_auc = None

    pr_auc: float | None
    try:
        pr_auc = float(average_precision_score(y_true, y_score_fake))
    except ValueError:
        pr_auc = None

    return BinaryMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
