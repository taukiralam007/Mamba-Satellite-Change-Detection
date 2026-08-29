import numpy as np


def trfl_assessment(gt, pred):

    gt = np.asarray(gt).reshape(-1).astype(int)
    pred = np.asarray(pred).reshape(-1).astype(int)

    cm = np.zeros((4, 4), dtype=np.int64)

    for p in range(4):
        for t in range(4):
            cm[p, t] = np.sum(
                (pred == p) &
                (gt == t)
            )

    n = cm.sum()
    PA = np.trace(cm)

    OA = PA / (n + 1e-8)

    NPJ = cm.sum(axis=0)
    NIP = cm.sum(axis=1)

    PE = np.dot(NPJ, NIP)

    kappa_den = n*n - PE

    kappa = (
        (n*PA - PE) / kappa_den
        if abs(kappa_den) > 1e-8
        else 0.0
    )

    change_correct = (
        PA
        - cm[0, 0]
        - cm[3, 3]
    )

    true_change = (
        cm[:, 1].sum()
        + cm[:, 2].sum()
    )

    predicted_change = (
        cm[1, :].sum()
        + cm[2, :].sum()
    )

    precision_c = (
        change_correct /
        (predicted_change + 1e-8)
    )

    recall_c = (
        change_correct /
        (true_change + 1e-8)
    )

    f1_c = (
        2 * precision_c * recall_c /
        (precision_c + recall_c + 1e-8)
    )

    iou_den = (
        predicted_change
        + true_change
        - change_correct
        - cm[1, 2]
        - cm[2, 1]
    )

    iou_c = (
        change_correct /
        (iou_den + 1e-8)
    )

    return {
        "OA": OA * 100,
        "Precision_c": precision_c * 100,
        "Recall_c": recall_c * 100,
        "F1_c": f1_c * 100,
        "IoU_c": iou_c * 100,
        "Kappa": kappa * 100,
        "ConfusionMatrix": cm
    }
