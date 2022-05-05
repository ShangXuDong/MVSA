from sklearn.metrics import precision_recall_curve, roc_curve, auc, balanced_accuracy_score

def auroc(prob, gt):
    """
    :param prob: torch.Tensor probability estimates of the positive class
    :param gt: torch.Tensor binary ground truth
    :return: area under ROC curve
    """
    y_true = gt.cpu().data.numpy().flatten()
    y_scores = prob.cpu().data.numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc_score = auc(fpr, tpr)
    return auroc_score

def auprc(prob, gt):
    """
    :param prob: torch.Tensor probability estimates of the positive class
    :param gt: torch.Tensor binary ground truth
    :return: area under PR curve
    """
    y_true = gt.cpu().data.numpy().flatten()
    y_scores = prob.cpu().data.numpy().flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc_score = auc(recall, precision)
    return auprc_score
