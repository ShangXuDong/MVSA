# https://github.com/storyandwine/LAGCN/blob/master/code/clac_metric.py

from sklearn.metrics import precision_recall_curve, roc_curve, auc, balanced_accuracy_score


def overall_acc(prediction, gt):
    """
    :param prediction: binary
    :param gt: torch.Tensor binary ground truth
    :return: overall balanced accuracy
    """
    prediction = prediction.cpu().data.numpy()
    gt = gt.cpu().data.numpy()
    acc = balanced_accuracy_score(gt.flatten(), prediction.flatten())

    return acc



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


import numpy as np
from sklearn import metrics

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


def evaluate(predict, label, is_final=False):
    if not is_final:
        res = get_metrics(real_score=label, predict_score=predict)
    else:
        res = [None]*7
    aupr = metrics.average_precision_score(y_true=label, y_score=predict)
    auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
    result = {"aupr":aupr,
              "auroc":auroc,
              "lagcn_aupr":res[0],
              "lagcn_auc":res[1],
              "lagcn_f1_score":res[2],
              "lagcn_accuracy":res[3],
              "lagcn_recall":res[4],
              "lagcn_specificity":res[5],
              "lagcn_precision":res[6]}
    return result