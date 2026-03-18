from collections import defaultdict
import numpy as np

from sklearn import metrics
from model.bedroc import bedroc_score


def metric(y_true, y_pred, y_prob, empty=-1):
    '''
    for classification evaluation on single task
    :param y_true: 1-D, e.g. [1, 0, 1, 1]
    :param y_pred: 1-D, e.g. [0, 0, 1, 1]
    :param y_prob: 1-D, e.g. [0.7, 0.5, 0.2, 0.7]
    :return:
    '''
    assert len(y_true) == len(y_pred) == len(y_prob)
    y_true, y_pred, y_prob = np.array(y_true).flatten(), np.array(y_pred).flatten(), np.array(y_prob).flatten()
    # filter empty data
    flag = y_true != empty
    y_true, y_pred, y_prob = y_true[flag], y_pred[flag], y_prob[flag]

    auc = metrics.roc_auc_score(y_true, y_prob)
    bedroc = bedroc_score(y_true, y_prob, alpha=20.0, empty=empty)
    precision_list, recall_list, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall_list, precision_list)
    matthews = metrics.matthews_corrcoef(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)


    return {"ROCAUC": auc, "BEDROC": bedroc, "AUPR": aupr, "Matthews": matthews, "F1": f1}


# Function to compute top-k precision and F1 for binary classification
def compute_topk_precision_f1(probs, labels, k=15):
    # probs: numpy array of predicted probabilities for positive class
    # labels: numpy array of true labels (0 or 1)
    idx_sorted = np.argsort(-probs)[:k]
    topk_labels = labels[idx_sorted]
    tp = np.sum(topk_labels == 1)
    fp = k - tp
    fn = np.sum(labels == 1) - tp
    precision = tp / k if k > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    return precision, f1


# function to compute top-k hit rate for binary classification
def compute_topk_hit_rate(probs, labels, k=15):
    # probs: numpy array of predicted probabilities for positive class
    # labels: numpy array of true labels (0 or 1)
    idx_sorted = np.argsort(-probs)[:k]
    topk_labels = labels[idx_sorted]
    hit_rate = np.sum(topk_labels == 1) / k if k > 0 else 0.0
    return hit_rate

def metric_reg(y_true, y_pred):
    '''
    for regression evaluation on single task
    :param y_true: 1-D, e.g. [1.1, 0.2, 1.5, 3.2]
    :param y_pred: 1-D, e.g. [-0.2, 1.1, 1.2, 3.1]
    :return:
    '''
    assert len(y_true) == len(y_pred)
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


def metric_multitask(y_true, y_pred, y_prob, num_tasks, empty=-1):
    '''
    :param y_true: ndarray, shape is [batch, num_tasks]
    :param y_pred: ndarray, shape is [batch, num_tasks]
    :param y_prob: ndarray, shape is [batch, num_tasks]
    :return:
    '''
    assert num_tasks == y_true.shape[1] == y_pred.shape[1] == y_prob.shape[1]
    assert y_prob.min() >= 0 and y_prob.max() <= 1

    result_list_dict_each_task = []

    cur_num_tasks = 0
    for i in range(num_tasks):
        flag = y_true[:, i] != empty
        if len(set(y_true[flag, i].flatten())) == 1:  # labels are all one value
            result_list_dict_each_task.append(None)
        else:
            result_list_dict_each_task.append(
                metric(y_true[flag, i].flatten(), y_pred[flag, i].flatten(), y_prob[flag, i].flatten()))
            cur_num_tasks += 1

    mean_performance = defaultdict(float)

    for i in range(num_tasks):
        if result_list_dict_each_task[i] is None:
            continue
        for key in result_list_dict_each_task[i].keys():
            if key in ["fpr", "tpr", "precision_list", "recall_list"]:
                continue
            mean_performance[key] += result_list_dict_each_task[i][key] / cur_num_tasks

    # Add mean BEDROC if present
    if any(r is not None and "BEDROC" in r for r in result_list_dict_each_task):
        mean_performance["mean_BEDROC"] = np.mean([
            r["BEDROC"] for r in result_list_dict_each_task if r is not None and r["BEDROC"] is not None
        ])

    mean_performance["result_list_dict_each_task"] = result_list_dict_each_task

    if cur_num_tasks < num_tasks:
        print("Some target is missing! Missing ratio: {:.2f} [{}/{}]".format(1 - float(cur_num_tasks) / num_tasks,
                                                                             cur_num_tasks, num_tasks))
        mean_performance["some_target_missing"] = "{:.2f} [{}/{}]".format(1 - float(cur_num_tasks) / num_tasks,
                                                                          cur_num_tasks, num_tasks)

    return mean_performance


def metric_reg_multitask(y_true, y_pred, num_tasks):
    '''
    :param y_true: 2-D, shape: (num_samples, num_tasks),  e.g. [[1.1, 1.2], [0.2, 0.6], [1.5, 1.2], [3.2, 1.1]]
    :param y_pred: 2-D, shape: (num_samples, num_tasks), e.g. [[-0.2, 1.5], [1.1, -0.2], [1.2, 1.8], [3.1, 1.0]]
    :return:
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert num_tasks == y_true.shape[1] == y_pred.shape[1]
    assert len(y_true) == len(y_pred)

    result_list_dict_each_task = []
    for i in range(num_tasks):
        result_list_dict_each_task.append(metric_reg(y_true[:, i].flatten(), y_pred[:, i].flatten()))

    mean_performance = defaultdict(float)
    for i in range(num_tasks):
        for key in result_list_dict_each_task[i].keys():
            mean_performance[key] += result_list_dict_each_task[i][key] / num_tasks

    mean_performance["result_list_dict_each_task"] = result_list_dict_each_task

    return mean_performance

