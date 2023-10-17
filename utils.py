# encoding = "utf-8"
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn import metrics
import numpy as np

def choose_best_threshold(labels, scores):
    '''following https://github.com/tingofurro/summac'''

    best_f1 = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 0.2)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        f1_score = metrics.balanced_accuracy_score(labels, preds)
        if f1_score >= best_f1:
            best_f1 = f1_score
            best_thresh = thresh
    return best_thresh, best_f1


def get_metrics(predicts, labels, full_score, is_correlation=False, is_balanced_acc=False, threshold=None):
    if is_correlation:
        pearson, _ = pearsonr(predicts, labels)
        print("pearson", pearson)
        spearman, _ = spearmanr(predicts, labels)
        print("spearman", spearman)
        kendall, _ = kendalltau(predicts, labels)
        print("kendall", kendall)

    if is_balanced_acc:
        labels = [0 if x!=full_score else 1 for x in labels]
        predicts = [1 if x>threshold else 0 for x in predicts]
        # balanced Acc
        b_acc = metrics.balanced_accuracy_score(y_true=labels, y_pred=predicts)
        print("balanced-accuracy", b_acc)


def score_calculation(content):
    s2s = content["s2s_tok_list"] if type(content["s2s_tok_list"]) == list else [content["s2s_tok_list"]]
    s2s_doc = content["s2s_tok_list_1"] if type(content["s2s_tok_list_1"]) == list else [content["s2s_tok_list_1"]]
    lm = content["lm_tok_list"] if type(content["lm_tok_list"]) == list else [content["lm_tok_list"]]
    lm_doc = content["lm_tok_list_1"] if type(content["lm_tok_list_1"]) == list else [content["lm_tok_list_1"]]
    prefix = content["prefix_tok_list"] if type(content["prefix_tok_list"]) == list else [content["prefix_tok_list"]]

    s2s = np.array([x if x != 0.0 else float(1e-6) for x in s2s])
    s2s_doc = np.array([x if x != 0.0 else float(1e-6) for x in s2s_doc])
    lm = np.array([x if x != 0.0 else float(1e-6) for x in lm])
    lm_doc = np.array([x if x != 0.0 else float(1e-6) for x in lm_doc])
    prefix = np.array([x if x != 0.0 else float(1e-6) for x in prefix])

    s2s_loss = - np.log(s2s)
    s2s_loss_doc = - np.log(s2s_doc)
    lm_loss = -np.log(lm)
    lm_loss_doc = -np.log(lm_doc)
    prefix_loss = -np.log(prefix)

    return s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss
