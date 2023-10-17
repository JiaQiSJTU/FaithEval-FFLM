import json
import numpy as np
from utils import *
import argparse

def inconsistency_detection_eval(file_path):
    val_delta_1 = []
    val_delta_2 = []
    val_delta_3 = []
    val_labels = []

    test_delta_1 = []
    test_delta_2 = []
    test_delta_3 = []
    test_labels = []

    with open(file_path, "r") as f:
        for line in f:
            line = json.loads(line.strip())
            s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss = score_calculation(line)

            score_1 = np.mean( np.exp((1 - s2s)) * (lm_loss - s2s_loss) )
            score_2 = np.mean(  np.exp((1 - s2s)) * (prefix_loss - s2s_loss) )
            score_3 = np.mean(np.exp((1 - s2s_doc)) *(lm_loss_doc - s2s_loss_doc))

            if line["cut"] == "val":
                val_labels.append(line["label"])
                val_delta_1.append(score_1)
                val_delta_2.append(score_2)
                val_delta_3.append(score_3)
            else:
                test_labels.append(line["label"])
                test_delta_1.append(score_1)
                test_delta_2.append(score_2)
                test_delta_3.append(score_3)

    all_best_threshold = 0.0
    all_best_f1 = 0.0
    all_test = None
    all_test_labels = None

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        beta_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][:int((1-alpha)*10)+1]
        for beta in beta_range:
            val_delta = alpha * np.array(val_delta_1) + beta * np.array(val_delta_3) + (1-alpha-beta) * np.array(val_delta_2)

            best_threshold, best_f1 = choose_best_threshold(val_labels, val_delta)
            if best_f1 > all_best_f1:
                print(alpha, beta)
                all_best_threshold = best_threshold
                all_best_f1 = best_f1
                all_test = alpha * np.array(test_delta_1) + beta * np.array(test_delta_3) +  (1-alpha-beta) * np.array(test_delta_2)
                all_test_labels = test_labels

    get_metrics(all_test, all_test_labels, 1, is_balanced_acc=True, threshold=all_best_threshold)

def faithfulness_rating_eval(file_path):

    # cop = []
    # harim = []
    fflm = []
    labels = []

    full_score = {"summeval": 5.0, "qagscnn": 1.0, "qagsxsum": 1.0, "frankcnn":1.0, "frankxsum":1.0}

    with open(file_path, "r") as f:
        for line in f:
            line = json.loads(line.strip())
            s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss = score_calculation(line)

            score_1 = np.mean((lm_loss - s2s_loss) * np.exp((1-s2s)))
            score_2 = np.mean((prefix_loss - s2s_loss) * np.exp((1-s2s)))
            score_3 = np.mean((lm_loss_doc - s2s_loss_doc) * np.exp(1-s2s_doc))

            # harim.append(np.mean(-(1-s2s)*(1-(s2s-lm))))
            # cop.append(np.mean(prefix_loss)-np.mean(s2s_loss))
            fflm.append(0.25 * score_1 + 0.5 * score_2 + 0.25 * score_3)

            labels.append(line["score"])

    print("===fflm===")
    get_metrics(fflm,labels, full_score=full_score["qagscnn"], is_correlation=True)
    # print("===Cop===")
    # get_metrics(cop, labels, full_score=full_score["qagscnn"], is_correlation=True)
    # print("===HaRiM===")
    # get_metrics(harim, labels, full_score=full_score["qagscnn"], is_correlation=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="")
    args = parser.parse_args()

    if "summac" in args.file_path:
        inconsistency_detection_eval(args.file_path)
    else:
        faithfulness_rating_eval(args.file_path)
