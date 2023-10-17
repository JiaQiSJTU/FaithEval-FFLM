# encoding = "utf-8"
import argparse
import json
from scipy.stats import pearsonr, kendalltau, spearmanr
import numpy
from utils import score_calculation


def summeval(file_path):
    from collections import defaultdict
    scores = defaultdict(list)
    predicts = defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            data_dict = json.loads(line.strip())
            model_name = data_dict["model_id"]

            s2s, s2s_doc, lm, lm_doc, prefix, s2s_loss, s2s_loss_doc, lm_loss, lm_loss_doc, prefix_loss = score_calculation(data_dict)

            score_1 = numpy.mean(numpy.exp((1 - s2s)) * (lm_loss - s2s_loss))
            score_3 = numpy.mean(numpy.exp((1 - s2s_doc)) * (lm_loss_doc - s2s_loss_doc))
            score_2 = numpy.mean(numpy.exp((1 - s2s)) * (prefix_loss - s2s_loss))

            pred = 0.25 * score_1 + 0.5 * score_2 + 0.25 * score_3

            tmp = []
            for expert in data_dict["expert_annotations"]:
                tmp.append(expert["consistency"])
            sc = sum(tmp) / len(tmp)

            scores[model_name].append(sc)
            predicts[model_name].append(pred)

    predict_scores = []
    human_scores = []
    extractive_scores = []
    abstractive_scores = []
    overall_scores = []
    for model_name in scores.keys():
        predict_scores.append(sum(predicts[model_name]) / len(predicts[model_name]))
        human_scores.append(sum(scores[model_name]) / len(scores[model_name]))
        print(model_name, predict_scores[-1], human_scores[-1])

        overall_scores.append(predict_scores[-1])
        if model_name in ["M0", "M1", "M2", "M5"]:
            extractive_scores.append(predict_scores[-1])
        else:
            abstractive_scores.append(predict_scores[-1])
    print("extractive {} abstractive {} all {}".format(numpy.mean(extractive_scores), numpy.mean(abstractive_scores),
                                                       numpy.mean(overall_scores)))

    print("system-level correlation:")
    pearson, _ = pearsonr(predict_scores, human_scores)
    print(pearson)
    spearman, _ = spearmanr(predict_scores, human_scores)
    print(spearman)
    kendall, _ = kendalltau(predict_scores, human_scores)
    print(kendall)

    # print("instance-level correlation: model by model")
    # for model_name in scores.keys():
    #
    #     print("model", model_name)
    #     predict_scores = predicts[model_name]
    #     human_scores = scores[model_name]
    #     pearson, _ = pearsonr(predict_scores, human_scores)
    #     print(pearson)
    #     spearman, _ = spearmanr(predict_scores, human_scores)
    #     print(spearman)
    #     kendall, _ = kendalltau(predict_scores, human_scores)
    #     print(kendall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="summeval")
    args = parser.parse_args()

    summeval(args.file_path)
