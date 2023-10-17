# encoding = "utf-8"

import argparse

import json
from scorers.delta import Delta_Scorer
from load_dataset import Dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch


def main(args):
    model = LlamaForCausalLM.from_pretrained(args.pretrained_name, torch_dtype=torch.float16).to(args.device)
    tokenizer = LlamaTokenizer.from_pretrained(args.pretrained_name)
    model.eval()

    for file_name in ["summeval", "xsumfaith", "qagsxsum", "qagscnn", "frankxsum", "frankcnn", \
                      "summac-summeval", "summac-xsumfaith", "summac-cogensum", "summac-factcc", "summac-polytope",
                      "summac-frank"]:

        args.file_name = file_name

        print("Testing on {} {}".format(args.file_name, args.scorer))

        '''load dataset'''
        dataset = Dataset(args.file_name)
        source_lines = dataset.source_lines
        target_lines = dataset.target_lines
        human_scores = dataset.human_scores
        data = dataset.data

        '''get scores'''
        scorer = Delta_Scorer(model=model, tokenizer=tokenizer, pretrained_name=args.pretrained_name,
                              device=args.device)
        s2s_tok_list, lm_tok_list, prefix_tok_list, s2s_tok_list_doc, lm_tok_list_doc = scorer.compute(sources=source_lines,
                                                                                       targets=target_lines,
                                                                                       seperator="TL;DR ")

        '''save to files'''
        model_name = {"decapoda-research/llama-7b-hf": "llama7b",
                      "decapoda-research/llama-13b-hf": "llama13b",
                      "openlm-research/open_llama_3b": "llama3b",
                      }

        outputpath = "./output/" + str(args.file_name) + "-fflm-" + model_name[
            args.pretrained_name] + ".jsonl"
        outputfile = open(outputpath, "a+")

        for d, s2s, lm, pf, s2s_doc, lm_doc in zip(data, s2s_tok_list, lm_tok_list, prefix_tok_list, s2s_tok_list_doc, lm_tok_list_doc):
            d["s2s_tok_list"] = s2s
            d["lm_tok_list"] = lm
            d["prefix_tok_list"] = pf
            d["s2s_tok_list_1"] = s2s_doc
            d["lm_tok_list_1"] = lm_doc
            outputfile.write(json.dumps(d) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name",
        type=str,
        default="decapoda-research/llama-7b-hf"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    args = parser.parse_args()
    main(args)
