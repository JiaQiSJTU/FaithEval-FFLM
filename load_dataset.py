# encoding = "utf-8"

import json
import numpy

filter_upper_token = ['the', 'a', 'this', 'there', 'an', 'in', 'on']
def re_upper(document, summary):
    '''following https://github.com/NJUNLP/CoP'''

    # document = document.split(' ')
    tokens = summary.split(" ")
    upper_tokens = []
    for i in tokens:
        if i.capitalize().replace('.', '').replace('\'s', '').replace(",", '') in document and (
                i.lower() not in filter_upper_token):
            i = i.capitalize()
        upper_tokens.append(i)

    return " ".join(upper_tokens)


class Dataset:

    def __init__(self, file_name):

        self.source_lines = []
        self.target_lines = []
        self.human_scores = []
        self.data = []
        self.file_name = file_name

        if self.file_name=="qagscnn" or self.file_name=="qagsxsum":
            self.load_qags()

        elif self.file_name=="frankcnn" or self.file_name=="frankxsum":
            self.load_frank()

        elif self.file_name == "summeval":
            self.load_summeval()

        elif self.file_name == "xsumfaith":
            self.load_xsumfaith()

        elif "summac" in self.file_name:
            self.load_summac_dataset()




    def load_qags(self):
        '''from https://github.com/NJUNLP/CoP'''
        f = open("./data/"+self.file_name.upper()+".jsonl", "r")
        lines = f.readlines()

        for line in lines:
            data_dict = json.loads(line.strip())
            self.source_lines.append(data_dict["text"])
            self.target_lines.append(data_dict["claim"])
            self.human_scores.append(data_dict["score"])
            self.data.append(data_dict)

    def load_frank(self):
        '''from https://github.com/NJUNLP/CoP'''
        f = open("./data/"+self.file_name.upper()+".json", "r")
        lines = f.readlines()

        for line in lines:
            data_dict = json.loads(line.strip())
            self.source_lines.append(data_dict["text"])
            data_dict["claim"] = data_dict["claim"].capitalize()
            data_dict["claim"] = re_upper(data_dict["text"], data_dict["claim"])
            self.target_lines.append(data_dict["claim"])
            self.human_scores.append(data_dict['score'])
            self.data.append(data_dict)

    def load_summeval(self):
        '''from https://github.com/Yale-LILY/SummEval'''
        f = open("./data/model_annotations.aligned.paired.jsonl", "r", encoding = "utf-8")
        lines = f.readlines()

        for line in lines:
            data_dict = json.loads(line.strip())
            self.source_lines.append(data_dict["text"])
            self.target_lines.append(data_dict["decoded"])
            tmp = []
            for expert in data_dict["expert_annotations"]:
                tmp.append(expert["consistency"])
            tmp = sum(tmp) / len(tmp)
            self.human_scores.append(tmp)
            data_dict["score"] = tmp
            self.data.append(data_dict)


    def load_summac_dataset(self):
        '''from https://github.com/tingofurro/summac'''

        file_name = self.file_name.split("-")[1]

        for cut in ["val", "test"]:
            with open("./data/"+file_name+'_'+cut+".jsonl", "r") as f:
                for line in f:
                    line = json.loads(line.strip())
                    self.source_lines.append(line["document"])
                    self.target_lines.append(line["claim"])
                    self.human_scores.append(line["label"])
                    line["cut"] = cut
                    self.data.append(line)

    def length_statistics(self, tokenizer):

        source_len = []
        target_len = []
        for s in self.source_lines:
            source_len.append(len(tokenizer.tokenize(s)))
        for t in self.target_lines:
            target_len.append(len(tokenizer.tokenize(t)))

        print("source len: max {} min {} avg {} std {}".format(numpy.max(source_len), numpy.min(source_len),
                                                               numpy.mean(source_len), numpy.std(source_len)))
        print("target len: max {} min {} avg {} std {}".format(numpy.max(target_len), numpy.min(target_len),
                                                               numpy.mean(target_len), numpy.std(target_len)))

# if __name__ == '__main__':

    # from transformers import LlamaTokenizer
    # from tqdm import tqdm
    # tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    # while True:
    #     file_name = str(input())
    #     dataset = Dataset(file_name)
    #     dataset.length_statistics(tokenizer)


