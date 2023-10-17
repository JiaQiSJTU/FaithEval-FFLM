# encoding = "utf-8"

from tqdm import tqdm
import torch
import torch.nn.functional as F


class Delta_Scorer:
    def __init__(self,
                 model,
                 tokenizer,
                 pretrained_name: str = "decapoda-research/llama-7b-hf",
                 device="cuda",
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pretrained_name = pretrained_name

    def prepare_input(self, target, source=None, seperator=None, is_doc=False):

        bos_token_id = 0
        max_len = 2048

        target = self.tokenizer.tokenize(target)

        if source:
            seperator = self.tokenizer.tokenize(seperator)
            source = self.tokenizer.tokenize(source)

            if not is_doc:
                if len(source) + len(seperator) + len(target) + 1 > max_len:
                    truncate = max_len - 1 - len(target) - len(seperator)
                    source = source[:truncate]
            else:
                if len(source) + len(seperator) + len(target) + 1 > max_len:
                    truncate = max_len - 1 - len(source) - len(seperator)
                    target = target[:truncate]

            source_ids = self.tokenizer.convert_tokens_to_ids(source)
            seperator_ids = self.tokenizer.convert_tokens_to_ids(seperator)
            target_ids = self.tokenizer.convert_tokens_to_ids(target)

            input_ids = torch.tensor([[bos_token_id] + source_ids + seperator_ids + target_ids])
            start_idx = len(source_ids) + len(seperator_ids)
        else:
            if len(target) + 1 > max_len:
                truncate = max_len - 1
                target = target[:truncate]
            target_ids = self.tokenizer.convert_tokens_to_ids(target)

            input_ids = torch.tensor([[bos_token_id] + target_ids])
            start_idx = 0

        attention_mask = torch.tensor([[1] * len(input_ids)])

        return input_ids, attention_mask, start_idx

    def likelihoods(self, logits, force_decode_indices):
        probs = F.softmax(logits, dim=-1)
        probs_force_decode = probs.gather(-1, force_decode_indices.unsqueeze(-1)).squeeze()
        assert probs_force_decode.shape == force_decode_indices.squeeze().shape

        return probs_force_decode

    def compute(self, sources, targets, seperator="TL;DR "):

        s2s_tok_list = []
        lm_tok_list = []
        prefix_tok_list = []
        s2s_tok_list_doc = []
        lm_tok_list_doc = []

        for source, target in tqdm(zip(sources, targets), desc="computing FFLM score", total=len(targets)):
            input_ids, attention_mask, start_idx = self.prepare_input(target=target, source=source, seperator=seperator)
            prior_input_ids, prior_attention_mask, prior_start_idx = self.prepare_input(target=target)
            prefix_input_ids, prefix_attention_mask, prefix_start_idx = self.prepare_input(target=target,
                                                                                           source=target + " " + source,
                                                                                           seperator=seperator)

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            prior_input_ids = prior_input_ids.to(self.device)
            prior_attention_mask = prior_attention_mask.to(self.device)

            prefix_input_ids = prefix_input_ids.to(self.device)
            prefix_attention_mask = prefix_attention_mask.to(self.device)

            input_ids_doc, attention_mask_doc, start_idx_doc = self.prepare_input(target=source, source=target,
                                                                                  seperator=seperator, is_doc=True)
            input_ids_doc = input_ids_doc.to(self.device)
            attention_mask_doc = attention_mask_doc.to(self.device)

            prior_input_ids_doc, prior_attention_mask_doc, prior_start_idx_doc = self.prepare_input(target=source,
                                                                                                    is_doc=True)
            prior_input_ids_doc = prior_input_ids_doc.to(self.device)
            prior_attention_mask_doc = prior_attention_mask_doc.to(self.device)

            with torch.no_grad():
                s2s_logits = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        return_dict=True).logits
                lm_logits = self.model(input_ids=prior_input_ids,
                                       attention_mask=prior_attention_mask,
                                       return_dict=True).logits
                prefix_logits = self.model(input_ids=prefix_input_ids,
                                           attention_mask=prefix_attention_mask,
                                           return_dict=True).logits

                s2s_logits_doc = self.model(input_ids=input_ids_doc,
                                            attention_mask=attention_mask_doc,
                                            return_dict=True).logits
                lm_logits_doc = self.model(input_ids=prior_input_ids_doc,
                                           attention_mask=prior_attention_mask_doc,
                                           return_dict=True).logits

                s2s_probs = self.likelihoods(s2s_logits[0][start_idx:-1], input_ids[0][start_idx + 1:])
                lm_probs = self.likelihoods(lm_logits[0][prior_start_idx:-1], prior_input_ids[0][prior_start_idx + 1:])
                prefix_probs = self.likelihoods(prefix_logits[0][prefix_start_idx:-1],
                                                prefix_input_ids[0][prefix_start_idx + 1:])
                s2s_probs_doc = self.likelihoods(s2s_logits_doc[0][start_idx_doc:-1],
                                                 input_ids_doc[0][start_idx_doc + 1:])
                lm_probs_doc = self.likelihoods(lm_logits_doc[0][prior_start_idx_doc:-1],
                                                prior_input_ids_doc[0][prior_start_idx_doc + 1:])[
                               :s2s_probs_doc.size(0)]

                s2s_tok_list.append(s2s_probs.tolist())
                lm_tok_list.append(lm_probs.tolist())
                prefix_tok_list.append(prefix_probs.tolist())

                s2s_tok_list_doc.append(s2s_probs_doc.tolist())
                lm_tok_list_doc.append(lm_probs_doc.cpu().tolist())

        return s2s_tok_list, lm_tok_list, prefix_tok_list, s2s_tok_list_doc, lm_tok_list_doc
