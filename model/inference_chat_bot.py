import numpy as np
import gc

import torch
from transformers import AutoTokenizer
from model.model_class import CrossEncoderBert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RetrivalBot:
    def __init__(
        self,
        finetuned_ce: CrossEncoderBert,
        data,
        context: str = "",
        size_patch=400,
        qty_rand_choose=5,
        max_out_context=200,
        max_length=128,
        tokenizer: AutoTokenizer = None,
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            self.tokenizer = tokenizer

        base_answers = data["close_reply"].values
        all_replies = []
        for rep in base_answers.tolist():
            all_replies.extend(rep)
        self.corpus = list(set(all_replies))

        self.finetuned_ce = finetuned_ce
        self.finetuned_ce.to(DEVICE)
        self.context = context
        self.size_patch = size_patch
        self.qty_rand_choose = qty_rand_choose
        self.max_out_context = max_out_context
        self.max_length = max_length

        if len(self.corpus) < self.qty_rand_choose * self.max_out_context:
            self.qty_rand_choose = int(len(self.corpus))

    def __flush_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        with torch.no_grad():
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def get_best_rand_reply(self, query) -> None:
        dic_answ = dict()
        dic_answ["score"] = []
        dic_answ["answer"] = []

        context_memory = query + "[SEP]" + self.context
        for i in range(self.qty_rand_choose):
            rand_patch_corpus = list(np.random.choice(self.corpus, self.size_patch))

            queries = [context_memory] * len(rand_patch_corpus)
            tokenized_texts = self.tokenizer(
                queries,
                rand_patch_corpus,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                ce_scores = self.finetuned_ce(
                    tokenized_texts["input_ids"],
                    tokenized_texts["attention_mask"],
                ).squeeze(-1)
                ce_scores = torch.sigmoid(ce_scores)
            scores = ce_scores.cpu().numpy()
            scores_ix = np.argsort(scores)[::-1][0]
            dic_answ["score"].append(scores[scores_ix])
            dic_answ["answer"].append(rand_patch_corpus[scores_ix])

        id = np.argsort(dic_answ["score"])[::-1][0]
        answer = dic_answ["answer"][id]
        self.conext_memory = answer + "[SEP]" + context_memory
        self.__flush_memory()
        self.conext_memory = self.conext_memory[: self.max_out_context]
        return answer, dic_answ["score"][id]


if __name__ == "__main__":
    pass
