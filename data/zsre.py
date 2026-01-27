from typing import Dict

import torch

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers import LlamaTokenizerFast

from data.base import BaseDataset


class ZSREDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        
        prompt = row["src"]
        equiv_prompt = row["rephrase"]
        answer = row["ans"] if "ans" in row else row["alt"]
        unrel_prompt = row["loc"] + "?"
        unrel_answer = row["loc_ans"]
        result={
            "edit_tuples": self.tok_tuples(prompt, answer),
            "equiv_tuples": self.tok_tuples(equiv_prompt, answer),
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
        }

        if "portability" in row:
            port_prompt= row["portability"]["New Question"]
            port_answer= row["portability"]["New Answer"]
            result["portability_tuples"] = self.tok_tuples(port_prompt, port_answer)
    
        return result
        
    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:

        if isinstance(self.tok, (GPT2TokenizerFast, LlamaTokenizerFast)):
            answer = " " + answer

        tok_prompt = self.tok(
            prompt,
            return_tensors = "pt",
        )
        tok_answer = self.tok(
            answer,
            return_tensors = "pt",
            add_special_tokens = False
        )

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
            
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)

        return tok_tuples
