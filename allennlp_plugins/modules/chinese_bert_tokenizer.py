"""
@Time    :2020/3/25 17:43
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
使用适配字的中文分词BERT模型分词器！
"""
import json
import random
import os
import logging
import collections
from pathlib import Path

from allennlp.data import  Token
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple, Union
from allennlp.data.tokenizers.tokenizer import Tokenizer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@Tokenizer.register("chinese_bert_tokenizer")
class ChineseBertTokenizer(Tokenizer):
    def __init__(self, model_name: str, ):
        self.model_name = model_name
        assert Path(model_name).exists(), "make sure your model_name exists!"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab: Dict[str, int] = self.tokenizer.vocab
        self.id2vocab = self.tokenizer.ids_to_tokens
        self.UNK_id = self.tokenizer.unk_token_id

    def tokenize(self, text: Union[str, List[str]]) -> List[Token]:
        if isinstance(text, str):
            text = list(text)
        ret_tokens = []
        for w in text:
            ret_tokens.append(Token(text=w, text_id=self.vocab.get(w, self.UNK_id)))

        return ret_tokens

