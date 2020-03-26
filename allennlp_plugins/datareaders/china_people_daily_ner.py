"""
@Time    :2020/3/25 16:10
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import itertools
import json
import random
import os
import logging
import collections
from pathlib import Path

from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers.tokenizer import Tokenizer
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple, Iterable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def is_divide(line: str) -> bool:
    if len(line.strip()) == 0:
        return False
    else:
        return True

@DatasetReader.register("chinapeopledailyner")
class ChinaPeopleDailyNer(DatasetReader):
    def __init__(self, 
                 tokenizer: Tokenizer,
                 token_indexers:  Dict[str, TokenIndexer],
                **kwargs
                 ):
        super(ChinaPeopleDailyNer, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexer = token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for is_divider, lines in itertools.groupby(f, is_divide):
                lines = list(lines)
                if is_divider:
                    raw_words = [w.strip().split(' ')[0] for w in lines]
                    labels = [w.strip().split(' ')[1] for w in lines]
                    assert len(raw_words) == len(labels)
                    assert len(raw_words) != 0
                    yield self.text_to_instance("".join(raw_words), labels)

    def text_to_instance(self, text: str, label: List[str] = None) -> Instance:
        if text.startswith("相比之下"):
            print()
        text = text[: 256]
        label = label[: 256]
        fields = {}
        tokens = self.tokenizer.tokenize(text)
        sequence = TextField(tokens, self.token_indexer)
        if len(sequence) != len(label):
            print(sequence)
            print(label)
            print(len(sequence), len(label))
            assert False
        fields['tokens'] = sequence
        if label:
            labels = SequenceLabelField(label, sequence)
            fields['tags'] = labels
        return Instance(fields)


