"""
@Time    :2020/3/26 10:31
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""
import json
import random
import os
import logging
import collections
from pathlib import Path

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import TokenIndexer, Vocabulary, IndexedTokenList
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.token import Token
from overrides import overrides
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



@TokenIndexer.register("chinese_bert_indexer")
class ChineseBertIndexer(TokenIndexer):
    """
    This `TokenIndexer` assumes that Tokens already have their indexes in them (see `text_id` field).
    We still require `model_name` because we want to form allennlp vocabulary from pretrained one.
    This `Indexer` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = None)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", max_length: int = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._namespace = namespace
        self._allennlp_tokenizer = PretrainedTransformerTokenizer(model_name)
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary = False

        self._num_added_start_tokens = self._allennlp_tokenizer.num_added_start_tokens
        self._num_added_end_tokens = self._allennlp_tokenizer.num_added_end_tokens

        self._max_length = max_length
        if self._max_length is not None:
            self._effective_max_length = (  # we need to take into account special tokens
                self._max_length - self._tokenizer.num_added_tokens()
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )

    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
        """
        Copies tokens from ```transformers``` model to the specified namespace.
        Transformers vocab is taken from the <vocab>/<encoder> keys of the tokenizer object.
        """
        if self._added_to_vocabulary:
            return

        vocab_field_name = None
        if hasattr(self._tokenizer, "vocab"):
            vocab_field_name = "vocab"
        elif hasattr(self._tokenizer, "encoder"):
            vocab_field_name = "encoder"
        else:
            logger.warning(
                """Wasn't able to fetch vocabulary from pretrained transformers lib.
                Neither <vocab> nor <encoder> are the valid fields for vocab.
                Your tokens will still be correctly indexed, but vocabulary file will not be saved."""
            )
        if vocab_field_name is not None:
            pretrained_vocab = getattr(self._tokenizer, vocab_field_name)
            for word, idx in pretrained_vocab.items():
                vocab._token_to_index[self._namespace][word] = idx
                vocab._index_to_token[self._namespace][idx] = word

        self._added_to_vocabulary = True

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)
        indices = [w.text_id for w in tokens]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        # output = {"token_ids": indices, 'mask': [True for _ in range(len(indices))]}
        output = {"token_ids": indices }
        return output


@TokenIndexer.register("test_pretrained_transformer")
class TestPretrainedTransformerIndexer(PretrainedTransformerIndexer):
    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        indices, type_ids = self._extract_token_and_type_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        output = {"token_ids": indices, "mask": [1] * len(indices)}
        if type_ids is not None:
            output["type_ids"] = type_ids

        return self._postprocess_output(output)

