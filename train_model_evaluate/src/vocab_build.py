"""
get words from jeju-korean parallel data(preprocessed_jeju-standard_parallel.tsv), and build a vocabulary for a transformer model
"""
# pip install torchtext --quiet
import re
from collections import Counter
from typing import Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, Vocab

# collect jeju and korean tokens seprately
jje_kor_tokenised = open("create_organise_data/data/processed/jje-kor_tokenised_filtered.tsv", encoding="utf-8")
jje_tokenised = iter([jje.split() for jje, _ in map(lambda x: x.strip().split("\t"), jje_kor_tokenised) if jje])

jje_kor_tokenised = open("create_organise_data/data/processed/jje-kor_tokenised_filtered.tsv", encoding="utf-8")
kor_tokenised = iter([kor.split() for _, kor in map(lambda x: x.strip().split("\t"), jje_kor_tokenised) if kor])


# build vocab
PAD_ID = 0
specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
jje_vocab = build_vocab_from_iterator(
    jje_tokenised,
    min_freq=5,                 
    specials=specials,
    special_first=True
)
# print first 20 tokens in the vocab
# print(jje_vocab.get_itos()[:20])  

kor_vocab = build_vocab_from_iterator(
    kor_tokenised,
    min_freq=5,                 
    specials=specials,
    special_first=True
)
jje_vocab.set_default_index(jje_vocab["<unk>"])
kor_vocab.set_default_index(kor_vocab["<unk>"])
print(f"Jeju Vocab Size: {len(jje_vocab)}")
print(f"Korean Vocab Size: {len(kor_vocab)}")


# def encode(tokens: List[str], vocab) -> List[int]:
#     tokens = ["<bos>"] + tokens + ["<eos>"]
#     return [vocab[t] for t in tokens]

# def decode(ids: List[int], vocab) -> List[str]:
#     return [vocab.lookup_token(i) for i in ids]


# ex_ids = encode(["안녕", "하", "세요", "제주", ".", "나", "는", "너", "를", "사랑", "해"], jje_vocab)
# print("Encoded:", ex_ids)
# print("Decoded:", decode(ex_ids,vocab=jje_vocab))

# 5) save the vocab
# torch.save(jje_vocab, "jje_vocab.pt")
# torch.save(kor_vocab, "kor_vocab.pt")

