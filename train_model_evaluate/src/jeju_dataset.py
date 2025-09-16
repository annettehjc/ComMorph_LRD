"""
build transformer model where it translates jeju to jeju. Vocabulary is aleady built in src/8_vocab_build.py
jje_vocab.pt should be used for jeju.
it's a auto-encoder model where it copies jeju to jeju. 
10th step
"""

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from typing import List

class JjeKorDataset(Dataset):
    def __init__(self, file_path: str, jje_vocab: Vocab, kor_vocab: Vocab):
        self.jje_vocab = jje_vocab
        self.kor_vocab = kor_vocab
        self.jje_data = []
        self.kor_data = []
        
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    jje, kor = line.strip().split("\t")
                    self.jje_data.append(self.encode(jje.split(), jje_vocab))
                    self.kor_data.append(self.encode(kor.split(), kor_vocab))

    def encode(self, tokens: List[str], vocab: Vocab) -> List[int]:
        tokens = ["<bos>"] + tokens + ["<eos>"]
        return [vocab[t] for t in tokens]
    
    def decode(ids: List[int], vocab: Vocab) -> List[str]:
        return [vocab.lookup_token(i) for i in ids]


    def __len__(self) -> int:
        return len(self.jje_data)

    def __getitem__(self, idx: int) -> List[int]:
        return torch.tensor(self.jje_data[idx], dtype=torch.long), \
            torch.tensor(self.kor_data[idx], dtype=torch.long)
    
