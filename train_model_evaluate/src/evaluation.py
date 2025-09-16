"""
evaluate the model on test file. this does not generate the output result from the model.
input_file: test_file has two columns: jje-jje OR jje-kor
output: evaluation results saved to eval_log_file as JSON
"""

import logging
# not to get weird warning messages with the training stuff.
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
from transformers import Trainer, TrainingArguments, EncoderDecoderModel
from importlib.machinery import SourceFileLoader
import torch
from jeju_dataset import JjeKorDataset
import json
from config import *

# file paths
vocab_path = "vocab_build.py"
test_file = "train_model_evaluate/data/input/jje-jje/jje-jje_test_small_gold.tsv" # <-- this test file must have both columsn: jje-jje or jje-kor
checkpoint_path = "model_output/jje-jje/checkpoint-333765"
eval_log_file = "./output/eval_results.json"


# load vocab from vocab.py
jje_vocab = torch.load("vocab/jje_vocab.pt")
kor_vocab = torch.load("vocab/kor_vocab.pt")


# load JejuDataset class
test_dataset = JjeKorDataset(test_file, jje_vocab)

# collate function for padding
def collate_fn(batch):
    # Remove None items
    batch = [item for item in batch if item is not None]
    max_len = SEQUENCE_LENGHT
    src_batch = []
    tgt_batch = []
    
    for src, tgt in batch:  # <-- now batch contains (jeju, korean)
        # process src (Jeju)
        src = src[:max_len]
        pad_len = max_len - len(src)
        if pad_len > 0:
            src = torch.cat([src, torch.full((pad_len,), jje_vocab["<pad>"], dtype=src.dtype)])
        src_batch.append(src)

        # process tgt (Korean)
        tgt = tgt[:max_len]
        pad_len = max_len - len(tgt)
        if pad_len > 0:
            tgt = torch.cat([tgt, torch.full((pad_len,), kor_vocab["<pad>"], dtype=tgt.dtype)])
        tgt_batch.append(tgt)

    input_ids = torch.stack(src_batch)
    labels = torch.stack(tgt_batch)

    return {"input_ids": input_ids, "labels": labels}


# load trained checkpoint(model)
print(f">> Loading model from {checkpoint_path}")
model = EncoderDecoderModel.from_pretrained(checkpoint_path)
model.config.decoder_start_token_id = 2  # <bos>
model.config.eos_token_id = 3            # <eos>

# training arguments (for eval only)
training_args = TrainingArguments(
    output_dir="./results_eval",
    per_device_eval_batch_size=8,
    do_eval=True,
    report_to="none",  
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset, # <-- use test dataset for eval variable
    data_collator=collate_fn,
)

# run evaluation
results = trainer.evaluate()
print("evaluation results:", results)

# save evaluation results to JSON
with open(eval_log_file, "a") as f:
    f.write(json.dumps(results) + "\n")
print(f"evaluation results saved to {eval_log_file}")
