# not to get weird warning messages with the training stuff.
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from transformers import Trainer, TrainingArguments, EncoderDecoderModel, TrainerCallback
from importlib.machinery import SourceFileLoader
import torch
from torch.utils.data import DataLoader
from jeju_dataset import JjeKorDataset
from config import *
import json

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(">> Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">> Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print(">> Using CPU (slow!)")

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_metrics.jsonl"):
        self.log_file = log_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(logs) + "\n")

# load model
model_path = "models/jeju-transformer-bigger"
model = EncoderDecoderModel.from_pretrained(model_path)
model.to(device)
model.config.decoder_start_token_id = 2  # <bos>
model.config.eos_token_id = 3            # <eos>


# load vocab
jje_vocab = torch.load("vocab/jje_vocab.pt")
kor_vocab = torch.load("vocab/kor_vocab.pt")


# load JejuDataset class
train_file = "train_model_evaluate/data/input/jje-kor/jje-kor_train_small.tsv"
dev_file = "train_model_evaluate/data/input/jje-kor/jje-kor_dev_small.tsv"
train_dataset = JjeKorDataset(train_file, jje_vocab, kor_vocab)
eval_dataset = JjeKorDataset(dev_file, jje_vocab, kor_vocab)

# collate function for padding
def collate_fn(batch):
    # Remove None items
    batch = [item for item in batch if item is not None]
    max_len = SEQUENCE_LENGHT
    src_batch = []
    tgt_batch = []
    
    for src, tgt in batch:  
        src = src[:max_len]
        pad_len = max_len - len(src)
        if pad_len > 0:
            src = torch.cat([src, torch.full((pad_len,), jje_vocab["<pad>"], dtype=src.dtype)])
        src_batch.append(src)

        tgt = tgt[:max_len]
        pad_len = max_len - len(tgt)
        if pad_len > 0:
            tgt = torch.cat([tgt, torch.full((pad_len,), kor_vocab["<pad>"], dtype=tgt.dtype)])
        tgt_batch.append(tgt)

    input_ids = torch.stack(src_batch)
    labels = torch.stack(tgt_batch)

    return {"input_ids": input_ids, "labels": labels}

# dataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_OF_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    logging_dir="./logs",
    learning_rate=LEARNING_RATE,
    save_total_limit=SAVE_TOTAL_LIMIT,
)


# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
        callbacks=[MetricsLoggerCallback("training_metrics.jsonl")],
)

# start training
trainer.train()
results = trainer.evaluate()
print("evaluation results:", results) # <-- this evaluation result is based on the dev set