"""
choose a tranformer model for jeju to jeju AND jeju to korean translation.
save the model using the configurations defined in config.py
"""
from transformers import EncoderDecoderModel, BertConfig, BertModel
import torch
from config import *


jje_vocab = torch.load("vocab/jje_vocab.pt")
kor_vocab = torch.load("vocab/kor_vocab.pt")

jje_vocab_size = len(jje_vocab)
kor_vocab_size = len(kor_vocab)

# Define common configuration
# #jeju-trainformer-small
# sequence_length = 32
# hidden_size = 128
# num_hidden_layers = 3
# num_attention_heads = 2
# intermediate_size = 512

# encoder config
encoder_config = BertConfig(
vocab_size=jje_vocab_size,
hidden_size=HIDDEN_SIZE,
num_hidden_layers=NUM_HIDDEN_LAYERS,
num_attention_heads=NUM_ATTENTION_HEADS,
intermediate_size=INTERMEDIATE_SIZE,
max_position_embeddings=SEQUENCE_LENGHT,
is_decoder=False,
add_cross_attention=False,
hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB
)


# decoder config 
decoder_config = BertConfig(
vocab_size=jje_vocab_size,
hidden_size=HIDDEN_SIZE,
num_hidden_layers=NUM_HIDDEN_LAYERS,
num_attention_heads=NUM_ATTENTION_HEADS,
intermediate_size=INTERMEDIATE_SIZE,
max_position_embeddings=SEQUENCE_LENGHT,
is_decoder=True,
add_cross_attention=True,
hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB
)


encoder = BertModel(encoder_config)
decoder = BertModel(decoder_config)

# create encoder-decoder model
model = EncoderDecoderModel(
encoder=encoder,
decoder=decoder
)
model.config.pad_token_id = 0 # <pad>
model.config.decoder_start_token_id = 2 # <bos>
model.config.eos_token_id = 3 # <eos>
model.decoder.config.decoder_start_token_id = 2 # <bos>
model.decoder.config.eos_token_id = 3 # <eos>


# # print model architecture for verification
# print(model)  
# save the model if needed
model.save_pretrained("./models/jeju-transformer-final")
print("model saved to ./models/jeju-transformer-final")

