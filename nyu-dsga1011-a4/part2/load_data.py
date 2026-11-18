import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
BOS_ID = T5TokenizerFast.from_pretrained("google-t5/t5-small").convert_tokens_to_ids("<extra_id_0>")

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        """
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        """
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        # Read English questions
        nl_lines = load_lines(os.path.join(data_folder, f"{split}.nl"))

        # Convert English into token IDs
        self.encoder_ids = [tokenizer(x)["input_ids"] for x in nl_lines]

        # Training/dev data have SQL targets
        if split != "test":
            sql_lines = load_lines(os.path.join(data_folder, f"{split}.sql"))
            # T5 vocab has no "<" token: replace with placeholders the tokenizer can represent
            sql_lines = [s.replace("<=", "le_op").replace("<", "lt_op") for s in sql_lines]

            # Convert SQL into token IDs
            self.decoder_ids = [tokenizer(x)["input_ids"] for x in sql_lines]

            assert len(self.encoder_ids) == len(self.decoder_ids)

    def __len__(self):
        # TODO
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        # TODO
        # Retrieve one English question
        enc = torch.tensor(self.encoder_ids[idx], dtype=torch.long)

        # No SQL targets for test examples
        if self.split == "test":
            return enc

        # Retrieve the matching SQL query
        dec = torch.tensor(self.decoder_ids[idx], dtype=torch.long)

        return enc, dec

def normal_collate_fn(batch):
    """
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    """
    # TODO
    encs = [enc for enc, dec in batch]
    decs = [dec for enc, dec in batch]

    # 1. encoder: pad + mask
    encoder_ids  = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # 2. decoder: inputs = BOS + all-but-last token, targets = all tokens
    bos = torch.tensor([BOS_ID])
    dec_in = [torch.cat([bos, d[:-1]]) for d in decs]
    decoder_inputs  = pad_sequence(dec_in, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decs,   batch_first=True, padding_value=PAD_IDX)

    # 3. first decoder token for each example (used for generation later)
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    """
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    """
    # TODO
    encs = batch                     # each item is already just the encoder tensor

    encoder_ids  = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x