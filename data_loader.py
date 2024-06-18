import torch
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load, chinese_tokenizer_load
import config

DEVICE = config.device

class MaskGenerator:
    @staticmethod
    def generate_subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    @staticmethod
    def generate_standard_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.autograd.Variable(MaskGenerator.generate_subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class DataBatch:
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src = src.to(DEVICE)
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.prepare_target(trg, pad)

    def prepare_target(self, trg, pad):
        trg = trg.to(DEVICE)
        self.trg = trg[:, :-1]
        self.trg_y = trg[:, 1:]
        self.trg_mask = MaskGenerator.generate_standard_mask(self.trg, pad)
        self.ntokens = (self.trg_y != pad).data.sum()

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.load_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()
        self.BOS = self.sp_eng.bos_id()
        self.EOS = self.sp_eng.eos_id()

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def load_dataset(self, data_path, sort=False):
        dataset = json.load(open(data_path, 'r'))
        out_en_sent = [data[0] for data in dataset]
        out_cn_sent = [data[1] for data in dataset]
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        return [self.out_en_sent[idx], self.out_cn_sent[idx]]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return DataBatch(src_text, tgt_text, batch_input, batch_target, self.PAD)