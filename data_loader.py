import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

import config
DEVICE = config.device

# 生成后续的mask矩阵
def subsequent_mask(size):
    """生成一个size x size的矩阵，上三角部分为1，其他为0的mask，用于屏蔽后续单词"""
    mask_shape = (1, size, size)
    mask = np.triu(np.ones(mask_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

# 定义一个Batch类，用于训练时持有数据和mask
class Batch:
    """包含训练时使用的源数据、目标数据和相应mask的Batch对象"""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src = src.to(DEVICE)
        self.src_mask = (src != pad).unsqueeze(-2)
        
        if trg is not None:
            self.trg = trg[:, :-1].to(DEVICE)
            self.trg_y = trg[:, 1:].to(DEVICE)
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """创建用于遮蔽padding和未来单词的mask"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask))
        return tgt_mask

# 定义用于加载和处理数据的MTDataset类
class MTDataset(Dataset):
    def __init__(self, data_path):
        self.src_sents, self.trg_sents = self._load_and_sort_dataset(data_path)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()
        self.BOS = self.sp_eng.bos_id()
        self.EOS = self.sp_eng.eos_id()

    @staticmethod
    def _load_and_sort_dataset(data_path):
        """加载数据集并按英文句子长度排序"""
        with open(data_path, 'r') as file:
            dataset = json.load(file)
        
        src_sents = [item[0] for item in dataset]
        trg_sents = [item[1] for item in dataset]
        
        sorted_indices = sorted(range(len(src_sents)), key=lambda i: len(src_sents[i]))
        sorted_src_sents = [src_sents[i] for i in sorted_indices]
        sorted_trg_sents = [trg_sents[i] for i in sorted_indices]
        
        return sorted_src_sents, sorted_trg_sents

    def __getitem__(self, index):
        return self.src_sents[index], self.trg_sents[index]

    def __len__(self):
        return len(self.src_sents)

    def collate_fn(self, batch):
        src_texts = [item[0] for item in batch]
        trg_texts = [item[1] for item in batch]

        src_sequences = [[self.BOS] + self.sp_eng.EncodeAsIds(text) + [self.EOS] for text in src_texts]
        trg_sequences = [[self.BOS] + self.sp_chn.EncodeAsIds(text) + [self.EOS] for text in trg_texts]

        src_tensor = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in src_sequences],
                                  batch_first=True, padding_value=self.PAD)
        trg_tensor = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in trg_sequences],
                                  batch_first=True, padding_value=self.PAD)

        return Batch(src_texts, trg_texts, src_tensor, trg_tensor, self.PAD)
