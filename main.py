import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from train import train, test, translate
from data_loader import MTDataset
from utils import english_tokenizer_load
from model import make_model, LabelSmoothing


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class ModelRunner:
    def __init__(self):
        self.train_dataset = MTDataset(config.train_data_path)
        self.dev_dataset = MTDataset(config.dev_data_path)
        self.test_dataset = MTDataset(config.test_data_path)

        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=config.batch_size,
                                           collate_fn=self.train_dataset.collate_fn)
        self.dev_dataloader = DataLoader(self.dev_dataset, shuffle=False, batch_size=config.batch_size,
                                         collate_fn=self.dev_dataset.collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, shuffle=False, batch_size=config.batch_size,
                                          collate_fn=self.test_dataset.collate_fn)

        self.model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                                config.d_model, config.d_ff, config.n_heads, config.dropout)
        self.model_par = torch.nn.DataParallel(self.model)

        if config.use_smoothing:
            self.criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
            self.criterion.cuda()
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        if config.use_noamopt:
            self.optimizer = self.get_std_opt(self.model)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

    @staticmethod
    def get_std_opt(model):
        """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
        return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    def run(self):
        utils.set_logger(config.log_path)
        logging.info("-------- Dataset Build! --------")
        logging.info("-------- Get Dataloader! --------")
        train(self.train_dataloader, self.dev_dataloader, self.model, self.model_par, self.criterion, self.optimizer)
        test(self.test_dataloader, self.model, self.criterion)

    def translate_example(self):
        """单句翻译示例"""
        sent = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
               "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
               "to childless workers."
        # tgt: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
        self.one_sentence_translate(sent, beam_search=True)

    def one_sentence_translate(self, sent, beam_search=True):
        BOS = english_tokenizer_load().bos_id()  # 2
        EOS = english_tokenizer_load().eos_id()  # 3
        src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
        batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
        translate(batch_input, self.model, use_beam=beam_search)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    runner = ModelRunner()
    # runner.run()
    runner.translate_example()