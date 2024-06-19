import torch
import torch.nn as nn
from torch.autograd import Variable
import logging
import sacrebleu
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import chinese_tokenizer_load

# 运行单个训练或验证周期
def run_epoch(data, model, loss_fn):
    total_tokens = 0.0
    accumulated_loss = 0.0

    for batch in tqdm(data):
        output = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_fn(output, batch.trg_y, batch.ntokens)
        accumulated_loss += loss
        total_tokens += batch.ntokens

    avg_loss = accumulated_loss / total_tokens
    return avg_loss

# 训练并保存模型
def train(train_data, dev_data, model, model_par, criterion, optimizer):
    best_bleu = 0.0
    remaining_patience = config.early_stop

    for epoch in range(1, config.epoch_num + 1):
        model.train()
        train_loss = run_epoch(train_data, model_par, LossCompute(model.generator, criterion, optimizer))
        logging.info(f"Epoch: {epoch}, Training Loss: {train_loss}")

        model.eval()
        dev_loss = run_epoch(dev_data, model_par, LossCompute(model.generator, criterion, None))
        bleu_score = evaluate(dev_data, model)
        logging.info(f"Epoch: {epoch}, Validation Loss: {dev_loss}, BLEU Score: {bleu_score}")

        if bleu_score > best_bleu:
            torch.save(model.state_dict(), config.model_path)
            best_bleu = bleu_score
            remaining_patience = config.early_stop
            logging.info("-------- Best Model Saved! --------")
        else:
            remaining_patience -= 1
            logging.info(f"Early Stopping in: {remaining_patience} Epochs")

        if remaining_patience == 0:
            logging.info("-------- Early Stopping Triggered! --------")
            break

# 损失计算和反向传播
class LossCompute:
    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.view(-1, x.size(-1)), y.view(-1)) / norm
        loss.backward()
        if self.optimizer:
            self.optimizer.step()
            if config.use_noamopt:
                self.optimizer.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
        return loss.item() * norm

# 多GPU损失计算
class MultiGPULossCompute:
    def __init__(self, generator, criterion, devices, optimizer=None, chunk_size=5):
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.optimizer = optimizer
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, norm):
        total_loss = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        scattered_out = nn.parallel.scatter(out, target_gpus=self.devices)
        out_gradients = [[] for _ in scattered_out]
        scattered_targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        for i in range(0, scattered_out[0].size(1), self.chunk_size):
            out_chunks = [Variable(o[:, i:i + self.chunk_size].data, requires_grad=bool(self.optimizer)) 
                          for o in scattered_out]
            generated_chunks = nn.parallel.parallel_apply(generator, [[chunk] for chunk in out_chunks])
            y_chunks = [(g.view(-1, g.size(-1)), t[:, i:i + self.chunk_size].view(-1))
                        for g, t in zip(generated_chunks, scattered_targets)]
            losses = nn.parallel.parallel_apply(self.criterion, y_chunks)

            combined_loss = nn.parallel.gather(losses, target_device=self.devices[0])
            combined_loss = combined_loss.sum() / norm
            total_loss += combined_loss.data

            if self.optimizer:
                combined_loss.backward()
                for j, l in enumerate(losses):
                    out_gradients[j].append(out_chunks[j].grad.data.clone())

        if self.optimizer:
            out_gradients = [Variable(torch.cat(grads, dim=1)) for grads in out_gradients]
            final_grads = nn.parallel.gather(out_gradients, target_device=self.devices[0])
            out.backward(gradient=final_grads)
            self.optimizer.step()
            if config.use_noamopt:
                self.optimizer.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

        return total_loss * norm

# 模型评估
def evaluate(data, model, mode='dev', use_beam=True):
    tokenizer = chinese_tokenizer_load()
    references = []
    translations = []

    with torch.no_grad():
        for batch in tqdm(data):
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decoded, _ = beam_search(model, src, src_mask, config.max_len, config.padding_idx,
                                         config.bos_idx, config.eos_idx, config.beam_size, config.device)
            else:
                decoded = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)

            decoded = [seq[0] for seq in decoded]
            translated_texts = [tokenizer.decode_ids(seq) for seq in decoded]

            references.extend(batch.trg_text)
            translations.extend(translated_texts)

    if mode == 'test':
        with open(config.output_path, "w") as output_file:
            for idx, (ref, trans) in enumerate(zip(references, translations)):
                output_file.write(f"idx:{idx} {ref} ||| {trans}\n")

    bleu_score = sacrebleu.corpus_bleu(translations, [references], tokenize='zh')
    return float(bleu_score.score)

# 测试模型
def test(data, model, criterion):
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()

        test_loss = run_epoch(data, model_par, MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info(f"Test Loss: {test_loss}, BLEU Score: {bleu_score}")

# 单句翻译
def translate(src, model, use_beam=True):
    tokenizer = chinese_tokenizer_load()
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()

        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decoded, _ = beam_search(model, src, src_mask, config.max_len, config.padding_idx,
                                     config.bos_idx, config.eos_idx, config.beam_size, config.device)
            decoded = [seq[0] for seq in decoded]
        else:
            decoded = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)

        translation = [tokenizer.decode_ids(seq) for seq in decoded]
        print(translation[0])
