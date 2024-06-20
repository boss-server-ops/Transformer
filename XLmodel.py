import config
from data_loader import subsequent_mask

import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = config.device

class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.rel_pos_embed = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, length):
        # 返回长度为length的相对位置编码
        range_vec = torch.arange(length, device=self.rel_pos_embed.device)
        range_mat = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
        return self.rel_pos_embed[(range_mat + self.max_len) % self.max_len]


class RelativeMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(RelativeMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, r, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 相对位置编码
        r = self.linears[3](r).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores += torch.matmul(query, r.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerXLMemory(nn.Module):
    def __init__(self, d_model, mem_len):
        super(TransformerXLMemory, self).__init__()
        self.mem_len = mem_len
        self.d_model = d_model

    def forward(self, x, mems):
        if mems is None:
            return x
        else:
            return torch.cat([mems, x], dim=1)[:, -self.mem_len:, :]


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mem, r, mask):
        m = mem
        x = torch.cat([m, x], dim=1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, r, mask))
        return self.sublayer[1](x[:, -x.size(1):], self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, mem_len):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.memory = TransformerXLMemory(d_model, mem_len)

    def forward(self, x, mem, r, mask):
        mems = mem
        new_mems = []
        for layer in self.layers:
            x = layer(x, mems, r, mask)
            new_mems.append(x)
        return self.norm(x), new_mems


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, mem, src_mask, tgt_mask, r):
        m = mem
        x = torch.cat([m, x], dim=1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, r, tgt_mask))
        x = self.sublayer[1](x[:, -x.size(1):], lambda x: self.src_attn(x, memory, memory, r, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N, d_model, mem_len):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.memory = TransformerXLMemory(d_model, mem_len)

    def forward(self, x, memory, mem, src_mask, tgt_mask, r):
        mems = mem
        new_mems = []
        for layer in self.layers:
            x = layer(x, memory, mems, src_mask, tgt_mask, r)
            new_mems.append(x)
        return self.norm(x), new_mems


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class TransformerXL(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, memory, rpe):
        super(TransformerXL, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.memory = memory
        self.rpe = rpe

    def forward(self, src, tgt, src_mask, tgt_mask, memory=None):
        if memory is None:
            memory = [None] * len(self.encoder.layers)
        r = self.rpe(src.size(1))
        enc_out, new_memory = self.encoder(self.src_embed(src), memory, r, src_mask)
        r = self.rpe(tgt.size(1))
        dec_out, new_memory = self.decoder(self.tgt_embed(tgt), enc_out, memory, src_mask, tgt_mask, r)
        return self.generator(dec_out)

    def encode(self, src, src_mask, memory=None):
        r = self.rpe(src.size(1))
        return self.encoder(self.src_embed(src), memory, r, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, memory=None):
        r = self.rpe(tgt.size(1))
        return self.decoder(self.tgt_embed(tgt), memory, memory, src_mask, tgt_mask, r)

