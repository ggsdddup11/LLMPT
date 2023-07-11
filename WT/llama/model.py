# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


@dataclass
class ModelArgs:#定义模型参数类
    dim: int = 512#维度512
    n_layers: int = 8#层数8
    n_heads: int = 8#头数8
    vocab_size: int = -1  # 词汇量大小由tokenizer决定defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32#最大 批大小
    max_seq_len: int = 2048#最大序列长度


class RMSNorm(torch.nn.Module):#继承nn.Module类
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()#调用Module构造函数
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))# torch.nn.Parameter是继承自torch.Tensor的子类，作为nn.Module中的可训练参数使用

    def _norm(self, x):#计算输入x在最后一个维度上的均值,保持输出结果维度
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        #平方均值开根号倒数，防分母为0
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)#type_as()将某个张量数据类型转换为另一个张量的数据类型
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    #dim:attention_head维度，end:最大长度的2倍
    #计算与绝对位置相关的旋转的角度在极坐标下对应的复数tensor
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    #创建tensor，维度为传入维度的一半，并且除以传入维度，作为基础角度theta的指数(dim//2,)
    t = torch.arange(end, device=freqs.device)  # type: ignore
    #t为绝对位置信息，维度大小为(end,)
    freqs = torch.outer(t, freqs).float()  # type: ignore
    #每个绝对位置分配到对应角度相乘，每一个绝对位置都有dim/2个角度
    #外积，将t转置与freqs相乘，维度变为(end,dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    #利用绝对值和角度值，在极坐标下构建复数张量:torch.polar(abs,angle)
    #abs*cos(angle)+j*abs*sin(angle),torch.ones_like(freqs)建立与freqs同形状的全1张量
    #即为cos(freqs[:])+j*sin(freqs[:])复数张量
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    #将freqs_cis与输入x变为同形状
    ndim = x.ndim
    #获取x维度数
    assert 0 <= 1 < ndim
    #？？断言x的维度数大于1，为什么还要加个0<=1，有点奇怪
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    #断言freqs_cis形状为（x的第1维度，x的最后维度），假如x维度为[2,1,2]
    #？？此处有问题freqs_cis.shape是Torch.Size类型，怎么会跟元组类型等同？但实际操作验证结果也确实跟代码一致
    #不明白为什么
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    #freqs的1维度和最后维度与x保持一样，其余维度设置为1，则freqs维度为[1,1,2]
    return freqs_cis.view(*shape)
    #？？freqs_cis与x只是第1维度和最后维度一致，其他维度置1，为何就能保证同形状，除非一种情况
    ##x的其余维度也均为1，则二者形状正好一致。


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    #位置信息加入原有编码结果上
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    #将tensor形状重塑为最后一个维度拆分为两个维度：任意行和两列，转化为复数形式，
    #因为将tensor转为复数形式，其最后维度必须为两列，对应一实一虚。
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    #freqs_cis为广播整形
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    #xq_和freqs_cis逐位相乘，将结果复数转换为实数
    #python里的flatten(dim)表示，从第dim个维度开始展开，从第3维度开始展开
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):#注意力类
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads
        
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        #维度
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        #原维度：批次大小，语句长度，__
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        #整形维度：批次大小，语句长度，本地头数，头维度

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
