import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


def get_padding_mask(lens, device):
    padding_mask = torch.arange(lens.max().item(), device=device)[None, :] < lens[:, None]
    return padding_mask.unsqueeze(1)


class ConvLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        x = einops.rearrange(x, 'b c l -> b l c')
        x = self.norm(x)
        x = einops.rearrange(x, 'b l c -> b c l')
        return x


class PadConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.norm = ConvLayerNorm(in_channels)

    def forward(self, x, padding_mask):
        x = self.norm(x)
        x = F.silu(x) * padding_mask
        x = F.pad(x, pad=(1, 1), mode='constant', value=0.0)
        x = self.conv(x) * padding_mask
        return x


class Upsample2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)

    def forward(self, x, prev_lens_type, lens):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        lens = torch.where(prev_lens_type, lens * 2 - 1, lens * 2)
        if lens.max() % 2 == 1:
            x = x[:, :, :-1]
        x = F.pad(x, pad=(1, 1), mode='constant', value=0.0)
        x = self.conv(x)
        padding_mask = get_padding_mask(lens, x.device)
        return x * padding_mask, lens, padding_mask


class Downsample2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)

    def forward(self, x, lens):
        kernel_size = self.conv.kernel_size[0]
        x = F.pad(x, pad=(1, 1), mode='constant', value=0.0)
        x = self.conv(x)
        lens = torch.floor((lens + 2 - kernel_size) / 2) + 1
        padding_mask = get_padding_mask(lens, x.device)
        return x * padding_mask, lens.long(), padding_mask


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.conv1 = PadConv1d(in_channels, out_channels, kernel_size=3)
        self.conv2 = PadConv1d(out_channels, out_channels, kernel_size=3)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x, padding_mask):
        h = self.conv1(x, padding_mask)
        h = self.dropout(h)
        h = self.conv2(h, padding_mask)
        return self.nin_shortcut(x) * padding_mask + h


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_cond=None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        if d_cond is not None:
            self.W_k = nn.Linear(d_cond, d_model)
            self.W_v = nn.Linear(d_cond, d_model)
        else:
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TransformerSA(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(TransformerSA, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, 4 * d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        x = src
        src = src.permute(0, 2, 1).contiguous()
        src_mask = src_mask[:, None, :, :].repeat(1, 1, src.size(1), 1)
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))
        return src.permute(0, 2, 1).contiguous() + x


class TransformerCA(nn.Module):
    def __init__(self, d_model, d_cond, num_heads, dropout):
        super(TransformerCA, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, d_cond)
        self.feed_forward = PositionWiseFeedForward(d_model, 4 * d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask, tgt, tgt_mask):
        x = tgt
        tgt = tgt.permute(0, 2, 1).contiguous()
        tgt_mask = tgt_mask[:, None, :, :].repeat(1, 1, tgt.size(1), 1)
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_attn_output))
        src_mask = src_mask[:, None, :, :].repeat(1, 1, tgt.size(1), 1)
        cross_attn_output = self.cross_attn(tgt, src, src, src_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_output))
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        return tgt.permute(0, 2, 1).contiguous() + x


class Encoder(nn.Module):
    def __init__(
        self,
        ch,
        ch_multi,
        num_res_blocks,
        dropout
    ):
        super().__init__()
        self.ch = ch
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Linear(2, ch)
        self.down = nn.ModuleList()

        self.num_resolution = len(ch_multi) - 1

        for i, multi in enumerate(ch_multi[:-1]):
            block_in = ch * multi
            block = nn.ModuleList()
            attn = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, dropout=dropout))
                attn.append(TransformerSA(block_in, multi, dropout))
            down = nn.Module()
            down.block = block
            down.attn = attn
            block_out = ch * ch_multi[i + 1]
            down.downsample = Downsample2x(in_channels=block_in, out_channels=block_out)
            self.down.append(down)

        block_in = ch * ch_multi[-1]
        self.mid = nn.Module()
        self.mid.block1 = ResnetBlock(in_channels=block_in, dropout=dropout)
        self.mid.block2 = TransformerSA(block_in, ch_multi[-1],  dropout)
        self.mid.block3 = ResnetBlock(in_channels=block_in, dropout=dropout)

        # end
        self.norm_out = ConvLayerNorm(block_in)
        self.conv_out = nn.Conv1d(in_channels=block_in, out_channels=block_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x, lens):
        padding_mask = get_padding_mask(lens, x.device)
        h = self.conv_in(x).transpose(-2, -1).contiguous() * padding_mask

        all_lens_type = [(lens % 2).bool()]
        for i_level in range(self.num_resolution):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, padding_mask)
                h = self.down[i_level].attn[i_block](h, padding_mask)
            h, lens, padding_mask = self.down[i_level].downsample(h, lens)
            if i_level != self.num_resolution - 1:
                all_lens_type.append((lens % 2).bool())

        h = self.mid.block1(h, padding_mask)
        h = self.mid.block2(h, padding_mask)
        h = self.mid.block3(h, padding_mask)

        # end
        h = F.silu(self.norm_out(h), inplace=True) * padding_mask
        h = self.conv_out(h).transpose(-2, -1).contiguous()
        return h, lens, all_lens_type, padding_mask.squeeze(1)


class Decoder(nn.Module):
    def __init__(
        self,
        ch,
        ch_multi,
        num_res_blocks,
        dropout,
        road_dim,
    ):
        super().__init__()
        self.ch = ch
        self.num_res_blocks = num_res_blocks

        self.num_resolution = len(ch_multi) - 1

        self.conv_in = nn.Conv1d(in_channels=ch * ch_multi[-1], out_channels=ch * ch_multi[-1], kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block1 = ResnetBlock(in_channels=ch * ch_multi[-1], dropout=dropout)
        self.mid.block2 = TransformerCA(ch * ch_multi[-1], road_dim, ch_multi[-1], dropout)
        self.mid.block3 = ResnetBlock(in_channels=ch * ch_multi[-1], dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolution)):
            up = nn.Module()
            block_in = ch * ch_multi[i_level + 1]
            block_out = ch * ch_multi[i_level]
            up.upsample = Upsample2x(in_channels=block_in, out_channels=block_out)
            block_in = block_out

            block = nn.ModuleList()
            attn = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, dropout=dropout))
                attn.append(TransformerCA(block_in, road_dim, ch_multi[i_level], dropout))
            up.block = block
            up.attn = attn
            self.up.insert(0, up)

        self.norm_out = nn.LayerNorm(ch)
        self.conv_out = nn.Linear(ch, 2)

    def forward(self, x, lens, all_next_lens_type, road_emb, road_padding_mask):
        padding_mask = get_padding_mask(lens, x.device)
        h = self.conv_in(x.transpose(-2, -1).contiguous()) * padding_mask

        h = self.mid.block3(h, padding_mask)
        h = self.mid.block2(road_emb, road_padding_mask, h, padding_mask)
        h = self.mid.block1(h, padding_mask)

        for i_level in reversed(range(self.num_resolution)):
            h, lens, padding_mask = self.up[i_level].upsample(h, all_next_lens_type[i_level], lens)
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, padding_mask)
                h = self.up[i_level].attn[i_block](road_emb, road_padding_mask, h, padding_mask)

        h = h.transpose(-2, -1).contiguous()
        h = F.silu(self.norm_out(h), inplace=True)

        return h, padding_mask.squeeze(1)
