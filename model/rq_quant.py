import torch
from torch import nn
from torch.nn import functional as F


class VQEmbedding(nn.Module):
    def __init__(self,
                 n_embed,
                 codebook_dim,
                 ):
        super().__init__()
        self.n_embed = n_embed
        self.eps = 1e-6

        self.codebook = nn.Embedding(n_embed, codebook_dim)
        self.codebook.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        self.register_buffer('codebook_hit', torch.zeros(n_embed))

    def forward(self, x):
        dist = (
            (x**2).sum(dim=-1, keepdim=True) +
            (self.codebook.weight.T**2).sum(dim=0, keepdim=True) -
            2 * x @ self.codebook.weight.T
        )
        embed_idxs = dist.argmin(dim=-1)
        z_q = self.codebook(embed_idxs)

        with torch.no_grad():
            info = {}
            hit_V = embed_idxs.bincount(minlength=self.n_embed).float()
            self.codebook_hit.add_(hit_V)
            info['avg_usage'] = (self.codebook_hit > 0).float().mean().item()
            one_hot_encodings = F.one_hot(embed_idxs, self.n_embed).type(z_q.dtype)
            avg_probs = torch.mean(one_hot_encodings, dim=0)
            ppl = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))
            info['ppl'] = ppl.item() / self.n_embed

        return z_q, embed_idxs, info

    @torch.no_grad()
    def get_quant_emb(self, x):
        dist = (
            (x**2).sum(dim=-1, keepdim=True) +
            (self.codebook.weight.T**2).sum(dim=0, keepdim=True) -
            2 * x @ self.codebook.weight.T
        )
        embed_idxs = dist.argmin(dim=-1)
        z_q = self.codebook(embed_idxs)
        return z_q

    @torch.no_grad()
    def encode(self, code):
        return self.codebook(code)


class RQBottleneck(nn.Module):
    def __init__(self,
                 base_n_embed,
                 codebook_dim,
                 rq_layer,
                 rq_multi,
                 beta=0.25,
                 ):
        super().__init__()

        self.base_n_embed = base_n_embed
        self.rq_multi = rq_multi
        self.rq_layer = rq_layer

        self.n_embed = [base_n_embed // rq_multi[layer_i] for layer_i in range(self.rq_layer)]
        self.codebooks = nn.ModuleList([
            VQEmbedding(
                n_embed=self.n_embed[i],
                codebook_dim=codebook_dim,
            )
            for i in range(self.rq_layer)]
        )
        self.beta = beta

    def updata_step(self):
        for i in range(self.rq_layer):
            self.codebooks[i]._update_step()

    def reset_codebook_hit(self):
        for i in range(self.rq_layer):
            self.codebooks[i].codebook_hit.zero_()

    def forward(self, x):
        info_list = []

        quant_embs, codes = [], []
        residual = x.detach().clone()

        aggregated_quant_list = []
        aggregated_quants = torch.zeros_like(x)

        for i in range(self.rq_layer):
            quant, code, info = self.codebooks[i](residual)
            residual = residual - quant
            aggregated_quants = aggregated_quants + quant
            aggregated_quant_list.append(aggregated_quants)
            quant_embs.append(quant)
            codes.append(code)
            info_list.append(info)

        loss_list = []
        for idx, quant in enumerate(aggregated_quant_list):
            quant_loss = F.mse_loss(x.detach(), quant)
            commitment_loss = self.beta * F.mse_loss(x, quant.detach())

            partial_loss = quant_loss + commitment_loss
            loss_list.append(partial_loss)
            info_list[idx]['loss'] = partial_loss.item()
            info_list[idx]['quant_loss'] = quant_loss.item()
            info_list[idx]['commitment_loss'] = commitment_loss.item()

        loss = torch.mean(torch.stack(loss_list))
        codes = torch.stack(codes, dim=-1)
        quant_embs = torch.stack(quant_embs, dim=-1).sum(dim=-1)
        if self.training:
            quant_embs = x + (quant_embs - x).detach()

        return quant_embs, codes, loss, info_list

    @torch.no_grad()
    def encode(self, code):
        quant_embs = []
        # for i in range(code.shape[0]):
        #     quant = self.codebooks[i].encode(code[i])
        #     quant_embs.append(quant)

        for i in range(3):
            quant = self.codebooks[i].encode(code[i])
            quant_embs.append(quant)

        quant_embs = torch.stack(quant_embs, dim=-1).sum(dim=-1)

        return quant_embs
