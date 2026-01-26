import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.unet import Encoder, Decoder, get_padding_mask
from model.rq_quant import RQBottleneck


class PosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[-2]]


class RQVAE(nn.Module):
    def __init__(self,
                 config,
                 road_emb
                 ):
        super().__init__()

        en_decoder_config = config['Unet']
        rq_quant_config = config['RQ_quant']
        dropout = config['dropout']
        hidden_chs = en_decoder_config['hidden_chs']
        road_dim = en_decoder_config['road_dim']
        road_layers = en_decoder_config['road_layers']

        self.encoder = Encoder(
            ch=hidden_chs,
            ch_multi=en_decoder_config['ch_multi'],
            num_res_blocks=en_decoder_config['num_res_blocks'],
            dropout=dropout
        )

        self.rq_pre_quant_linear = nn.Linear(
            hidden_chs * en_decoder_config['ch_multi'][-1],
            rq_quant_config['d_model'],
        )

        self.rq_quant = RQBottleneck(
            base_n_embed=rq_quant_config['base_vocab_size'],
            codebook_dim=rq_quant_config['d_model'],
            rq_layer=rq_quant_config['n_codebooks'],
            rq_multi=rq_quant_config['vocab_multi'],
            beta=rq_quant_config['beta'],
        )

        self.rq_post_quant_linear = nn.Linear(
            rq_quant_config['d_model'],
            hidden_chs * en_decoder_config['ch_multi'][-1],
        )

        self.decoder = Decoder(
            ch=hidden_chs,
            ch_multi=en_decoder_config['ch_multi'],
            num_res_blocks=en_decoder_config['num_res_blocks'],
            dropout=dropout,
            road_dim=road_dim
        )

        road_emb = torch.cat([torch.zeros(1, road_emb.shape[-1]), road_emb], dim=0)
        self.road_emb = nn.Embedding.from_pretrained(road_emb, freeze=False, padding_idx=0)
        self.road_gps_lin = nn.Linear(2, road_dim)
        self.pos_emb = PosEmbedding(road_dim)

        self.road_trm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                road_dim, 4, 4 * road_dim, dropout,
                batch_first=True,
                activation=F.silu,
            ),
            road_layers,
        )

        self.percent_head = nn.Sequential(
            nn.Linear(hidden_chs, 2 * hidden_chs),
            nn.ReLU(),
            nn.Linear(2 * hidden_chs, 1)
        )
        self.dxy_head = nn.Sequential(
            nn.Linear(hidden_chs, 4 * hidden_chs),
            nn.ReLU(),
            nn.Linear(4 * hidden_chs, 2)
        )

    def reset_codebook_hit(self):
        self.rq_quant.reset_codebook_hit()

    def forward(self, x, lens, road_seq, road_gps, road_gps_len, dxy, road_percent):
        road_emb = self.road_emb(road_seq)
        road_gps_emb = self.road_gps_lin(road_gps)
        road_gps_mask = get_padding_mask(road_gps_len, x.device).squeeze(1)
        road_gps_emb = torch.sum(road_gps_emb * road_gps_mask.unsqueeze(-1), dim=1) / road_gps_len.unsqueeze(1)
        road_lens = (road_seq != 0).sum(dim=-1).long()
        road_gps_emb = torch.split(road_gps_emb, road_lens.tolist())
        road_gps_emb = torch.nn.utils.rnn.pad_sequence(road_gps_emb, batch_first=True, padding_value=0)
        road_emb = road_emb + road_gps_emb + self.pos_emb(road_emb)
        road_padding_mask = get_padding_mask(road_lens, x.device)
        road_emb = self.road_trm(road_emb, src_key_padding_mask=~road_padding_mask.squeeze(1))

        z_e, lens, all_lens_type, padding_mask = self.encoder(x, lens)

        # * remove padding
        z_e_valid = z_e[padding_mask]
        z_e_valid = self.rq_pre_quant_linear(z_e_valid)

        z_q_valid, rq_code, quant_loss, rq_info = self.rq_quant(z_e_valid)

        z_q_valid = self.rq_post_quant_linear(z_q_valid)

        # * restore padding
        z_q_restored = torch.zeros_like(z_e).to(x.device)
        z_q_restored[padding_mask] = z_q_valid

        rq_code_restored = torch.zeros((*padding_mask.shape, rq_code.shape[-1]), dtype=torch.long, device=z_e.device) - 1
        rq_code_restored[padding_mask] = rq_code

        z_d, padding_mask = self.decoder(z_q_restored, lens, all_lens_type, road_emb, road_padding_mask)

        z_d_valid = z_d[padding_mask]
        road_percent_valid = road_percent[padding_mask]
        dxy_valid = dxy[padding_mask]

        preds_percent = self.percent_head(z_d_valid)
        preds_dxy = self.dxy_head(z_d_valid)

        percent_loss = F.mse_loss(preds_percent, road_percent_valid.unsqueeze(-1))
        dxy_loss = F.mse_loss(preds_dxy, dxy_valid)

        recon_loss = percent_loss + dxy_loss

        rq_rtn = {
            'lens_type': torch.stack(all_lens_type).T,
            'recon_loss': recon_loss,
            'percent_loss': percent_loss,
            'dxy_loss': dxy_loss,
            'quant_loss': quant_loss,
            'code': rq_code_restored,
            'info': rq_info,
        }

        return rq_rtn

    def encode(self, x, lens, road_seq, road_gps, road_gps_len, dxy, road_percent):
        road_emb = self.road_emb(road_seq)
        road_gps_emb = self.road_gps_lin(road_gps)

        road_gps_mask = get_padding_mask(road_gps_len, x.device).squeeze(1)
        road_gps_emb = torch.sum(road_gps_emb * road_gps_mask.unsqueeze(-1), dim=1) / road_gps_len.unsqueeze(1)
        road_lens = (road_seq != 0).sum(dim=-1).long()
        road_gps_emb = torch.split(road_gps_emb, road_lens.tolist())
        road_gps_emb = torch.nn.utils.rnn.pad_sequence(road_gps_emb, batch_first=True, padding_value=0)
        road_emb = road_emb + road_gps_emb + self.pos_emb(road_emb)
        road_padding_mask = get_padding_mask(road_lens, x.device)
        road_emb = self.road_trm(road_emb, src_key_padding_mask=~road_padding_mask.squeeze(1))

        z_e, lens, all_lens_type, padding_mask = self.encoder(x, lens)

        # * remove padding
        z_e_valid = z_e[padding_mask]
        z_e_valid = self.rq_pre_quant_linear(z_e_valid)

        z_q_valid, rq_code, quant_loss, rq_info = self.rq_quant(z_e_valid)

        z_q_valid = self.rq_post_quant_linear(z_q_valid)

        padding_mask_e = padding_mask

        # * restore padding
        z_q_restored = torch.zeros_like(z_e).to(x.device)
        z_q_restored[padding_mask] = z_q_valid

        rq_code_restored = torch.zeros((*padding_mask.shape, rq_code.shape[-1]), dtype=torch.long, device=z_e.device) - 1
        rq_code_restored[padding_mask] = rq_code

        z_d, padding_mask = self.decoder(z_q_restored, lens, all_lens_type, road_emb, road_padding_mask)

        z_d_valid = z_d[padding_mask]

        road_percent_valid = road_percent[padding_mask]
        dxy_valid = dxy[padding_mask]

        preds_percent = self.percent_head(z_d_valid)
        preds_dxy = self.dxy_head(z_d_valid)

        percent_loss = F.mse_loss(preds_percent, road_percent_valid.unsqueeze(-1))
        dxy_loss = F.mse_loss(preds_dxy, dxy_valid)

        recon_loss = percent_loss + dxy_loss

        rq_rtn = {
            'lens_type': torch.stack(all_lens_type).T, 
            'recon_loss': recon_loss,
            'percent_loss': percent_loss,
            'dxy_loss': dxy_loss,
            'quant_loss': quant_loss,
            'code': rq_code_restored,
            'info': rq_info,
        }

        return rq_rtn, padding_mask_e.sum(dim=-1)

    def decode(self, code, code_len, all_lens_type, road_seq, road_gps, road_gps_len):
        road_emb = self.road_emb(road_seq)
        road_gps_emb = self.road_gps_lin(road_gps)

        road_gps_mask = get_padding_mask(road_gps_len, code.device).squeeze(1)
        road_gps_emb = torch.sum(road_gps_emb * road_gps_mask.unsqueeze(-1), dim=1) / road_gps_len.unsqueeze(1)
        road_lens = (road_seq != 0).sum(dim=-1).long()
        road_gps_emb = torch.split(road_gps_emb, road_lens.tolist())
        road_gps_emb = torch.nn.utils.rnn.pad_sequence(road_gps_emb, batch_first=True, padding_value=0)
        road_emb = road_emb + road_gps_emb + self.pos_emb(road_emb)
        road_padding_mask = get_padding_mask(road_lens, code.device)
        road_emb = self.road_trm(road_emb, src_key_padding_mask=~road_padding_mask.squeeze(1))

        z_q_valid = self.rq_quant.encode(code.T)
        z_q_valid = self.rq_post_quant_linear(z_q_valid)

        z_q_valid = torch.split(z_q_valid, code_len.tolist())
        z_q_valid = torch.nn.utils.rnn.pad_sequence(z_q_valid, batch_first=True, padding_value=0.0)

        z_d, padding_mask = self.decoder(z_q_valid, code_len, all_lens_type.bool().T, road_emb, road_padding_mask)

        z_d_valid = z_d[padding_mask]

        preds_percent = self.percent_head(z_d_valid)
        preds_dxy = self.dxy_head(z_d_valid)

        final_preds_percent = torch.zeros(padding_mask.shape[0], padding_mask.shape[1]).to(z_d.device)
        final_preds_dxy = torch.zeros(padding_mask.shape[0], padding_mask.shape[1], 2).to(z_d.device)

        final_preds_percent[padding_mask] = preds_percent.squeeze(-1)
        final_preds_dxy[padding_mask] = preds_dxy

        return final_preds_percent, final_preds_dxy, padding_mask.sum(dim=-1)
