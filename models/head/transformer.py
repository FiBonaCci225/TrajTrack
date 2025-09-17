import torch
import torch.nn as nn
from .rle_loss import RLELoss
from mmengine.registry import MODELS
from .Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
from datasets.misc_utils import get_tensor_corners_batch

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class Encoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src_seq, src_mask=None, return_attns=False, global_feature=False):

        enc_slf_attn_list = []
        # -- Forward
        if global_feature:
            enc_output = self.dropout(self.with_pos_embed(src_seq))  # --positional encoding off
        else:
            enc_output = self.dropout(self.with_pos_embed(src_seq))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)  # vanilla attention mechanism
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, dec_enc_attn_list

@MODELS.register_module()
class transformer_head(nn.Module):
    """
    A sequence-to-sequence transformer model that facilitates deep interaction between
    point cloud sequences and bounding box (bbox) sequences through an attention-based mechanism.
    This leverages the inherent spatial and temporal relationships within the sequences to
    enhance feature representation for tasks involving point clouds and their associated bounding boxes.
    """

    def __init__(
            self, use_rot=False, box_aware=False, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.2, n_position=100):

        super().__init__()
        self.criterion = RLELoss(q_distribution='laplace')
        # self.use_motion_cls = getattr(cfg, 'use_motion_cls', True)
        # self.config = cfg
        self.d_model = d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.proj = nn.Linear(128, d_model)
        self.proj2 = nn.Linear(3, d_model)  # 4 represents the dimensions for x, y, z, plus a time stamp
        self.l1 = nn.Linear(d_model * 8, d_model)
        self.l2 = nn.Linear(d_model, 6)
        self.pos_emd = PositionEmbeddingLearned(d_model//2)
        # self.pos_emd_global = PositionEmbeddingLearned(128)
        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.wlh_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Linear(128, 512)
        )

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

    def forward(self, trg_seq, src_seq, wlh):
        wlh = self.wlh_mlp(wlh)
        # src_seq = src_seq + wlh

        B = src_seq.shape[0]
        C = src_seq.shape[1]
        pos = self.pos_emd(src_seq)
        # pos_global = self.pos_emd_global(src_seq)
        trg_seq = get_tensor_corners_batch(trg_seq, wlh, torch.zeros(B//2).cuda())
        src_seq_ = self.proj(src_seq.reshape(B,C,-1).transpose(1,2)) + pos.reshape(B, 64, -1).transpose(1,2)  # Adjust the input features to 128 dimensions
        trg_seq_ = self.proj2(trg_seq)  # Also adjust Q to 128 dimensions, corresponding to the features of the input box

        enc_output, *_ = self.encoder(src_seq_)  # Locally apply self-attention to every single frame

        enc_others, *_ = self.encoder_global(src_seq_.reshape(B//2,-1,self.d_model), global_feature=True)  # Apply attention across frames globally

        # Implementing cross-decoder
        # Q: trg_seq_
        # K, V: Concatenate(enc_output, enc_others)
        enc_output = torch.cat([enc_output.reshape(-1, 2 * 256, self.d_model), enc_others], dim=1)  # default 4 frames
        dec_output, dec_attention, *_ = self.decoder(trg_seq_, None, enc_output, None)

        # Project to output
        dec_output = dec_output.view(dec_output.shape[0], -1) + wlh
        dec_output = self.l1(dec_output)
        dec_output = self.l2(dec_output)
        results = {
            'coors': dec_output[:, :3],
            'sigma': dec_output[:, 3:],
        }
        return results

    def loss(self, results, data_samples):
        losses = dict()
        pred_coors = results['coors']
        sigma = results['sigma']
        gt_coors = torch.stack(data_samples['box_label'])
        losses['regression_loss'] = self.criterion(pred_coors, sigma, gt_coors)

        # if self.use_rot:
        #     pred_rot = results['rotation']
        #     gt_rot = torch.stack(data_samples['theta'])
        #     losses['rotation_loss'] = F.smooth_l1_loss(pred_rot, gt_rot)

        return losses
