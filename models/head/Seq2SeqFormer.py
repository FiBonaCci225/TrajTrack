''' Define the seq2seq model '''
import torch
import torch.nn as nn
from .Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
from mmengine.registry import MODELS
from .rle_loss import RLELoss


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
            enc_output = self.dropout(self.with_pos_embed(src_seq)) #--positional encoding off
        else:
            enc_output = self.dropout(self.with_pos_embed(src_seq)) 

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask) # vanilla attention mechanism
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
class Seq2SeqFormer(nn.Module):
    """
    A sequence-to-sequence transformer model that facilitates deep interaction between 
    point cloud sequences and bounding box (bbox) sequences through an attention-based mechanism.
    This leverages the inherent spatial and temporal relationships within the sequences to 
    enhance feature representation for tasks involving point clouds and their associated bounding boxes.
    """

    def __init__(
            self, cfg, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.2, n_position=100):

        super().__init__()
        self.criterion = RLELoss(q_distribution='laplace')
        self.use_motion_cls = getattr(cfg, 'use_motion_cls', True)
        self.config = cfg
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.proj=nn.Linear(128,d_model)
        self.proj2=nn.Linear(4,d_model) # 4 represents the dimensions for x, y, z, plus a time stamp
        self.l1=nn.Linear(d_model*8, d_model)
        self.l2=nn.Linear(d_model, 7)

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

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, trg_seq,src_seq,valid_mask):

        src_seq_=self.proj(src_seq) # Adjust the input features to 128 dimensions
        trg_seq_=self.proj2(trg_seq) # Also adjust Q to 128 dimensions, corresponding to the features of the input box

        enc_output, *_ = self.encoder(src_seq_.reshape(-1,256,self.d_model)) # Locally apply self-attention to every single frame

        enc_others,*_=self.encoder_global(src_seq_, global_feature=True) # Apply attention across frames globally

        # Implementing cross-decoder
        # Q: trg_seq_
        # K, V: Concatenate(enc_output, enc_others)
        enc_output=torch.cat([enc_output.reshape(-1,4*256,self.d_model), enc_others], dim=1) # default 4 frames
        dec_output, dec_attention,*_ = self.decoder(trg_seq_, None, enc_output, None) 
                                                

        # Project to output
        dec_output=dec_output.view(dec_output.shape[0],4,self.d_model*8)
        dec_output= self.l1(dec_output)
        dec_output= self.l2(dec_output)
        
        return dec_output

    def loss(self, data_samples, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']
        motion_pred = output['motion_pred']
        # seg_logits = output['seg_logits']
        updated_ref_boxs = output['updated_ref_boxs']
        with torch.no_grad():
            # seg_label = data_samples['seg_label']
            box_label = torch.stack(data_samples['box_label'])
            # box_label_prev = torch.stack(data_samples['box_label_prev'])
            motion_label = torch.stack(data_samples['motion_label'])
            # motion_state_label = torch.stack(data_samples['motion_state_label'])[:, 0]
            center_label = box_label[:, :3]
            angle_label = torch.sin(box_label[:, 3])
            # center_label_prev = box_label_prev[:, :3]
            # angle_label_prev = torch.sin(box_label_prev[:, 0, 3])
            center_label_motion = motion_label[:, 0, :3]
            angle_label_motion = torch.sin(motion_label[:, 0, 3])

            ref_label = torch.stack(data_samples['box_label_prev'])
            ref_center_label = ref_label[:, :, :3]  # B*hist_num*3
            ref_angle_label = torch.sin(ref_label[:, :, 3])

        # loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        loss_center_motion = self.criterion(motion_pred[:, :3], motion_pred[:, 4:], center_label_motion)
        # loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
        loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)

        # ----- Stage 1 loss ---------------------
        estimation_boxes = output['estimation_boxes']
        loss_center = self.criterion(estimation_boxes[:, :3], estimation_boxes[:, 4:], center_label)
        # loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
        loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
        # loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
        loss_dict["loss_center"] = loss_center# * self.config.center_weight
        loss_dict["loss_angle"] = loss_angle# * self.config.angle_weight
        # -----------------------------------------

        loss_center_aux = self.criterion(aux_estimation_boxes[:, :3], aux_estimation_boxes[:, 4:], center_label)
        # loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)
        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)

        # ---------------------refbox loss---------
        loss_center_ref = self.criterion(updated_ref_boxs.reshape(-1,7)[:, :3], updated_ref_boxs.reshape(-1,7)[:, 4:], ref_center_label.reshape(-1,3))
        # loss_center_ref = F.smooth_l1_loss(updated_ref_boxs[:, :, :3], ref_center_label)
        loss_angle_ref = F.smooth_l1_loss(torch.sin(updated_ref_boxs[:, :, 3]), ref_angle_label)
        # ---------------------refbox loss---------

        # loss_total += 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
        #               + 1 * (
        #                           loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight) \
        #               + 1 * (
        #                           loss_center_ref * self.config.ref_center_weight + loss_angle_ref * self.config.ref_angle_weight)

        loss_dict.update({
            # "loss_total": loss_total,
            # "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux, #* self.config.center_weight,
            "loss_center_motion": loss_center_motion, #* self.config.center_weight,
            "loss_angle_aux": loss_angle_aux, #* self.config.angle_weight,
            "loss_angle_motion": loss_angle_motion, #* self.config.angle_weight,
            "loss_center_ref": loss_center_ref, #* self.config.ref_center_weight,
            "loss_angle_ref": loss_angle_ref, #* self.config.ref_angle_weight,
        })
        # if self.box_aware:
        #     prev_bc = torch.flatten(data_samples['prev_bc'], start_dim=1, end_dim=2)
        #     this_bc = data_samples['this_bc']  # torch.Size([B, 1024, 9])
        #     bc_label = torch.cat([prev_bc, this_bc], dim=1)  # torch.Size([B, 4096, 9])
        #     pred_bc = output['pred_bc']  # torch.Size([B, 4096, 9])
        #     loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
        #     loss_total += loss_bc * self.config.bc_weight
        #     loss_dict.update({
        #         "loss_total": loss_total,
        #         "loss_bc": loss_bc
        #     })

        return loss_dict