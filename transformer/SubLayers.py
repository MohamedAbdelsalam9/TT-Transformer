''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention
from t3nsor.layers import TTLinear
import transformer.Constants as Constants

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, tt_params={}):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # TODO create normal initialization for TTLinear
        if Constants.attention_ in tt_params:
            self.w_qs = TTLinear(d_model, d_model * d_k, auto_shapes=True,
                                 d=tt_params[Constants.attention_]["n_tt_cores"],
                                 tt_rank=tt_params[Constants.attention_]["tt_rank"])
            self.w_ks = TTLinear(d_model, d_model * d_k, auto_shapes=True,
                                 d=tt_params[Constants.attention_]["n_tt_cores"],
                                 tt_rank=tt_params[Constants.attention_]["tt_rank"])
            self.w_vs = TTLinear(d_model, d_model * d_v, auto_shapes=True,
                                 d=tt_params[Constants.attention_]["n_tt_cores"],
                                 tt_rank=tt_params[Constants.attention_]["tt_rank"])
        else:
            self.w_qs = nn.Linear(d_model, n_head * d_k)
            self.w_ks = nn.Linear(d_model, n_head * d_k)
            self.w_vs = nn.Linear(d_model, n_head * d_v)
            nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
            nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        if Constants.attention_ in tt_params:
            self.fc = TTLinear(n_head * d_v, d_model, auto_shapes=True,
                               d=tt_params[Constants.attention_]["n_tt_cores"],
                               tt_rank=tt_params[Constants.attention_]["tt_rank"])
        else:
            self.fc = nn.Linear(n_head*d_v, d_model)
            nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = output.contiguous().view(sz_b*len_q, -1)
        output = self.dropout(self.fc(output).contiguous().view(sz_b, len_q, -1))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, tt_params={}):
        super().__init__()
        if Constants.pff_ in tt_params:
            self.w_1 = TTLinear(d_in, d_hid, bias=True, auto_shapes=True,
                                d=tt_params[Constants.pff_]["n_tt_cores"],
                                tt_rank=tt_params[Constants.pff_]["tt_rank"])
            self.w_2 = TTLinear(d_hid, d_in, bias=True, auto_shapes=True,
                                d=tt_params[Constants.pff_]["n_tt_cores"],
                                tt_rank=tt_params[Constants.pff_]["tt_rank"])
        else:
            self.w_1 = nn.Linear(d_in, d_hid)
            self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        output = x.contiguous().view(batch_size*seq_len, -1)  # To apply Fully Connected position-wise
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.contiguous().view(batch_size, seq_len, -1)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
