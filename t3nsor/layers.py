import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3


class TTEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 voc_size=None,
                 emb_size=None,
                 auto_shapes=None,
                 auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 d=3,
                 tt_rank=8,
                 batch_dim_last=None,
                 padding_idx=None):

        super(TTEmbedding, self).__init__()

        if auto_shapes:
            voc_quantization = t3.utils.suggest_shape(
                voc_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            emb_quantization = t3.utils.auto_shape(
                emb_size, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [voc_quantization, emb_quantization]
            self.shape = shape
            print('Created TTEmbeding layer with voc shape: {}. emb shape: {}'.format(voc_quantization, emb_quantization))
            
        else:
            self.shape = shape

        if init is None:
            if shape is None:
                raise ValueError('if init is not provided,'
                                 ' please specify shape')
        else:
            self.shape = init.raw_shape
        

        if init is None:
            init = t3.glorot_initializer(self.shape, tt_rank=tt_rank)

        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter

        # for p in self.parameters():
        #    p.name = 'tt_core'

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = self.shape[0]
        self.emb_quant = self.shape[1]

        self.padding_idx = padding_idx

    def forward(self, x):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.contiguous().view(-1)

        # x_ind = t3.ind2sub(self.voc_quant, x)
        # rows = t3.gather_rows(self.tt_matrix, x_ind)

        # rows = rows.view(x.shape[0], -1)

        full = self.weight.full()
        rows = full[x]

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        return rows.to(x.device)


class TTLinear(nn.Module):
    """
    Handles linear transformation in tensor train format on inputs of dimension (batch_size, input_dim)
    """
    def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 ):
        super(TTLinear, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]
            print('Created TTLinear layer with input shape: {}. output shape: {}'.format(in_quantization, out_quantization))


        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.shape = shape
        self.weight = t3.transpose(init).to_parameter()
        self.parameters = self.weight.parameter
        if bias:
            self.bias = torch.nn.Parameter(1e-3 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight_t = self.weight
        x_t = x.transpose(0, 1)
        if self.bias is None:
            return t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1)
        else:
            return t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias


class TTLinearSeq(TTLinear):
    """
    TT-Linear module to handle linear transformation on inputs that are sequences (batch_size, seq_len, input_dim)
    """
    def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 ):
        super(TTLinearSeq, self).__init__(in_features, out_features, bias, init, shape,
                 auto_shapes, d, tt_rank, auto_shape_mode, auto_shape_criterion)

    def forward(self, x):
        # print('-------Start of TTLinearSeq forward pass--------')
        # print('x shape before reshaping: ', x.shape)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        output = x.contiguous().view(batch_size * seq_len, -1)
        # print('x shape after reshaping: ', output.shape)
        output = super(TTLinearSeq, self).forward(output)
        # print('output shape after TTLinear: ', output.shape)
        output = output.contiguous().view(batch_size, seq_len, -1)
        # print('output shape after reshaping: ', output.shape)
        # print('-------End of TTLinearSeq forward pass--------')
        return output
