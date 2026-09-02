import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class ReIDProjector(nn.Module):
    def __init__(self, in_dim=256, num_cls=576):
        super().__init__()
        self.in_dim = in_dim
        self.reid_pred = nn.Linear(in_dim, num_cls)

    def forward(self, x):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        pred_all = self.reid_pred(x)
        return pred_all


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 512]):
        super(FPN, self).__init__()
        # text projection
        # self.txt_proj = linear_layer(in_channels[2], out_channels[2])  # 1024->1024
        # fusion 1: v5 & seq -> f_5: b, 1024, 8, 8
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)  # 1024->1024
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 16, 16
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 32, 32
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 32, 32
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state=None):
        # input: v3, v4, v5: 512, 32, 32 / 1024, 16, 16 / 1024, 8, 8
        # result: v3, v4, v5: 256, 32, 32 / 512, 16, 16 / 1024, 8, 8
        v3, v4, v5 = imgs
        # fusion 1: b, 512, 8, 8
        if state is not None:
            # text projection: b, 1024 -> b, 1024
            # state = self.txt_proj(state).unsqueeze(-1).unsqueeze(
            #     -1)  # b, 1024, 1, 1
            state = state.unsqueeze(-1).unsqueeze(-1)
            f5 = self.f1_v_proj(v5)  # b, 512, 8, 8
            f5 = self.norm_layer(f5 * state)  # b, 512, 8, 8
        else:
            f5 = self.f1_v_proj(v5)
            f5 = self.norm_layer(f5)
        # fusion 2: b, 512, 16, 16
        f4 = self.f2_v_proj(v4)  # b, 512, 16, 16
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')  # b, 512, 16, 16
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 16, 16
        f3 = self.f3_v_proj(v3)  # 512->256
        f3 = F.avg_pool2d(f3, 2, 2)  # b, 256, 16, 16
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))  # b, 512, 16, 16
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)  # b, 512, 8, 8
        fq4 = self.f4_proj4(f4)  # b, 512, 16, 16
        fq3 = self.f4_proj3(f3)  # b, 512, 16, 16
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 16, 16
        return fq


class MoEProjector(nn.Module):
    def __init__(self, word_dim=512, in_dim=512, out_dim=512):
        super(MoEProjector, self).__init__()
        self.vis = nn.Linear(in_dim, out_dim)
        self.txt = nn.Linear(word_dim, out_dim)
        self.reid_moe = nn.Sequential(nn.Linear(out_dim * 2, out_dim),
                                      nn.ReLU(),
                                      nn.Linear(out_dim, out_dim // 2),
                                      nn.ReLU(), nn.Linear(out_dim // 2, 1))

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
            b: bs*num_cls
        '''
        # b, c, h, w = x.shape
        # x = torch.reshape(x, (b, c, -1))
        # x = x.mean(dim=2)  # b, 512
        vis_feat = self.vis(x)  # b, 512
        txt_feat = self.txt(word)  # b, 512
        feat = torch.cat([vis_feat, txt_feat], dim=1)
        reid_moe = self.reid_moe(feat)
        reid_moe = reid_moe.reshape(-1)

        return reid_moe


if __name__ == '__main__':
    batch_size = 2
    max_sent_num = 3
    v3 = torch.randn(2, 512, 32, 32)
    v4 = torch.randn(2, 1024, 16, 16)
    v5 = torch.randn(2, 1024, 8, 8)

    word = torch.randn(2, max_sent_num, 77)
    x = torch.randn(2, max_sent_num, 77, 512)
    text_feats = torch.randn(2, max_sent_num, 512)

    pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

    text_feats = text_feats.reshape((batch_size * max_sent_num, -1))
    x = x.reshape((batch_size * max_sent_num, x.shape[2], x.shape[3]))
    pad_mask = pad_mask.reshape((batch_size * max_sent_num, -1))

    # neck = FPN(in_channels=[512, 1024, 1024], out_channels=[256, 512, 512])
    decoder = TransformerDecoder(num_layers=3, nhead=8, d_model=512, dim_ffn=512, dropout=0.1)
    #
    # fq_list = []
    # for i_sent in range(max_sent_num):
    #     this_fq = neck((v3, v4, v5), text_feats[i_sent * batch_size:(i_sent + 1) * batch_size])
    #     fq_list.append(this_fq)
    # fq = torch.stack(fq_list, dim=0)
    # fq = fq.transpose(0, 1)
    #
    # ori_shape = fq.shape
    # fq = fq.reshape((ori_shape[0] * ori_shape[1],) + ori_shape[2:])
    fq = torch.randn(6, 512, 1, 1)
    b, c, h, w = fq.size()

    fq = decoder(fq, x, pad_mask)
    fq = fq.reshape(b, c, h, w)

    # proj = Projector(word_dim=512, in_dim=256, num_cls=576, kernel_size=1)
    # pred = proj(fq, text_feats)
    #
    # print(fq.shape)
