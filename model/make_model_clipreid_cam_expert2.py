import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from model.DA_cam import _InstanceDA_En, _InstanceDA
from model.AC import InsEncoder, InsDecoder

from model.layers import TransformerDecoder, ReIDProjector, MoEProjector


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return state


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.clip_image_encoder = clip_model.visual

        # DA_MV
        self.max_sent_num = 2  # expert_num
        self.instanceDA = _InstanceDA(512, self.camera_num)
        self.insDA_en1 = _InstanceDA_En(512, self.camera_num)
        self.insDA_en2 = _InstanceDA_En(512, self.camera_num)
        self.insDA_en3 = _InstanceDA_En(512, self.camera_num)
        self.insDA_en4 = _InstanceDA_En(512, self.camera_num)
        self.insDA_en5 = _InstanceDA_En(512, self.camera_num)
        self.insDA_en6 = _InstanceDA_En(512, self.camera_num)

        # AutoEncoder
        self.insEn_1 = InsEncoder()
        self.insDe_1 = InsDecoder()

        self.insEn_2 = InsEncoder()
        self.insDe_2 = InsDecoder()

        self.insEn_3 = InsEncoder()
        self.insDe_3 = InsDecoder()

        self.insEn_4 = InsEncoder()
        self.insDe_4 = InsDecoder()

        self.insEn_5 = InsEncoder()
        self.insDe_5 = InsDecoder()

        self.insEn_6 = InsEncoder()
        self.insDe_6 = InsDecoder()

        self.ln_ins = nn.LayerNorm(normalized_shape=[512])

        self.mse_loss = nn.MSELoss()

        # Decoder
        self.decoder = TransformerDecoder(num_layers=2, d_model=512, nhead=8, dim_ffn=512, dropout=0.1,
                                          return_intermediate=False)

        # MoE Projector
        self.moe_proj = MoEProjector(word_dim=512, in_dim=512, out_dim=512)

        # ReID Projector
        self.reid_proj = ReIDProjector(in_dim=512, num_cls=num_classes)

        self.moe_consistency_loss = nn.MSELoss()

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.SOURCE_NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x=None, label=None, img_feats=None, text_feats=None, cam_label=None, get_image=False,
                get_text=False, mvdc=False, epoch=None):
        if get_text == True:
            prompts_view1, prompts_view2, prompts_view3, prompts_view4, prompts_view5, prompts_view6 = self.prompt_learner(
                label)
            # prompts = self.prompt_learner(label)
            # text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            text_features_view1 = self.text_encoder(prompts_view1, self.prompt_learner.tokenized_prompts)
            text_features_view2 = self.text_encoder(prompts_view2, self.prompt_learner.tokenized_prompts)

            return text_features_view1, text_features_view2
            # return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.clip_image_encoder(x)
            if self.model_name == 'RN50':
                img_feats = image_features_proj[0]

            elif self.model_name == 'ViT-B-16':
                img_feats = image_features_proj[:, 0, :]

            return img_feats

        if mvdc:
            # s1
            instance_sigmoid_s1 = self.instanceDA(img_feats)
            # instance_loss_s1 = nn.BCELoss()
            DA_ins_loss_cls_s1 = F.cross_entropy(instance_sigmoid_s1, cam_label)

            DA_ins_loss_cls = DA_ins_loss_cls_s1

            # region [ins MV]

            # region [视角1]
            MV1_ins_feat_s1 = self.insEn_1(img_feats)
            re1_ins_feat_s1 = self.insDe_1(MV1_ins_feat_s1)
            # 归一化
            MV1_ins_feat_s1 = self.ln_ins(MV1_ins_feat_s1)

            # 1. recon_loss(重构损失)
            MV1_ins_s1 = self.mse_loss(re1_ins_feat_s1, img_feats.detach())

            # 2. 领域分类损失
            # s1
            instance_sigmoid_s1_MV1 = self.insDA_en1(MV1_ins_feat_s1)
            DA_ins_MV1_s1 = F.cross_entropy(instance_sigmoid_s1_MV1, cam_label)

            # ins_MV1_loss = DA_ins_MV1_s1 + DA_ins_MV1_s2 + MV1_ins_s1 + MV1_ins_s2
            ins_MV1_recon_loss = MV1_ins_s1
            ins_MV1_cls_loss = DA_ins_MV1_s1

            # endregion

            # region [视角2]
            MV2_ins_feat_s1 = self.insEn_2(img_feats)
            re2_ins_feat_s1 = self.insDe_2(MV2_ins_feat_s1)
            # 归一化
            MV2_ins_feat_s1 = self.ln_ins(MV2_ins_feat_s1)

            # 1. recon_loss(重构损失)
            MV2_ins_s1 = self.mse_loss(re2_ins_feat_s1, img_feats.detach())

            # 2. 领域分类损失
            # s1
            instance_sigmoid_s1_MV2 = self.insDA_en2(MV2_ins_feat_s1)
            DA_ins_MV2_s1 = F.cross_entropy(instance_sigmoid_s1_MV2, cam_label)

            # ins_MV2_loss = DA_ins_MV2_s1 + DA_ins_MV2_s2 + MV2_ins_s1 + MV2_ins_s2
            ins_MV2_recon_loss = MV2_ins_s1
            ins_MV2_cls_loss = DA_ins_MV2_s1

            # endregion

            # 3. 多视角损失
            dif12_ins_s1 = (self.mse_loss(MV1_ins_feat_s1, MV2_ins_feat_s1.detach()) + self.mse_loss(MV2_ins_feat_s1,
                                                                                                     MV1_ins_feat_s1.detach())) / 2



            # ins_mv_dis_loss = torch.exp(-(dif12_ins_s1 + dif12_ins_s2 + dif13_ins_s1 + dif13_ins_s2 + dif23_ins_s1 + dif23_ins_s2))
            ins_mv_dis_loss = 1 / ( dif12_ins_s1)

            # ins_MV_loss = ins_MV1_loss + ins_MV2_loss + ins_MV3_loss + un_ins_dis # - (0.01) * (dif_ins_s1 + dif_ins_s2)
            ins_mv_recon_loss = ins_MV2_recon_loss + ins_MV1_recon_loss
            ins_mv_cls_loss = ins_MV2_cls_loss + ins_MV1_cls_loss

            # endregion

            return (MV1_ins_feat_s1, MV2_ins_feat_s1), DA_ins_loss_cls, ins_mv_recon_loss, ins_mv_cls_loss, ins_mv_dis_loss

        image_features_last, image_features, vis = self.clip_image_encoder(x)
        # img_feature_last = image_features_last[:, 0]
        img_feature = image_features[:, 0]
        img_feature_proj = vis[:, 0]
        # b, n, l = vis.shape
        # patch_n = int(math.sqrt(n - 1))
        # img_feat_map_proj = vis[:, 1:].reshape((b, l, patch_n, patch_n))
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        if self.training:
            batch_size = x.shape[0]
            # padding mask used in decoder
            pad_mask = torch.zeros([batch_size, self.max_sent_num]).bool().cuda()

            text_feats = text_feats.reshape((batch_size * self.max_sent_num, -1))
            pad_mask = pad_mask.reshape((batch_size * self.max_sent_num, -1))

            # MVDC
            fq_list = []
            # MV1_fq = self.insEn_1(img_feat_map_proj.permute(0, 3, 2, 1))
            # MV1_fq = self.ln_ins(MV1_fq).permute(0, 3, 2, 1)
            MV1_feat = self.insEn_1(img_feature_proj)
            MV1_feat = self.ln_ins(MV1_feat).half()
            fq_list.append(MV1_feat)

            # MV2_fq = self.insEn_2(img_feat_map_proj.permute(0, 3, 2, 1))
            # MV2_fq = self.ln_ins(MV2_fq).permute(0, 3, 2, 1)
            MV2_feat = self.insEn_2(img_feature_proj)
            MV2_feat = self.ln_ins(MV2_feat).half()
            fq_list.append(MV2_feat.half())

            # del MV1_fq, MV2_fq, MV3_fq

            # Decoder
            fq = torch.stack(fq_list, dim=1)
            fq = fq.transpose(0, 1)
            fq = fq.unsqueeze(-1).unsqueeze(-1)
            ori_shape = fq.shape
            fq = fq.reshape((ori_shape[0] * ori_shape[1],) + ori_shape[2:])
            b, c, h, w = fq.size()
            text_feats = text_feats.unsqueeze(1)
            fq = self.decoder(fq, text_feats, pad_mask).half()
            fq = fq.reshape(b, c, h, w)

            feat_moe_proj = fq.squeeze()
            # feat_moe_proj = self.bottleneck_proj(fq)
            pred = self.reid_proj(feat_moe_proj)
            pred_all = pred.reshape(
                (batch_size, self.max_sent_num, self.num_classes))  # b, max_sent_num, num_cls
            score = self.moe_proj(feat_moe_proj, text_feats.squeeze())  # b, max_sent_num

            # Moe weight sum reid pred
            score_all = score.reshape((batch_size, self.max_sent_num))
            score_all = score_all.softmax(dim=1)

            Moe_reid_pred = torch.einsum('bsd,bs->bd', pred_all, score_all)  # bs*ID_num

            # Moe_reid_pred = Moe_reid_pred.sigmoid()
            pred_all = pred_all.sigmoid()
            moe_consistency_loss = img_feature_proj.new_zeros(
                (self.max_sent_num, self.max_sent_num))
            for c_i in range(self.max_sent_num):
                for c_j in range(self.max_sent_num):
                    if c_i == c_j:
                        continue
                    moe_consistency_loss[c_i, c_j] = self.moe_consistency_loss(pred_all[:, c_i], pred_all[:, c_j])
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            # return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj, [
            #     Moe_reid_pred, moe_consistency_loss]
            return [cls_score, cls_score_proj], [MV1_feat, MV2_feat], [Moe_reid_pred,  moe_consistency_loss]
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i not in self.state_dict().keys():
                continue
            if 'decoder' in i or 'classifier' in i or 'classifer' in i or 'reid_proj' in i or 'prompt' in i:
                continue
            # if 'classifier' in i or 'classifer' in i or 'reid_proj' in i or 'prompt' in i:
            #     continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num=None):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        n_view = 6

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_view, n_cls_ctx, ctx_dim, dtype=dtype)
        # cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        # cls_ctx = self.cls_ctx[label]
        cls_ctx_view1 = self.cls_ctx[label, 0]
        cls_ctx_view2 = self.cls_ctx[label, 1]
        cls_ctx_view3 = self.cls_ctx[label, 2]
        cls_ctx_view4 = self.cls_ctx[label, 3]
        cls_ctx_view5 = self.cls_ctx[label, 4]
        cls_ctx_view6 = self.cls_ctx[label, 5]

        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts_view1 = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx_view1,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_view2 = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx_view2,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_view3 = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx_view3,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_view4 = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx_view4,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_view5 = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx_view5,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_view6 = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx_view6,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts_view1, prompts_view2, prompts_view3, prompts_view4, prompts_view5, prompts_view6
