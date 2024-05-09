import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers, mlp_adv
# from nets.CrossAttn import avatar_cross_atten
from utils.human_models import smpl
from config import cfg

class ClothNet(nn.Module):
    def __init__(self, type):
        super(ClothNet, self).__init__()
        self.type = type
        if self.type == 'res':
            input_feat_dim = 4096
        elif self.type == 'vit' or self.type == 'vtm':
            input_feat_dim = 4096
        elif self.type == 'stmix':
            input_feat_dim = 4096
        else:
            input_feat_dim = 4096

        # self.crossattn = avatar_cross_atten(f_dim=384)

        if 'uppercloth' in cfg.cloth_types:
            self.z_cut_uppercloth = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_uppercloth = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'coat' in cfg.cloth_types:
            self.z_cut_coat = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_coat = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'pants' in cfg.cloth_types:
            self.z_cut_pants = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_pants = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'skirts' in cfg.cloth_types:
            self.z_cut_skirts = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_skirts = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'hair' in cfg.cloth_types:
            self.z_cut_hair = make_linear_layers([input_feat_dim,6], relu_final=False)
            self.z_style_hair = make_linear_layers([input_feat_dim,12], relu_final=False)
        if 'shoes' in cfg.cloth_types:
            self.z_style_shoes = make_linear_layers([input_feat_dim,4], relu_final=False)

        self.cloth_cls_layer = make_linear_layers([input_feat_dim, len(cfg.cloth_types)], relu_final=False)
        # self.cloth_cls_layer = make_linear_layers([1536, 768], relu_final=False)
        # self.cloth_cls_layer1 = make_linear_layers([768, 384], relu_final=False)
        # self.cloth_cls_layer2 = make_linear_layers([768, 384], relu_final=False)
        # self.cloth_cls_layer3 = make_linear_layers([768, 1], relu_final=False)
        # self.cloth_cls_layer4 = make_linear_layers([384, 2], relu_final=False)
        # self.cloth_cls_layer5 = make_linear_layers([384, 2], relu_final=False)

        self.gender_cls_layer = make_linear_layers([input_feat_dim, 2], relu_final=False)
        # bs = 8
        #
        # if 'uppercloth' in cfg.cloth_types:
        #     self.z_cut_uppercloth = mlp_adv(384,bs,12,768,6)
        #     self.z_style_uppercloth = mlp_adv(384,bs,12,768,12)
        # if 'coat' in cfg.cloth_types:
        #     self.z_cut_coat = mlp_adv(384,bs,12,768,6)
        #     self.z_style_coat = mlp_adv(384,bs,12,768,12)
        # if 'pants' in cfg.cloth_types:
        #     self.z_cut_pants = mlp_adv(384,bs,12,768,6)
        #     self.z_style_pants = mlp_adv(384,bs,12,768,12)
        # if 'skirts' in cfg.cloth_types:
        #     self.z_cut_skirts = mlp_adv(384,bs,12,768,6)
        #     self.z_style_skirts = mlp_adv(384,bs,12,768,12)
        # if 'hair' in cfg.cloth_types:
        #     self.z_cut_hair = mlp_adv(384,bs,12,768,6)
        #     self.z_style_hair = mlp_adv(384,bs,12,768,12)
        # if 'shoes' in cfg.cloth_types:
        #     self.z_style_shoes = mlp_adv(384,bs,12,768,4)
        #
        # self.cloth_cls_layer = mlp_adv(384,bs,12, input_feat_dim, len(cfg.cloth_types))
        #
        # self.gender_cls_layer = mlp_adv(384,bs,12,input_feat_dim, 2)

    def forward(self, img_feat1, img_feat2):
        batch_size = img_feat1.shape[0]
        # if self.type == 'res':
        #     img_feat1 = img_feat1.mean((2,3))
        
        z_cuts, z_styles = [], []

        img_feat2 = img_feat2.to(torch.float32)

        for cloth_type in cfg.cloth_types:
            if cloth_type == 'uppercloth':
                z_cuts.append(self.z_cut_uppercloth(img_feat2))
                z_styles.append(self.z_style_uppercloth(img_feat2))
            elif cloth_type == 'coat':
                z_cuts.append(self.z_cut_coat(img_feat2))
                z_styles.append(self.z_style_coat(img_feat2))
            elif cloth_type == 'pants':
                z_cuts.append(self.z_cut_pants(img_feat2))
                z_styles.append(self.z_style_pants(img_feat2))
            elif cloth_type == 'skirts':
                z_cuts.append(self.z_cut_skirts(img_feat2))
                z_styles.append(self.z_style_skirts(img_feat2))
            elif cloth_type == 'hair':
                z_cuts.append(self.z_cut_hair(img_feat2))
                z_styles.append(self.z_style_hair(img_feat2))
            elif cloth_type == 'shoes':
                z_cuts.append(torch.zeros((batch_size,0)).float().cuda())
                z_styles.append(self.z_style_shoes(img_feat2))

        scores = self.cloth_cls_layer(img_feat2)
        # feat1 = self.cloth_cls_layer1(feat)
        # feat2 = self.cloth_cls_layer2(feat)
        # feat3 = self.cloth_cls_layer3(feat)
        # feat1, feat2 = self.crossattn(feat1, feat2)
        # feat1 = self.cloth_cls_layer4(feat1)
        # feat2 = self.cloth_cls_layer5(feat2)
        #
        # scores = torch.cat((feat1, feat2, feat3), dim=1)
        scores = torch.sigmoid(scores)

        genders = self.gender_cls_layer(img_feat2)
        genders = F.softmax(genders, dim=-1)


        return genders, scores, z_cuts, z_styles
