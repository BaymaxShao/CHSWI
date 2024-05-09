import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import cfg
from nets.diffmlp import PureMLP


def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def mlp_adv(latent_dim=384, seq=98, num_layers=12, input_dim=768, output_dim=5):

    return PureMLP(latent_dim, seq, num_layers, input_dim, output_dim)


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


# class ExpertMLP(nn.Module):
#
#     def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead):
#         super(ExpertMLP, self).__init__()
#         self.num_expert = 2
#         self.hidden_dim = 128
#         self.output_dim = 5
#         self.linear_embedding = nn.Linear(input_dim,embed_dim)
#         encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
#         self.marker = nn.ModuleList(nn.Sequential(nn.Linear(embed_dim, 256),nn.ReLU(),nn.Linear(256, output_dim)) for _ in range(self.num_expert))
#         self.gate = nn.Sequential(
#             nn.Linear(self.input_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim/2),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim/2,self.num_expert)
#             )
#
#     def forward(self, input_tensor):
#
#         x = self.linear_embedding(input_tensor)
#         x = x.permute(1,0,2)
#         x = self.transformer_encoder(x)
#         x = x.permute(1,0,2)[:,-1]
#         gate = self.gate(x)  #batch_41_4
#         gate_output = torch.softmax(gate.reshape,dim=-1)
#         activate_score,id_chosen = torch.max(gate_output,dim=-1)
#         id_chosen = id_chosen.reshape(1,256)
#         hist = torch.histc(id_chosen,bins=4,min=0,max=3)
#         x_list = (id_chosen*torch.where(torch.eq(id_chosen,i),x,torch.zeros_like(x)) for i in range[0,self.num_expert-1])
#         scores = 0
#         loss_score = 0
#         for i in range(self.num_expert):
#             scores +=self.marker[i](x_list[i])
#             loss_score += hist[i]*torch.sum(gate_output[i])
#         expert_loss = self.num_expert*loss_score/x.shape(0)
#
#         return scores, expert_loss