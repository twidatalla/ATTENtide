# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/8/23 10:10
@author: Qichang Zhao
@Filename: model.py
@Software: PyCharm
"""
### Protein --> MHC
### drug --> peptide
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDTI(nn.Module):
    def __init__(self,hp,
                 MHC_MAX_LENGH = 1000,
                 peptide_MAX_LENGH = 100):
        super(AttentionDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.peptide_MAX_LENGH = peptide_MAX_LENGH
        self.peptide_kernel = hp.peptide_kernel
        self.MHC_MAX_LENGH = MHC_MAX_LENGH
        self.MHC_kernel = hp.MHC_kernel

        self.MHC_embed = nn.Embedding(26, self.dim,padding_idx=0) ### TODO why 26 embeddings? change to 20?
        self.peptide_embed = nn.Embedding(26, self.dim,padding_idx=0)
        self.peptide_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.peptide_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.peptide_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*4,  kernel_size=self.peptide_kernel[2]),
            nn.ReLU(),
        )
        self.peptide_max_pool = nn.MaxPool1d(self.peptide_MAX_LENGH-self.peptide_kernel[0]-self.peptide_kernel[1]-self.peptide_kernel[2]+3)
        self.MHC_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.MHC_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.MHC_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.MHC_kernel[2]),
            nn.ReLU(),
        )
        self.MHC_max_pool = nn.MaxPool1d(self.MHC_MAX_LENGH - self.MHC_kernel[0] - self.MHC_kernel[1] - self.MHC_kernel[2] + 3)
        self.attention_layer = nn.Linear(self.conv*4,self.conv*4)
        self.MHC_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.peptide_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, peptide, MHC):
        peptideembed = self.peptide_embed(peptide)
        MHCembed = self.MHC_embed(MHC)
        peptideembed = peptideembed.permute(0, 2, 1)
        MHCembed = MHCembed.permute(0, 2, 1)

        peptideConv = self.peptide_CNNs(peptideembed)
        MHCConv = self.MHC_CNNs(MHCembed)

        peptide_att = self.peptide_attention_layer(peptideConv.permute(0, 2, 1))
        MHC_att = self.MHC_attention_layer(MHCConv.permute(0, 2, 1))

        d_att_layers = torch.unsqueeze(peptide_att, 2).repeat(1, 1, MHCConv.shape[-1], 1)  # repeat along MHC size
        p_att_layers = torch.unsqueeze(MHC_att, 1).repeat(1, peptideConv.shape[-1], 1, 1)  # repeat along peptide size
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        Compound_atte = torch.mean(Atten_matrix, 2)
        MHC_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        MHC_atte = self.sigmoid(MHC_atte.permute(0, 2, 1))

        peptideConv = peptideConv * 0.5 + peptideConv * Compound_atte
        MHCConv = MHCConv * 0.5 + MHCConv * MHC_atte

        peptideConv = self.peptide_max_pool(peptideConv).squeeze(2)
        MHCConv = self.MHC_max_pool(MHCConv).squeeze(2)

        pair = torch.cat([peptideConv, MHCConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

