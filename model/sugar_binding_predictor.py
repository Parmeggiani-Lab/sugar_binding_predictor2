import torch

from spherenet_M import SphereNet_M #SchNet, DimeNetPP, ComENet

import torch
import torch.nn as nn
import torch.nn.functional as F
import os,re
import sys
from torch_geometric.nn import PNAConv,global_add_pool
from torch_geometric.nn.dense import dense_diff_pool
from torch_geometric.utils import to_dense_adj
import argparse

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, act_func, num_layers):
        super(MLP, self).__init__()
        assert num_layers > 0
        if num_layers == 1:
            self.seq = nn.Linear(dim_in, dim_out)
        else:
            seq = [nn.Linear(dim_in, dim_hidden), act_func()]
            for i in range(num_layers - 2):
                seq.append(nn.Linear(dim_hidden, dim_hidden))
                seq.append(act_func())
            seq.append(nn.Linear(dim_hidden, dim_out))
            self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)
# model define

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device( 'cpu')

class sugar_binding_predictor(nn.Module):
    def __init__(self,node_hidden_dim=128,num_edge_type=3,n_feat = 26 ,piece_seq_len_max=23,piece_max_atom_num=9,ConvT=0,device='cuda',deg=None,fragment_cluster=None,sugar_cluster=None
                ):
        super(sugar_binding_predictor, self).__init__()
        self.ConvT=ConvT
        self.device=device
        self.node_hidden_dim=node_hidden_dim
        self.num_edge_type=num_edge_type
        self.n_feat=n_feat  # atom type number
        self.piece_seq_len_max=piece_seq_len_max
        self.piece_max_atom_num=piece_max_atom_num
    ######## define involve module
        self.SMP = SphereNet_M(energy_and_force=False, cutoff=4.0, num_layers=4,
                hidden_channels=40, out_channels=node_hidden_dim, int_emb_size=64,
                basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                num_spherical=3, num_radial=8,edge_feat=3, envelope_exponent=5,
                num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
                )
        self.deg=deg
        self.fragment_cluster=fragment_cluster
        self.sugar_cluster= sugar_cluster
        self.relu_atom = nn.ReLU().to(device)
        self.relu_fragment=nn.ReLU().to(device)
        self.relu_molecule = nn.ReLU().to(device)

        if self.deg!=None:
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            self.PNAConv=PNAConv(in_channels=128, out_channels=128,
                               aggregators=aggregators, scalers=scalers, deg=deg,
                               edge_dim=40, towers=1, pre_layers=1, post_layers=1,
                               divide_input=False).to(device)

        self.h_to_pred = nn.Sequential(
            MLP(
                dim_in=node_hidden_dim,
                dim_hidden=node_hidden_dim // 2,
                dim_out=node_hidden_dim // 4,
                act_func=nn.ReLU,
                num_layers=3
            ),
            nn.ReLU(),
            MLP(
                dim_in=node_hidden_dim//4,
                dim_hidden=node_hidden_dim // 8,
                dim_out=node_hidden_dim // 16,
                act_func=nn.ReLU,
                num_layers=3
            ),
            nn.ReLU(),
            MLP(
                dim_in=node_hidden_dim //16 ,
                dim_hidden=node_hidden_dim // 32,
                dim_out=node_hidden_dim // 64,
                act_func=nn.ReLU,
                num_layers=3
            ),
            nn.ReLU(),
        nn.Linear(node_hidden_dim // 64, 1),nn.Sigmoid()).to(device)

    @staticmethod
    def piece_to_molecule(pieces_ori):
        print(pieces_ori)
        print(len(pieces_ori))
        pieces = [i for i in pieces_ori if i != 0]
        pieces = [i - 14 for i in pieces]
        #print(pieces)
        molecule_list = []
        molecule_idx = 0
        sugar_number=0
        for i in pieces:
            if i == 0:
                molecule_idx += 1
                sugar_number+=1
            elif i > 0:
                pass
            elif i < 0:
                molecule_idx += 1
            if molecule_idx>0:
                molecule_list.append(molecule_idx)
            else:
                molecule_list.append(1)

        return torch.LongTensor(molecule_list), sugar_number

    @staticmethod
    def data_loading(info_dict):
        #####data loadding
        atom_number=int(eval(info_dict['atom_number']))
        edges=[torch.tensor(eval(info_dict['clean_pairwise_matrix'])[0]).to(device),torch.tensor(eval(info_dict['clean_pairwise_matrix'])[1]).to(device)]
        edge_attr=torch.FloatTensor(info_dict['clean_interacting_list']).to(device)


        h = torch.LongTensor(eval(info_dict['atomtype_id_list'])).to(torch.int64).to(device)
        coord=torch.FloatTensor(eval(info_dict['atom_coord_list'])).to(device)
        length=torch.zeros(atom_number).to(torch.int64).to(device)

        pieces_ori = eval(info_dict['piece_list'])

        atom_pos=torch.LongTensor(eval(info_dict['atompos_list'])).to(torch.int64)

        return h, coord, length, edges, edge_attr ,atom_number, atom_pos ,pieces_ori#,pieces_coord_i,,edge_select,gold_edge,pieces_coord_for_rnn

    def forward(self,info_dict,return_u=False):

        ## dataloading
        try:
            h, coord, length, edges, edge_attr, atom_number, atom_pos,pieces_ori=self.data_loading(info_dict)
        except TypeError :
            if return_u == True:
                return 0, 0
            else:
                return 0
        ## sphereNet embeding
        print('edge_attr',edge_attr.device)
        print(edge_attr.size())
        print(edge_attr)

        ### remove all CH-pi interaction


        h, u , edge_attr = self.SMP(h, coord, length, edges, edge_attr, )

        edge_index=torch.LongTensor([edges[0].tolist(),edges[1].tolist()]).to(device)

        if self.deg!=None:
            h=self.relu_atom(self.PNAConv(h, edge_index, edge_attr))
            u=global_add_pool(h,torch.zeros(atom_number,dtype=torch.long,device=device))

            print('agg in atom-level:', u.size())

        if self.fragment_cluster!=None:
            print(atom_pos)
            atom_pos=F.one_hot((atom_pos)-1,num_classes=max(atom_pos)).to(torch.float).to(device)
            print('atom_pos:', atom_pos.size())
            adj=to_dense_adj(edge_index)
            hf,adj_f,link_loss_f,ent_loss_f=dense_diff_pool(x=h,adj=adj,s=atom_pos)
            hf=self.relu_fragment(hf)
            print('hf:', hf.size())

            u = global_add_pool(hf.squeeze(0), torch.zeros(hf.size(1), dtype=torch.long,device=device))#/int(hf.size(1))
            print('agg in fragment-level:', u.size())

        if self.sugar_cluster!=None:
            molecule_pos,sugar_number=self.piece_to_molecule(pieces_ori)
            print(molecule_pos,sugar_number)
            molecule_pos = F.one_hot((molecule_pos)-1 , num_classes=max(molecule_pos)).to(torch.float).to(device)
            print('molecule_pos:', molecule_pos.size())
            print( molecule_pos)
            print('hf:', hf.size())
            print('adj_f:', adj_f.size())
            hm, adj_m, link_loss_m, ent_loss_m = dense_diff_pool(x=hf, adj=adj_f, s=molecule_pos)
            hm=self.relu_molecule(hm)
            print('hm:', hm.size())

            u = global_add_pool(hm.squeeze(0), torch.zeros(hm.size(1), dtype=torch.long,device=device)) / int(sugar_number)
            print('agg in molecule-level:', u.size())

        out_u = u
        print('u',u.device)
        pred = self.h_to_pred(u)

        print('pred: ',pred.size())

        if return_u==True:
            return pred, out_u
        else:
            return pred



