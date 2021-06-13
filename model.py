import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from tdnn import TDNN
from adacos import AdaCos

def get_mask(tensor, lengths):
    mask = np.zeros((tensor.shape[0], tensor.shape[1], 1))

    for n in range(len(lengths)):
        l=lengths[n]
        if l > tensor.shape[1]:
            l=tensor.shape[1]
        mask[n:, :l, :] = 1.

    return torch.from_numpy(mask.astype(np.float32)).clone()

def stats_with_mask(tensor, mask):
    mean = torch.div(torch.sum(tensor*mask, dim=1, keepdim=True),torch.sum(mask, dim=1, keepdim=True))
    var = torch.square(tensor-mean)
    var = torch.sum(var*mask, dim=1, keepdim=True)
    var = torch.div(var, torch.sum(mask, dim=1, keepdim=True)+1.0e-8)
    std = torch.sqrt(var)
    if mean.shape[0] < 2:
        mean, std = mean.squeeze(), std.squeeze()
        mena,std = mean.unsqueeze(0), std.unsqueeze(0)
    else:
        mean, std = mean.squeeze(), std.squeeze()
    return mean, std

class X_vector(nn.Module):
    def __init__(self, input_dim = 40, class_num=2):
        super(X_vector, self).__init__()

        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=4, dilation=4,dropout_p=0.5)
        self.segment5 = nn.Linear(512, 1500)
        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)

        self.criterion = AdaCos(512, class_num)

        self.id2spk={}
        with open('speakers.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                spk = line.strip().split()[0]
                id = int(line.strip().split()[1])
                self.id2spk[id] = spk
                
    def forward(self, inputs, speakers=None, lengths=None):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        out = self.segment5(tdnn4_out) # (b, t, f)
        if lengths is None:
            mean = torch.mean(out,1) # (b, f)
            std = torch.std(out,1) # (b, f)
        else:
            mask=get_mask(out, lengths).cuda()
            mean, std = stats_with_mask(out, mask)
        stat_pooling = torch.cat((mean,std),1) # (b, fx2)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        #return x_vec
        loss, logits = self.criterion(x_vec, speakers)
        
        return loss, logits
    
    def predict(self, x):
        with torch.no_grad():
            x -= self.mean
            x /= self.std
            x = torch.from_numpy(x).clone().float()
            x = x.unsqueeze(0)
            _, pred = self.forward(x.cuda())
            idx = torch.argmax(pred, dim=1).to('cpu').detach().numpy().copy().astype(int)[0]
            
        return self.id2spk[idx]
    
    def top_k(self, x, k=5):
        with torch.no_grad():
            x -= self.mean
            x /= self.std
            x = torch.from_numpy(x).clone().float()
            x = x.unsqueeze(0)
            _, pred = self.forward(x.cuda())
            _, idx = torch.topk(pred, k)
            idx = idx.to('cpu').detach().numpy().copy().astype(int).squeeze().tolist()
            return [ self.id2spk[k] for k in idx ]
        
    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std
        
    def load_model(self, file):
        self.model.load_state_dict(torch.load(file, map_location=torch.device('cpu')), strict=False)

    def get_speaker(self, k):
        return self.id2spk[k]
