import torch
import torch.nn as nn
from allennlp.common import Registrable
from Transparency.model.modelUtils import jsd, isTrue

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
import pickle

def masked_softmax(attn_odds, masks) :
    inf = 1e6
    # attn_odds = attn_odds + (masks)*(-inf)
    attn_odds.masked_fill_(masks, -float('inf'))
    attn = nn.Softmax(dim=-1)(attn_odds)
    return attn


def masked_softmax_np(attn_odds, masks) :
    attn = odds.masked_fill_(masks.bool(), -float('inf'))
    attn = nn.Softmax(dim=-1)(attn_odds)
    return attn

class Attention(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")

@Attention.register('tanh')
class TanhAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.hidden_size = hidden_size
        
    def forward(self, data) :
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)
        input_seq, hidden, masks = data.seq, data.hidden, data.masks
        lengths = data.lengths

        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        data.attn_logit = attn2
        attn = masked_softmax(attn2, masks)
        
        inf = 1e9
        if isTrue(data, 'erase_max'):
            attn2[:,attn.max(dim=1)[1]] = -1*inf
            attn = masked_softmax(attn2, masks)
        
        if isTrue(data, 'erase_random'):
            rand_len = (torch.rand(size=lengths.size()).to(device) * (lengths).float()).long()
            attn2[:,rand_len] = -1*inf
            attn = masked_softmax(attn2, masks)

        if isTrue(data, 'erase_given'):
            attn2[:,data.erase_attn] = -1*inf
            attn = masked_softmax(attn2,masks) 

        return attn

@Attention.register('multi_tanh')
class MultiTanhAttention(Attention) :
    def __init__(self, hidden_size, heads=2) :
        super().__init__()

        self.heads = heads
        self.projection = nn.Linear(hidden_size,hidden_size*heads,bias=False)
        self.attn1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size // 2).to(device) for _ in range(heads)])
        self.attn2 = nn.ModuleList([nn.Linear(hidden_size // 2, 1, bias=False).to(device) for _ in range(heads)])
        self.hidden_size = hidden_size*heads    

    def forward(self, data) :
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)
        input_seq, hidden, masks = data.seq, data.hidden, data.masks
        lengths = data.lengths

        b,l,h = hidden.shape

        proj_hiddens = torch.unbind(self.projection(hidden).view(b,l,self.heads,h),dim=2)  # heads - [ (B,L,H), ...]
        data.proj_hiddens = proj_hiddens

        attns = []

        erase_head = getattr(data, 'erase_head', -1)

        for idx,hidden in enumerate(proj_hiddens):
            
            hidden = hidden.to(device)
            attn1 = nn.Tanh()(self.attn1[idx](hidden))
            attn2 = self.attn2[idx](attn1).squeeze(-1)
            data.attn_logit = attn2
            attn = masked_softmax(attn2, masks)
            
            inf = 1e9
            if isTrue(data, 'erase_max') and (erase_head == idx):
                attn2[:,attn.max(dim=1)[1]] = -1*inf
                attn = masked_softmax(attn2, masks)

            if isTrue(data, 'erase_random') and (erase_head == idx):
                rand_len = (torch.rand(size=lengths.size()).to(device) * (lengths).float()).long()
                attn2[:,rand_len] = -1*inf
                attn = masked_softmax(attn2, masks)

            if isTrue(data, 'erase_given') and (erase_head == idx):
                attn2[:,data.erase_attn] = -1*inf
                attn = masked_softmax(attn2,masks) 

            attns.append(attn)
        return attns

@Attention.register('dot')
class DotAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1 = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size

    def forward(self, data) :
        #input_seq = (B, L), hidden = (B, L, H), masks = (B, L)
        input_seq, hidden, masks = data.seq, data.hidden, data.masks
        lengths = data.lengths

        attn1 = self.attn1(hidden) #/ (self.hidden_size)**0.5
        attn1 = attn1.squeeze(-1)
        data.attn_logit = attn1
        attn = masked_softmax(attn1, masks)

        inf = 1e6
        if isTrue(data, 'erase_max'):
            attn1[:,attn.max(dim=1)[1]] = -1*inf
            attn = masked_softmax(attn1, masks)
        
        if isTrue(data, 'erase_random'):
            rand_len = (torch.rand(size=lengths.size()).to(device) * (lengths).float()).long()
            attn1[:,rand_len] = -1*inf
            attn = masked_softmax(attn1, masks)

        if isTrue(data, 'erase_given'):
            attn1[:,data.erase_attn] = -1*inf
            attn = masked_softmax(attn1,masks) 

        return attn

@Attention.register('tanh_qa')
class TanhQAAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.attn1p = nn.Linear(hidden_size, hidden_size // 2)
        self.attn1q = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.hidden_size = hidden_size
        
    def forward(self, input_seq, hidden_1, hidden_2, masks, data) :
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)

        attn1 = nn.Tanh()(self.attn1p(hidden_1) + self.attn1q(hidden_2).unsqueeze(1))
        attn2 = self.attn2(attn1).squeeze(-1)
        attn = masked_softmax(attn2, masks)

        inf = 1e9
        
        if isTrue(data, 'erase_given'):
            attn2[:,data.erase_attn] = -1*inf
            attn = masked_softmax(attn2,masks) 

        return attn

@Attention.register('dot_qa')
class DotQAAttention(Attention) :
    def __init__(self, hidden_size) :
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_seq, hidden_1, hidden_2, masks, data) :
        #input_seq = (B, L), hidden = (B, L, H), masks = (B, L)
        attn1 = torch.bmm(hidden_1, hidden_2.unsqueeze(-1)) / self.hidden_size**0.5
        attn1 = attn1.squeeze(-1)
        attn = masked_softmax(attn1, masks)

        return attn

from collections import defaultdict
@Attention.register('logodds')
class LogOddsAttention(Attention) :
    def __init__(self, hidden_size, logodds_file:str) :
        super().__init__()
        logodds = pickle.load(open(logodds_file, 'rb'))
        logodds_combined = defaultdict(float)
        for e in logodds :
            for k, v in logodds[e].items() :
                if v is not None :
                    logodds_combined[k] += abs(v) / len(logodds.keys())
                else :
                    logodds_combined[k] = None
                    
        logodds_map = logodds_combined
        vocab_size = max(logodds_map.keys())+1
        logodds = np.zeros((vocab_size, ))
        for k, v in logodds_map.items() :
            if v is not None :
                logodds[k] = abs(v)
            else :
                logodds[k] = float('-inf')
        self.logodds = torch.Tensor(logodds).to(device)
        
        self.linear_1 = nn.Linear(hidden_size, 1)

    def forward(self, input_seq, hidden, masks) :
        #input_seq = (B, L), hidden = (B, L, H), masks = (B, L)
        attn1 = self.logodds[input_seq]
        attn = masked_softmax(attn1, masks)

        return attn

    def regularise(self, input_seq, hidden, masks, previous_attn) :
        attn = self.forward(input_seq, hidden, masks)
        js_divergence = jsd(attn, previous_attn)
        return js_divergence.mean()
