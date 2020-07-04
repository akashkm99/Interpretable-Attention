import torch
import torch.nn as nn
from Transparency.model.modelUtils import isTrue
from allennlp.common import Registrable
from allennlp.nn.activations import Activation
from torch.nn import LSTMCell
from collections import namedtuple
from Transparency.model.modelUtils import isTrue, BatchHolder, BatchMultiHolder

from allennlp.common.from_params import FromParams
from typing import Dict
from allennlp.common import Params


class RGenerator(nn.Module) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True,dropout=0.3)
        self.hidden_size = 2*hidden_size

        self.linear_1 = nn.Linear(self.hidden_size, 1)#self.hidden_size // 2)
        # self.linear_2 = nn.Linear(self.hidden_size//2, 1)

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)
        mask = data.masks.unsqueeze(-1).float() #(B, L, 1)

        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        logits = (self.linear_1(output))

        probs = torch.sigmoid(logits)
        probs = probs * (1-mask)  #(B, L, 1)

        return probs


class RGenerator_QA(nn.Module) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True,dropout=0.3)
        self.rnn_q = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True,dropout=0.3)
        self.hidden_size = 2*hidden_size

        self.linear_1 = nn.Linear(self.hidden_size, 1)#self.hidden_size // 2)
        # self.linear_2 = nn.Linear(self.hidden_size//2, 1)

    def forward(self, Pdata, Qdata) :

        seq = Qdata.seq
        lengths = Qdata.lengths
        embedding = self.embedding(seq) #(B, L, E)
        mask = Qdata.masks.unsqueeze(-1).float() #(B, L, 1)

        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.rnn_q(packseq)

        ########  Passage Encoder   ###############

        seq = Pdata.seq
        lengths = Pdata.lengths
        embedding = self.embedding(seq) #(B, L, E)
        mask = Pdata.masks.unsqueeze(-1).float() #(B, L, 1)

        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq,(h_n,c_n))
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        logits = (self.linear_1(output))

        probs = torch.sigmoid(logits)
        probs = probs * (1-mask)  #(B, L, 1)
        return probs


