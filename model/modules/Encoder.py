import torch
import torch.nn as nn
import numpy as np
from Transparency.model.modelUtils import isTrue
from allennlp.common import Registrable
from allennlp.nn.activations import Activation
from Transparency.model.modules.lstm import LSTM
from torch.nn import LSTMCell
from Transparency.model.modules.ortholstm import OrthoLSTM
from collections import namedtuple

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class Encoder(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")


@Encoder.register('ortholstm')
class EncoderorthoRNN(Encoder) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = 1

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = OrthoLSTM(LSTMCell, input_size=embed_size, hidden_size=2*hidden_size, num_layers=self.num_layers, batch_first=True)
        self.output_size = self.hidden_size * 2

    def forward(self, data, is_embds=False) :
        seq = data.seq
        lengths = data.lengths

        if (len(seq.shape) == 2):  # look up embeds table
            embedding = self.embedding(seq)

        else:  # skip lookup
            embedding = data.seq
            embedding.requires_grad = True

        batch_size = embedding.size()[0]

        self.init_states = [ [LSTMState(cx=torch.zeros(batch_size, self.hidden_size),
                        hx=torch.zeros(batch_size, self.hidden_size)) for _ in range(2)] for _ in range(self.num_layers)]

        output, cell_output, output_state  = self.rnn(embedding, lengths)
        h, c = output_state[self.num_layers - 1]

        data.hidden = output
        data.cell_state = cell_output
        data.last_hidden = h
        data.embedding = embedding

        if isTrue(data, 'keep_grads') :
            data.embedding.retain_grad()
            data.hidden.retain_grad()


@Encoder.register('vanillalstm')
class EncoderorthoRNN(Encoder) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = 1

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = LSTM(LSTMCell, input_size=embed_size, hidden_size=2*hidden_size, num_layers=self.num_layers, batch_first=True)
        self.output_size = self.hidden_size * 2

    def forward(self, data,hx=None,is_embds=False) :
        seq = data.seq
        lengths = data.lengths

        if (len(seq.shape) == 2):  # look up embeds table
            embedding = self.embedding(seq)

        else:  # skip lookup
            embedding = data.seq
            embedding.requires_grad = True

        batch_size = embedding.size()[0]

        self.init_states = [ [LSTMState(cx=torch.zeros(batch_size, self.hidden_size),
                        hx=torch.zeros(batch_size, self.hidden_size)) for _ in range(2)] for _ in range(self.num_layers)]

        output, cell_output, output_state  = self.rnn(embedding, lengths, hx=hx)
        h, c = output_state[self.num_layers - 1]

        data.hidden = output
        data.cell_state = cell_output
        data.last_hidden = h
        data.embedding = embedding

        if isTrue(data, 'keep_grads') :
            data.embedding.retain_grad()
            data.hidden.retain_grad()
