"""Implementation of batch-normalized LSTM."""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init

class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        
        output = []
        cell_output = []

        for time in range(max_time):
            h_next, c_next = cell(input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)

            output.append(h_next)
            cell_output.append(c_next)
            hx = hx_next
        
        output = torch.stack(output, 0)
        cell_output = torch.stack(cell_output,0)

        return output, cell_output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
                hx = (Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size,device=device))),
                  Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size,device=device))))
            else:

                hx = (Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size))),
                  Variable(nn.init.xavier_uniform(torch.empty(self.num_layers, batch_size, self.hidden_size))))
        # h_n = []
        # c_n = []
        output_states = []
        layer_output = None
        layer_cell_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = (hx[0][layer,:,:], hx[1][layer,:,:])
            
            if layer == 0:
                layer_output, layer_cell_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx_layer)
            else:
                layer_output, layer_cell_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            # h_n.append(layer_h_n)
            # c_n.append(layer_c_n)
            output_states.append((layer_h_n,layer_c_n))
        
        output = layer_output
        cell_output = layer_cell_output

        if self.batch_first:
            output = torch.transpose(output,0,1)
            cell_output = torch.transpose(cell_output,0,1)

        # h_n = torch.stack(h_n, 0)
        # c_n = torch.stack(c_n, 0)
        return output, cell_output, output_states
