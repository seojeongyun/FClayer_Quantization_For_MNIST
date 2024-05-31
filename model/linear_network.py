import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class ClassifierModule(nn.Module):
    def __init__(self, num_layer=[(784,98,10)], dropout=0.5, dropout_pos=0):
        super(ClassifierModule, self).__init__()

        self.layers = self.make_layer()
        self.forward_prop = self.make_forward_prop()
        #
        self.num_layer = num_layer
        self.dropout_pos = dropout_pos
        #
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def make_layer(self):
        layers = []
        for layer_idx in self.num_layer:
            if layer_idx != len(self.num_layer):
                layers.append(nn.Linear(self.num_layer[0][layer_idx], self.num_layer[0][layer_idx+1]))
            else:
                continue
        return layers

    def make_forward_prop(self):
        forward_prop = []
        for layer_idx in self.layers:
            forward_prop.append(self.layers[layer_idx])
            forward_prop.append(self.activation)
            if layer_idx == self.dropout_pos:
                forward_prop.append(self.dropout)
        return forward_prop

    def forward(self, X, **kwargs):
        if self.dropout_pos >= len(self.num_layer):
            raise ValueError('dropout_pos value must smaller than len(self.num_layer)')

        for layer in self.forward_prop:
            X = layer(X)

        return X

        # X = self.layers[0](X)
        # X = self.activation(X)
        # X = self.dropout(X)
        # X = self.output(X)

    def predict(self, output):
        out = F.softmax(output, dim=-1)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        return out