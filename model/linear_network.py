import torch
import torch.nn.functional as F

from torch import nn

class ClassifierModule(nn.Module):
    def __init__(self, layer_dim=[784,98,10], atfc='ReLU', relu6_max=10.0, dropout=0.5, dropout_pos=0):
        super(ClassifierModule, self).__init__()

        self.relu6_max = relu6_max
        self.layer_dim = layer_dim
        self.dropout_pos = dropout_pos
        self.atfc_type = atfc

        #
        self.activation = self.get_atfc()
        self.dropout = nn.Dropout(dropout)
        #
        self.layers = nn.ModuleList(self.make_layer())
        self.forward_prop = self.make_forward_prop()

        print(nn.ModuleList(self.forward_prop))

    def get_atfc(self):
        #
        if self.atfc_type == 'ReLU':
            atfc = torch.nn.ReLU()
        elif self.atfc_type == 'ReLU6':
            atfc = torch.nn.ReLU6(max_value=self.relu6_max)
        else:
            raise NotImplementedError
        #
        return atfc
    def make_layer(self):
        layers = []
        for layer_idx in range(len(self.layer_dim)):
            if layer_idx != len(self.layer_dim)-1:
                layers.append(nn.Linear(self.layer_dim[layer_idx], self.layer_dim[layer_idx+1], bias=False))
            else:
                continue
        return layers

    def make_forward_prop(self):
        forward_prop = []
        for layer_idx in range(len(self.layers)):
            forward_prop.append(self.layers[layer_idx])
            if layer_idx != len(self.layers)-1:
                forward_prop.append(self.activation)
            if layer_idx == self.dropout_pos:
                forward_prop.append(self.dropout)
        return forward_prop

    def forward(self, X, **kwargs):
        if self.dropout_pos >= len(self.layer_dim):
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

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.biasis is not None:
                    nn.init.constant_(m.bias, 0)