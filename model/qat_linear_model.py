import torch
import torch.nn.functional as F

from torch import nn

class qat_model(nn.Module):
    def __init__(self, quantized_model, layer_dim=[784,98,10], dropout=0.5, dropout_pos=0):
        super(qat_model, self).__init__()

        self.layer_dim = layer_dim
        self.dropout_pos = dropout_pos
        self.weights = self.get_weights(quantized_model)
        #
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        #
        self.layers = nn.ModuleList(self.make_layer())
        self.forward_prop = self.make_forward_prop()

        print(nn.ModuleList(self.forward_prop))

    def get_weights(self, quantized_model):
        weights = []
        for layer_idx in range(len(self.layer_dim)-1):
            weights.append(torch.int_repr(quantized_model.get_submodule('model_fp32').get_submodule('layers').get_submodule(f'{layer_idx}')._weight_bias()[0]))

        return weights

    def make_layer(self):
        layers = []
        for layer_idx in range(len(self.layer_dim)):
            if layer_idx != len(self.layer_dim)-1:
                layers.append(nn.Linear(self.layer_dim[layer_idx], self.layer_dim[layer_idx+1]))
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

        weight_num = 0

        for layer in self.forward_prop:
            if layer is isinstance(layer, nn.Linear):
                out = F.linear(out, self.weights[weight_num].float())
                weight_num += 1
            else:
                X = layer(X)

        return X

        # values_btw_layers = []
        #
        # out = x
        # for layer_i in range(self.n_layers):
        #     out = F.linear(out, self.weights[layer_i].float())
        #     values_btw_layers.append(out)
        #     out = self.act(out)

        # return out if not statistics else (out, torch.cat(values_btw_layers, dim=1))
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