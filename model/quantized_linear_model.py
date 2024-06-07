import torch
import torch.nn.functional as F

from torch import nn

class quantizedLinearModule(nn.Module):
    def __init__(self, model_fp32):
        super(quantizedLinearModule, self).__init__()
        # QuantStub converts tensors, which will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point, which will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP2 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted
        # from floating point to quantized in the quantized model.
        x = self.quant(x)
        x = self.model_fp32(x)

        # manually specify where tensors will be converted
        # from quantized to floating point in the quantized model.
        x = self.dequant(x)
        return x

    def predict(self, output):
        out = F.softmax(output, dim=-1)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        return out