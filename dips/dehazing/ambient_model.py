from torch import nn
import torch


class AmbientModel(nn.Module):
    # TODO: verify no bug (in the channels)
    def __init__(self, shape):
        super(AmbientModel, self).__init__()
        self.ambient = nn.Conv2d(3, 3, 1, 1, 0)  # input_c, output_c, k_size, stride, padding
        self.sig = nn.Sigmoid()
        self.input = torch.ones([1, 3, shape[0], shape[1]], requires_grad=False).cuda()

    def forward(self):
        return self.sig(self.ambient(self.input))