import torch
import torch.nn.functional as F
import torch.optim as optim


class SeparableConv(torch.nn.Module):
    """Depthwise separable convolution layer implementation."""

    def __init__(self, nin, nout, kernel_size=5):
        super(SeparableConv, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=kernel_size, groups=nin)
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



