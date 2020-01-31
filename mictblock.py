def _to_4d_tensor(x, depth_stride=None):
    x = x.transpose(0, 2)  # DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # temporal down-sampling
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # 1x(N*D)xCxHxW
    x = x.squeeze(0)  # (N*D)xCxHxW
    return x, depth

def _to_5d_tensor(x, depth):
    x = torch.split(x, depth)  # N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # NxDxCxHxW
    x = x.transpose(1, 2)  # NxCxDxHxW
    return x

def forward(self, x):
    out1 = F.pad(x, (1, 1, 1, 1, 0, 2), 'constant', 0)
    out1 = self.conv3d(out1)   # 3D convolution
    out1 = self.bn(out1)
    out1 = self.relu(out1)

    x, depth = _to_4d_tensor(x, depth_stride=self.stride[0])
    out2 = self.conv2d[0](x)   # 2D convolution with BN & ReLU
    out2 = _to_5d_tensor(out2, depth)
    out = out1 + out2   # features fusion  

    out, depth = _to_4d_tensor(out)
    out = self.conv2d[1](out)   # 2D convolution with BN & ReLU
    out = _to_5d_tensor(out, depth)
    return out
  
