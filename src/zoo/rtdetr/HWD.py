import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar').cuda()  # 将DWTForward移动到GPU上
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1).cuda(),  # 将Conv2d移动到GPU上
            nn.BatchNorm2d(out_ch).cuda(),  # 将BatchNorm2d移动到GPU上
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x.cuda())  # 将输入x移动到GPU上进行计算
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = Down_wt(64, 64).cuda()  # 将模型移动到GPU上
    input = torch.rand(3, 64, 64, 64).cuda()  # 将输入数据移动到GPU上
    output = block(input)
    print(output.size())
