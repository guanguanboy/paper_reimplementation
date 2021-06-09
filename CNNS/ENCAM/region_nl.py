import torch
from torch import nn
from NLblock import NONLocalBlock2D
import numpy as np

class RegionNONLocalBlock(nn.Module):
    def __init__(self, in_channels, grid=[4, 4]):
        super(RegionNONLocalBlock, self).__init__()

        self.non_local_block = NONLocalBlock2D(in_channels, sub_sample=True, bn_layer=False)
        self.grid = grid

    def forward(self, x):
        batch_size, _, height, width = x.size()

        #self.grid[0]表示分割的块数，dim表示沿着dimension along which to split the tensor（沿着哪个轴分块）
        input_row_list = x.chunk(self.grid[0], dim=2) #chunk方法可以对张量分块，返回一个张量列表：

        output_row_list = []
        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)

            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):

                grid = self.non_local_block(grid)
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        return output

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    bn_layer = True

    # img = Variable(torch.zeros(2, 3, 20))
    # net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    # out = net(img)
    # print(out.size())
    #
    img = Variable(torch.zeros(2, 3, 20, 20))
    net = RegionNONLocalBlock(3)
    out = net(img)
    print(out.size())
