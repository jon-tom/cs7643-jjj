import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CNNNode(nn.Module):
    # prev_nodes: a list of Node objects to represent sampled previous nodes described in the paper
    # operation: a number from 0 to 3 to represent what operation this node will perform:
    #            0 : 3 x 3 convolution
    #            1 : 5 x 5 convolution
    #            2 : 3 x 3 Max pool
    #            3 : 3 x 3 Average pool
    # in_channel: the number of input channels
    # out_channel: the number of output channels
    def __init__(self, prev_nodes, operation, in_channel, out_channel):
        super(CNN, self).__init__()
        self.prev_nodes = prev_nodes
        self.operation = operation
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.output = None
        self.module = []

    # x: A [ N x C x H x W ] tensor representing the input
    #        N : batch size
    #        C : number of channels
    #        H : height of image
    #        W : width of image
    def forward(self, x):
        input = x
        N, C, H, W = x.shape
        for prev in self.prev_nodes:
            if prev.output != None:
                prev_output = prev.output
                N_in, C_in, W_in, H_in = prev_output.shape
                # if the outputs from previous layers don't have the same spatial dimensions as x, we need to resize the outputs so the spatial dimensions can align with x
                if W_in != W or H_in != H:
                    # the ENAS paper didn't mention how to handle this, so I am just using transforms.resize here
                    prev_output = transforms.Resize((H, W))(prev_output)
                # concatenate the inputs along the channel dimension
                input = torch.cat(input, prev_output, dim=1)
        in_channel = input.shape[1]
        # apply 1 x 1 convolution on the concatenated inputs to make the number of channels = self.channel
        input = nn.Conv2d(in_channel, self.in_channel, 1, padding='same')
        if self.operation == 0:
            m = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel, 3, padding='same'),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(),
            )
        elif self.operation == 1:
            m = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel, 5, padding='same'),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(),
            )
        elif self.operation == 2:
            m = nn.MaxPool2d(3, stride=2)
        else:
            m = nn.AvgPool2d(3, stride=2)
        # TODO: the paper also mentioned using 3x3 and 5x5 depthwise convolution layers
        self.output = m(input)
        return self.output
