import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):

    # nodes: a list of Node objects, see the Node class
    def __init__(self, nodes, num_classes):
        super(CNN, self).__init__()
        self.nodes = nodes
        self.num_classes = num_classes
    
    # x: A [ N x C x H x W ] tensor representing the input
    #        N : batch size
    #        C : number of channels
    #        H : height of image
    #        W : width of image
    def forward(self, x):
        data = x
        for node in self.nodes:
            data = node.forward(data)
        n, c, h, w = data.shape
        # take average across channels before passing to linear layer
        output = torch.sum(data, dim=1, keepdim=True) / c
        output = nn.Flatten()(output)
        output = nn.Linear(h * w, self.num_classes)(output)
        return output

                
        