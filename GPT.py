import os
import torch
import numpy as np 

from torch import nn 
from Core.ANEGPT import AppleNeuralEngineGPT

#from Core.ANEGPT import AppleNeuralEngineGPT

# 

class CFG :
    vocab = 59
    embed_dim = 512
    inputs_len = 4
    block_size = 4


class GPT(nn.Module) :
    def __init__(self, cfg=CFG) :
        super(GPT, self).__init__()

        self.ANE_GPT = AppleNeuralEngineGPT()

    def forward(self, inputs) :
       
        x = self.ANE_GPT(inputs)
        
        
        #print(x.shape)
        #print(x.shape)

        return x

# test 

a = torch.LongTensor(
    [[1,2,3,8], [2,3,5,6], [3,3,8,3]]
    )

model = GPT()
#print(a.shape)

c = model(a)
print('aaaaaa',c.shape)
