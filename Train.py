import torch
#import jieba
import os
import numpy as np

from Core.Datasets import MyDataset
from Core import CFG
from Core import func
from Core.ANEGPT import AppleNeuralEngineGPT



GPTconfig = CFG.OneGPT_Alpha#模型配置
Trainer = CFG.Trainer#模型训练器
Trainerconfig = CFG.TrainerCFG#训练配置
Sample = func.sample#示例


pp = 'C:\\Users\\xbj0916\\Desktop\\ANE-GPT-New-main\\datas'#str(input("输入您存放训练数据的文件夹目录："))
################
# 得到文本文件
def getFiles(dir, suffix): # 查找根目录，文件后缀 
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename) # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename)) # =>吧一串字符串组合成路径
    return res



  
txts = []
for file in getFiles(pp, '.txt'):  # =>查找以.txt结尾的文件
       with open(file, "r",encoding='utf-8') as f: 
            
            #打开文件
            data = f.read()   #读取文件
            txts.append(data)
        
f = ''.join(txts)#转化为非数组类型 



#######功能实现代码：

batch_size = int(input("\nplease inputs your batch_size:\n请输入您的要训练的batch_size这将取决于您显存的大小(如果您不确定请输入20):"))
epochs = int(input("\nEpochs:"))
#aa = jieba.lcut(f)
#print(aa)




# 数据集将会给模型为size(Batch_size,Block_size)的数据进行分批训练，从而避免了一口气之下直接将所有数据放入模型导致内存爆炸
# 使用Block_size来控制每批的数据长度一定也就是每批的数据特征值相同


train_dataset = MyDataset(f,20)
mconf = GPTconfig(vocab=train_dataset.vocab_size, block_size=train_dataset.block_size)
model = AppleNeuralEngineGPT(cfg=mconf)
print(model)

bar = "=="
print("{}START TRAIN{}".format(bar*19,bar*19))


tconf = Trainerconfig(max_epochs=epochs, batch_size=batch_size)
trainer = Trainer(model, train_dataset, test_dataset=None, cfg=tconf, saved_path='./')
trainer.train()




