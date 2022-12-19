import os 
import math
import torch 
import numpy as np 


from torch import nn 
from tqdm import tqdm
from torch.nn import functional
from torch.utils.data import DataLoader


class GPTBaseCFG :
    
    embed_dim=512
    ffn_dim=2048
    nb_dec_layers=6
    nb_attention_heads=8
    dropout=0.1
    return_intermediate_decoder_layers=False

    def __init__(self, vocab, block_size, **kwargs):
        self.vocab = vocab
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)



class OneGPT_Alpha(GPTBaseCFG) :
    
    embed_dim = 512
    nb_attention_heads = 8




class TrainerCFG :
    # optimization parameters
    Epochs = 5
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)


    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)



class Trainer :
    def __init__(self, model, train_dataset, test_dataset, cfg, saved_path) :
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.cfg = cfg
        self.saved = saved_path

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda' :
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self) :
        model = self.model
        cfg = self.cfg

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        def run_epoch(split:str) :
            is_train = split == 'train'
            model.train(is_train)

            data = self.train_dataset if is_train else self.test_dataset

            loader = DataLoader(
                data,
                shuffle=True, 
                pin_memory=True,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for index, (x, y) in pbar :
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train) :
                    out = model(x)
                    loss = loss = functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train :

                    model.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
                    optimizer.step()
    
        best_loss = float('inf')# 最佳损失

        for epoch in range(cfg.Epochs) :
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
            
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss

            if self.saved is not None and good_model:
                self.save_checkpoint()
        
    def save_checkpoint(self) :
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), 'model.bin')
        #saved_model = torch.jit.script(self.model)
        #saved_model.save('model_jit.bin')





