'''
Inspiration taken from https://github.com/neonbjb/tortoise-tts/blob/main/tortoise/models/autoregressive.py
'''
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import functools

from typing import Any
from torch.utils.data import Dataset,DataLoader
from transformers import GPT2Tokenizer,GPT2Config, GPT2Model, GPT2LMHeadModel
from tqdm import tqdm
from maha_tts.config import config
from maha_tts.text.symbols import labels,code_labels,text_labels
from maha_tts.models.modules import GST

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

class TS_model(nn.Module):
    def __init__(self,n_embed = 512, n_layer = 16, n_head = 8):
        super(TS_model,self).__init__()

        self.vocab_size=len(labels)
        self.n_positions=config.t2s_position
        self.n_embed=n_embed
        self.n_layer=n_layer
        self.n_head=n_head

        self.config = GPT2Config(vocab_size=self.vocab_size,n_positions=self.n_positions,n_embd=self.n_embed,n_layer=self.n_layer,n_head=self.n_head)
        self.gpt = GPT2Model(self.config)
        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=self.n_embed)
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.GST = GST(model_channels=self.n_embed,num_heads=self.n_head,in_channels=config.n_mel_channels,k=1)
        self.text_head = nn.Linear(self.n_embed,len(text_labels))
        self.code_head = nn.Linear(self.n_embed,len(code_labels))

        self.text_positional_embed = LearnedPositionEmbeddings(self.n_positions,self.n_embed)
        self.code_positional_embed = LearnedPositionEmbeddings(self.n_positions,self.n_embed)
        
        self.text_embed = nn.Embedding(len(text_labels),self.n_embed)
        self.code_embed = nn.Embedding(len(code_labels),self.n_embed)
        self.final_norm = nn.LayerNorm(self.n_embed)

    def get_speaker_latent(self, ref_mels):
        ref_mels = ref_mels.unsqueeze(1) if len(
            ref_mels.shape) == 3 else ref_mels

        conds = []
        for j in range(ref_mels.shape[1]):
            conds.append(self.GST(ref_mels[:, j,:,:]))

        conds = torch.cat(conds, dim=-1)
        conds = conds.mean(dim=-1)

        return conds.unsqueeze(1)

    def forward(self,text_ids,codes_ids = None,speaker_embed=None,ref_clips=None,return_loss = False):
        assert speaker_embed is not None or ref_clips is not None
        text_embed = self.text_embed(text_ids)
        text_embed += self.text_positional_embed(text_embed)

        code_embed = None
        code_probs= None

        if codes_ids is not None:
            code_embed = self.code_embed(codes_ids)
            code_embed+= self.code_positional_embed(code_embed)

        if ref_clips is not None:
            speaker_embed = self.get_speaker_latent(ref_clips)

        text_embed,code_embed = self.get_logits(speaker_embed=speaker_embed,text_embed=text_embed,code_embed=code_embed)

        text_probs = self.text_head(text_embed).permute(0,2,1)
        
        if codes_ids is not None:
            code_probs = self.code_head(code_embed).permute(0,2,1)

        if return_loss:
            loss_text = F.cross_entropy(text_probs[:,:,:-1], text_ids[:,1:].long(), reduce=False)
            loss_mel = F.cross_entropy(code_probs[:,:,:-1], codes_ids[:,1:].long(), reduce=False)
            return loss_text,loss_mel,code_probs
        
        return text_probs,code_probs


    def get_logits(self,speaker_embed,text_embed,code_embed=None):
        
        if code_embed is not None:
            embed = torch.cat([speaker_embed,text_embed,code_embed],dim=1)
        else:
            embed = torch.cat([speaker_embed,text_embed],dim=1)
        
        gpt_output = self.gpt(inputs_embeds=embed, return_dict=True)
        enc = gpt_output.last_hidden_state[:, 1:]
        enc = self.final_norm(enc)
        if code_embed is not None:
            return enc[:,:text_embed.shape[1]],enc[:,-code_embed.shape[1]:]
        
        return enc[:,:text_embed.shape[1]],None

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)

def load_TS_model(checkpoint,device):
    sem_model= TS_model(n_embed = 512, n_layer = 16, n_head = 8)
    sem_model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')),strict=False)
    sem_model.eval().to(device)

    return sem_model

if __name__ == '__main__':
    model=TS_model(n_embed = 256, n_layer = 6, n_head = 4)

    text_ids = torch.randint(0,100,(5,20))
    code_ids = torch.randint(0,100,(5,200))
    speaker_embed = torch.randn((5,1,256))

    output=model(text_ids=text_ids,speaker_embed=speaker_embed,codes_ids=code_ids,return_loss=True)