'''
inspiration taken from https://github.com/neonbjb/tortoise-tts/blob/main/tortoise/models/diffusion_decoder.py
'''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from maha_tts.config import config
from torch import autocast
from maha_tts.models.modules import QuartzNetBlock,AttentionBlock,mySequential,normalization,SCBD,SqueezeExcite,GST

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class QuartzNetBlock(TimestepBlock):
    '''Similar to Resnet block with Batchnorm and dropout, and using Separable conv in the middle.
    if its the last layer,set se = False and separable = False, and use a projection layer on top of this.
    '''
    def __init__(self,nin,nout,emb_channels,kernel_size=3,dropout=0.1,R=1,se=True,ratio=8,separable=False,bias=True,use_scale_shift_norm=True):
        super(QuartzNetBlock,self).__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.se=se
        self.in_layers = mySequential(
            nn.Conv1d(nin,nout,kernel_size=1,padding='same',bias=bias),
            normalization(nout) #nn.BatchNorm1d(nout,eps)
        )

        self.residual=mySequential(
            nn.Conv1d(nin,nout,kernel_size=1,padding='same',bias=bias),
            normalization(nout) #nn.BatchNorm1d(nout,eps)
        )

        nin=nout
        model=[]

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * nout if use_scale_shift_norm else nout,
            ),
        )

        for i in range(R-1):
            model.append(SCBD(nin,nout,kernel_size,dropout,bias=bias))
            nin=nout

        if separable:
            model.append(SCBD(nin,nout,kernel_size,dropout,rd=False,bias=bias))
        else:
            model.append(SCBD(nin,nout,kernel_size,dropout,rd=False,separable=False,bias=bias))

        self.model=mySequential(*model)
        if self.se:
            self.se_layer=SqueezeExcite(nin,ratio)

        self.mout= mySequential(nn.SiLU(),nn.Dropout(dropout))

    def forward(self,x,emb,mask=None):
        x_new=self.in_layers(x)
        emb = self.emb_layers(emb)
        while len(emb.shape) < len(x_new.shape):
            emb = emb[..., None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x_new = x_new * (1 + scale) + shift
        y,_=self.model(x_new)

        if self.se:
            y,_=self.se_layer(y,mask)
        y+=self.residual(x)
        y=self.mout(y)

        return y

class QuartzAttn(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = QuartzNetBlock(model_channels, model_channels, model_channels,dropout=dropout,use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)

class QuartzNet9x5(nn.Module):
    def __init__(self,model_channels,num_heads,enable_fp16=False):
        super(QuartzNet9x5,self).__init__()
        self.enable_fp16 = enable_fp16

        self.conv1=QuartzNetBlock(model_channels,model_channels,model_channels,kernel_size=3,dropout=0.1,R=3)
        kernels=[5,7,9,13,15,17]
        quartznet=[]
        attn=[]
        for i in kernels:
            quartznet.append(QuartzNetBlock(model_channels,model_channels,model_channels,kernel_size=i,dropout=0.1,R=5,se=True))
            attn.append(AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True))
        kernels=[21,23,25]
        quartznet.append(QuartzNetBlock(model_channels,model_channels,model_channels,kernel_size=21,dropout=0.1,R=5,se=True))
        attn.append(AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True))

        for i in kernels[1:]:
            quartznet.append(QuartzNetBlock(model_channels,model_channels,model_channels,kernel_size=i,dropout=0.1,R=5,se=True))
            attn.append(AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True))
        self.quartznet= nn.ModuleList(quartznet)
        self.attn = nn.ModuleList(attn)
        self.conv3=nn.Conv1d(model_channels, model_channels, 1, padding='same')


    def forward(self, x, time_emb):
        x = self.conv1(x,time_emb)
        # with autocast(x.device.type, enabled=self.enable_fp16):
        for n,(layer,attn) in enumerate(zip(self.quartznet,self.attn)):
            x = layer(x,time_emb) #256 dim
            x = attn(x)
        x = self.conv3(x.float())
        return x

class DiffModel(nn.Module):

    def __init__(
        self,
        input_channels=80,
        output_channels=160,
        model_channels=512,
        num_heads=8,
        dropout=0.0,
        multispeaker = True,
        condition_free_per=0.1,
        training = False,
        ar_active = False,
        in_latent_channels = 10004
    ):

        super().__init__()
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.condition_free_per = condition_free_per
        self.training = training
        self.multispeaker = multispeaker
        self.ar_active = ar_active
        self.in_latent_channels = in_latent_channels

        if not self.ar_active:
            self.code_emb = nn.Embedding(config.semantic_model_centroids+1,model_channels)
            self.code_converter = mySequential(
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            )
        else:
            self.code_converter = mySequential(
                nn.Conv1d(self.in_latent_channels, model_channels, 3, padding=1),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
                AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            )
        if self.multispeaker:
            self.GST = GST(model_channels,num_heads)

        self.code_norm = normalization(model_channels)
        self.time_norm = normalization(model_channels)
        self.noise_norm = normalization(model_channels)
        self.code_time_norm = normalization(model_channels)
        
        # self.code_latent = []
        self.time_embed = mySequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),)
        
        self.input_block = nn.Conv1d(input_channels,model_channels,3,1,1)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))

        self.code_time = TimestepEmbedSequential(QuartzAttn(model_channels, dropout, num_heads),QuartzAttn(model_channels, dropout, num_heads),QuartzAttn(model_channels, dropout, num_heads))
        self.layers = QuartzNet9x5(model_channels,num_heads)

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, output_channels, 3, padding=1),
        )
    
    def get_speaker_latent(self, ref_mels):
        ref_mels = ref_mels.unsqueeze(1) if len(
            ref_mels.shape) == 3 else ref_mels

        conds = []
        for j in range(ref_mels.shape[1]):
            conds.append(self.GST(ref_mels[:, j,:,:]))

        conds = torch.cat(conds, dim=-1)
        conds = conds.mean(dim=-1)

        return conds.unsqueeze(2)

    def forward(self ,x,t,code_emb,ref_clips=None,speaker_latents=None,conditioning_free=False):
        time_embed = self.time_norm(self.time_embed(timestep_embedding(t.unsqueeze(-1),self.model_channels)).permute(0,2,1)).squeeze(2)
        if conditioning_free:
            code_embed = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
        else:
            if not self.ar_active:
                code_embed = self.code_norm(self.code_converter(self.code_emb(code_emb).permute(0,2,1)))
            else:
                code_embed = self.code_norm(self.code_converter(code_emb))
        if self.multispeaker:
            assert speaker_latents is not None or ref_clips is not None
            if ref_clips is not None:
                speaker_latents = self.get_speaker_latent(ref_clips)
            cond_scale, cond_shift = torch.chunk(speaker_latents, 2, dim=1)
            code_embed = code_embed * (1 + cond_scale) + cond_shift
        if self.training and self.condition_free_per > 0:
            unconditioned_batches = torch.rand((code_embed.shape[0], 1, 1),
                                               device=code_embed.device) < self.condition_free_per
            code_embed = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(code_embed.shape[0], 1, 1),
                                   code_embed)

        expanded_code_emb = F.interpolate(code_embed, size=x.shape[-1], mode='nearest') #try different modes
        
        x_cond = self.code_time_norm(self.code_time(expanded_code_emb,time_embed))

        x = self.noise_norm(self.input_block(x))
        x += x_cond
        x = self.layers(x, time_embed)
        out = self.out(x)
        return out

def load_diff_model(checkpoint,device,model_channels=512,ar_active=False,len_code_labels=10004):
    diff_model = DiffModel(input_channels=80,
                 output_channels=160,
                 model_channels=512,
                 num_heads=8,
                 dropout=0.15,
                 condition_free_per=0.15,
                 multispeaker=True,
                 training=False,
                 ar_active=ar_active,
                 in_latent_channels = len_code_labels)

    # diff_model.load_state_dict(torch.load('/content/LibriTTS_fp64_10k/S2A/_latest.pt',map_location=torch.device('cpu')),strict=True)
    diff_model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')),strict=True)
    diff_model=diff_model.eval().to(device)
    return diff_model


if __name__ == '__main__':

    device = torch.device('cpu')
    diff_model = DiffModel(input_channels=80,
                 output_channels=160,
                 model_channels=1024,
                 num_heads=8,
                 dropout=0.1,
                 num_layers=8,
                 enable_fp16=True,
                 condition_free_per=0.1,
                 multispeaker=True,
                 training=True).to(device)

    batch_Size = 32
    timeseries = 800
    from torchinfo import summary
    summary(diff_model, input_data={'x': torch.randn(batch_Size, 80, timeseries).to(device),
    'ref_clips': torch.randn(batch_Size,3, 80, timeseries).to(device),
    't':torch.LongTensor(size=[batch_Size,]).to(device),
    'code_emb':torch.randint(0,201,(batch_Size,timeseries)).to(device)})