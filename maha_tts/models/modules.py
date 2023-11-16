import torch,math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    Using it for Zero Convolutions
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer. of groups ranging from 2 to 32.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


class mySequential(nn.Sequential):
    '''Using this to pass mask variable to nn layers
    '''
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class SepConv1D(nn.Module):
    '''Depth wise separable Convolution layer with mask
    '''
    def __init__(self,nin,nout,kernel_size,stride=1,dilation=1,padding_mode='same',bias=True):
        super(SepConv1D,self).__init__()
        self.conv1=nn.Conv1d(nin, nin, kernel_size=kernel_size, stride=stride,groups=nin,dilation=dilation,padding=padding_mode,bias=bias)
        self.conv2=nn.Conv1d(nin,nout,kernel_size=1,stride=1,padding=padding_mode,bias=bias)

    def forward(self,x,mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        x=self.conv1(x)
        x=self.conv2(x)
        return x,mask

class Conv1DBN(nn.Module):
    def __init__(self,nin,nout,kernel_size,stride=1,dilation=1,dropout=0.1,padding_mode='same',bias=False):
        super(Conv1DBN,self).__init__()
        self.conv1=nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride,padding=padding_mode,dilation=dilation,bias=bias)
        self.bn=nn.BatchNorm1d(nout)
        self.drop=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        x=self.conv1(x)
        x=self.bn(x)
        x=F.relu(x)
        x=self.drop(x)
        return x,mask

class Conv1d(nn.Module):
    '''normal conv1d with mask
    '''
    def __init__(self,nin,nout,kernel_size,padding,bias=True):
        super(Conv1d,self).__init__()
        self.l=nn.Conv1d(nin,nout,kernel_size,padding=padding,bias=bias)
    def forward(self,x,mask):
        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        y=self.l(x)
        return y,mask
    
class SqueezeExcite(nn.Module):
    '''Let the CNN decide how to add across channels
    '''
    def __init__(self,nin,ratio=8):
        super(SqueezeExcite,self).__init__()
        self.nin=nin
        self.ratio=ratio

        self.fc=mySequential(
            nn.Linear(nin,nin//ratio,bias=True),nn.SiLU(inplace=True),nn.Linear(nin//ratio,nin,bias=True)
        )

    def forward(self,x,mask=None):
        if mask is None:
            mask = torch.ones((x.shape[0],x.shape[-1]),dtype=torch.bool).to(x.device)
        mask=~mask
        x=x.float()
        x.masked_fill_(mask.unsqueeze(1), 0.0)
        mask=~mask
        y = (torch.sum(x, dim=-1, keepdim=True) / mask.unsqueeze(1).sum(dim=-1, keepdim=True)).type(x.dtype)
        # y=torch.mean(x,-1,keepdim=True)
        y=y.transpose(1, -1)
        y=self.fc(y)
        y=torch.sigmoid(y)
        y=y.transpose(1, -1)
        y= x * y
        return y,mask



class SCBD(nn.Module):
    '''SeparableConv1D + Batchnorm + Dropout, Generally use it for middle layers and resnet
    '''
    def __init__(self,nin,nout,kernel_size,p=0.1,rd=True,separable=True,bias=True):
        super(SCBD,self).__init__()
        if separable:
            self.SC=SepConv1D(nin,nout,kernel_size,bias=bias)
        else:
            self.SC=Conv1d(nin,nout,kernel_size,padding='same',bias=bias)

        if rd: #relu and Dropout
            self.mout=mySequential(normalization(nout),nn.SiLU(), # nn.BatchNorm1d(nout,eps)
                nn.Dropout(p))
        else:
            self.mout=normalization(nout) # nn.BatchNorm1d(nout,eps)

    def forward(self,x,mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        x,_= self.SC(x,mask)
        y = self.mout(x)
        return y,mask
    
class QuartzNetBlock(nn.Module):
    '''Similar to Resnet block with Batchnorm and dropout, and using Separable conv in the middle.
    if its the last layer,set se = False and separable = False, and use a projection layer on top of this.
    '''
    def __init__(self,nin,nout,kernel_size,dropout=0.1,R=5,se=False,ratio=8,separable=False,bias=True):
        super(QuartzNetBlock,self).__init__()
        self.se=se
        self.residual=mySequential(
            nn.Conv1d(nin,nout,kernel_size=1,padding='same',bias=bias),
            normalization(nout) #nn.BatchNorm1d(nout,eps)
        )
        model=[]

        for i in range(R-1):
            model.append(SCBD(nin,nout,kernel_size,dropout,eps=0.001,bias=bias))
            nin=nout

        if separable:
            model.append(SCBD(nin,nout,kernel_size,dropout,eps=0.001,rd=False,bias=bias))
        else:
            model.append(SCBD(nin,nout,kernel_size,dropout,eps=0.001,rd=False,separable=False,bias=bias))
        self.model=mySequential(*model)

        if self.se:
            self.se_layer=SqueezeExcite(nin,ratio)

        self.mout= mySequential(nn.SiLU(),nn.Dropout(dropout))

    def forward(self,x,mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        y,_=self.model(x,mask)
        if self.se:
            y,_=self.se_layer(y,mask)
        y+=self.residual(x)
        y=self.mout(y)
        return y,mask

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)
    
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0   
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1)) # no effect of attention in the inital stages.
        # if relative_pos_embeddings: 
        self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64) #need to read about this, vit and swin transformers
        # self.relative_pos_embeddings = FixedPositionalEmbedding(dim=channels)
        # else:
        # self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        pos_emb = self.emb(n)
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        return pos_emb * self.scale


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return rearrange(emb, 'n d -> () n d')
    
class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)



class MultiHeadAttention(nn.Module):
    '''
    only for GST
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
        

class GST(nn.Module):
    def __init__(self,model_channels=512,num_heads=8,in_channels=80,k=2):
        super(GST,self).__init__()
        self.model_channels=model_channels
        self.num_heads=num_heads

        self.reference_encoder=nn.Sequential(
            nn.Conv1d(in_channels,model_channels,3,padding=1,stride=2),
            nn.Conv1d(model_channels, model_channels*k,3,padding=1,stride=2),
            AttentionBlock(model_channels*k, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
            AttentionBlock(model_channels*k, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
            AttentionBlock(model_channels*k, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
            AttentionBlock(model_channels*k, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
            AttentionBlock(model_channels*k, num_heads, relative_pos_embeddings=True, do_checkpoint=False)
        )

    def forward(self,x):
        x=self.reference_encoder(x)
        return x
    

if __name__ == '__main__':
    device = torch.device('cpu')
    m = GST(512,10).to(device)
    mels = torch.rand((16,80,1000)).to(device)

    o = m(mels)
    print(o.shape,'final output')

    from torchinfo import summary
    summary(m, input_data={'x': torch.randn(16,80,500).to(device)})