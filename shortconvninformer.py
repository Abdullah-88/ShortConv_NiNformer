import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
        
class ShortConv(nn.Module):
    def __init__(self, dim,hidden_dim, kernel_size=4):
        super().__init__()
        self.dim = dim
        
        self.short_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim 
        )
      
        self.value_proj =  nn.Linear(dim,hidden_dim,bias=False)
        self.gate_proj = nn.Linear(dim,hidden_dim,bias=False)
        self.out_proj = nn.Linear(hidden_dim,dim,bias=False)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
      
        B, L, D = x.shape
        x = self.norm(x)       
        x_conv = x.transpose(1, 2)
        x_conv = self.short_conv(x_conv)[..., :L] 
        x_conv = x_conv.transpose(1, 2) 
               
        gate = self.gate_proj(x_conv)
        gate = self.gelu(gate)
        value = self.value_proj(x_conv)
                
        return self.out_proj(gate * value)
      
class MixerGatingUnit(nn.Module):
    def __init__(self,dim, hidden_dim):
        super().__init__()     
        self.Mixer = ShortConv(dim,hidden_dim)
        self.proj = nn.Linear(dim,dim)

    def forward(self, x):
        u, v = x, x 
        u = self.proj(u)  
        v = self.Mixer(v)
        out = u * v
        return out

class NiNBlock(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.mgu = MixerGatingUnit(d_model, d_ffn)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out

class NiNformer(nn.Module):
    def __init__(self, d_model, d_ffn, num_layers, dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[NiNBlock(d_model, d_ffn, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)
        