import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',32);d=c.get('input_shape',[8,6])[-1]
        self.encoder=nn.Linear(d,w);self.q=nn.Linear(w,w);self.k=nn.Linear(w,w);self.v=nn.Linear(w,w);self.norm=nn.LayerNorm(w)
        self.quality=nn.Parameter(torch.zeros(c.get('quality_dimensions',2)));self.wafer=nn.Linear(w,2);self.lot=nn.Linear(w,2)
    def forward(self,b):
        h=self.encoder(b['x'].float());available=b['available'].float();reliability=available*(1-b['uncertainty'].float()).clamp(0,1)*torch.exp(-(F.softplus(self.quality)*b['quality'].float()).sum(-1))
        allowed=b['adjacency'].bool();time=b['time'];cutoff=b['cutoff'][:,None];visible=time<=cutoff
        if self.c['variant']!='no_time_guard':
            visible=time<=cutoff;allowed=allowed&visible[:,:,None]&visible[:,None,:]&(time[:,:,None]<=time[:,None,:]);reliability=reliability*visible
        if self.c['variant'] in {'baseline','no_reliability'}:reliability=available*visible
        attention=None
        for _ in range(2):
            score=(self.q(h)@self.k(h).transpose(-1,-2))/h.shape[-1]**.5;score=score.masked_fill(~allowed.transpose(1,2),-1e4)
            attention=score.softmax(-1)*allowed.transpose(1,2);attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
            h=self.norm(h+attention@(self.v(h)*reliability[:,:,None]))
        weighted=(h*reliability[:,:,None]).sum(1)/reliability.sum(1,keepdim=True).clamp_min(1e-8)
        logits=self.wafer(weighted);lot=self.lot(weighted);path=attention*reliability[:,None,:]
        return {'probability':logits.softmax(-1),'logits':logits,'lot_logits':lot,'reliability':reliability,'path':path}
    def objective(self,b,p):
        probability=p['probability'][:,1];loss=F.cross_entropy(p['logits'],b['y'].long())+.5*F.cross_entropy(p['lot_logits'],b['lot_y'].long())+.1*(probability-b['y']).square().mean()
        if 'path_target' in b:loss=loss+.1*F.mse_loss(p['path'],b['path_target'].float())
        return loss


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,8,6)).astype('float32');y=(x.mean((1,2))>0).astype('int64')
    return {'x':x,'y':y,'lot_y':y.copy(),'adjacency':np.tile(np.triu(np.ones((8,8)),1),(n,1,1)).astype('float32'),'time':np.tile(np.arange(8),(n,1)).astype('float32'),
            'cutoff':np.full(n,7,np.float32),'available':np.ones((n,8),np.float32),'uncertainty':np.zeros((n,8),np.float32),'quality':np.zeros((n,8,2),np.float32),'group':np.arange(n)}


def measure(p,d,c):
    allowed=d['adjacency'].transpose(0,2,1)>0;mass=p['path'].sum();return {'path_order_consistency':float(p['path'][allowed].sum()/max(mass,1e-8)),'mean_source_reliability':float(p['reliability'].mean())}
