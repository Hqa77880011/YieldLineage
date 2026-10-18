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
    rng=np.random.default_rng(seed);packets=[];labels=[];lots=[];wafer_ids=[];cutoffs=[];event_times=[];groups=[]
    lot_risk=rng.normal(size=max(1,(n+11)//12))
    for wafer in range((n+2)//3):
        lot=wafer//4;latent=.6*lot_risk[lot]+.4*rng.normal();x=(latent+rng.normal(0,.4,(8,6))).astype('float32');y=int(latent>0)
        for cutoff in (2.,4.,6.):
            packets.append(x.copy());labels.append(y);lots.append(int(lot_risk[lot]>0));wafer_ids.append(wafer);cutoffs.append(cutoff);event_times.append(8. if y else np.inf);groups.append(lot)
    x=np.asarray(packets[:n]);return {'x':x,'y':np.array(labels[:n]),'lot_y':np.array(lots[:n]),'wafer':np.array(wafer_ids[:n]),'event_time':np.array(event_times[:n]),
            'adjacency':np.tile(np.triu(np.ones((8,8)),1),(n,1,1)).astype('float32'),'time':np.tile(np.arange(8),(n,1)).astype('float32'),
            'cutoff':np.array(cutoffs[:n],np.float32),'available':np.ones((n,8),np.float32),'uncertainty':np.zeros((n,8),np.float32),'quality':np.zeros((n,8,2),np.float32),'group':np.array(groups[:n])}


def measure(p,d,c):
    allowed=d['adjacency'].transpose(0,2,1)>0;mass=p['path'].sum();result={'path_order_consistency':float(p['path'][allowed].sum()/max(mass,1e-8)),'mean_source_reliability':float(p['reliability'].mean())}
    if 'event_time' in d:result.update(warning_metrics(p['probability'][:,1],d,c))
    return result


def warning_metrics(probability,data,config):
    """Confirm persistent warnings at observed cutoffs, then score event lead time."""
    if 'wafer' not in data:raise ValueError('Warning evaluation requires numeric wafer IDs')
    leads=[];events=detected=controls=false_alarms=0
    persistence=int(config.get('warning_persistence',2));threshold=float(config.get('warning_threshold',.5))
    if persistence<1 or not 0<=threshold<=1:raise ValueError('Invalid fixed warning rule')
    for wafer in np.unique(data['wafer']):
        ids=np.flatnonzero(data['wafer']==wafer);ids=ids[np.argsort(data['cutoff'][ids],kind='stable')]
        times=np.asarray(data['cutoff'][ids]);truth=np.asarray(data['event_time'][ids])
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or np.isnan(truth).any() or not np.all(truth==truth[0]):raise ValueError('Use increasing cutoffs and one event time per wafer')
        event=float(truth[0]);streak=0;alarm=None
        for index,time in zip(ids,times):
            if time>=event:break
            streak=streak+1 if probability[index]>=threshold else 0
            if streak>=persistence:alarm=float(time);break
        if np.isfinite(event):
            events+=1
            if alarm is not None:detected+=1;leads.append(event-alarm)
        else:
            controls+=1;false_alarms+=alarm is not None
    return {'warning_event_count':events,'warning_detected_event_count':detected,
            'warning_detection_fraction':detected/events if events else None,
            'warning_lead_time_mean':float(np.mean(leads)) if leads else None,
            'false_alarm_wafer_fraction':false_alarms/controls if controls else None}
