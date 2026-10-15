import importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,region,image_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',32);self.encoder=Encoder(1,w);self.mean=nn.Conv2d(w,1,1);self.basis=nn.Conv2d(w,c.get('rank',4),1)
    def forward(self,b):
        h=self.encoder(b['x'].float());mean=self.mean(h);basis=F.interpolate(F.adaptive_avg_pool2d(self.basis(h),(8,8)),mean.shape[-2:],mode='bilinear',align_corners=False)
        if self.c['variant'] in {'baseline','no_spatial_basis'}:basis=torch.zeros_like(basis)
        variance=basis.square().sum(1,keepdim=True)+self.c.get('annotation_noise',.8)**2;probability=torch.special.ndtr(-mean/variance.sqrt())
        return {'probability':probability,'sdf':mean,'basis':basis,'variance':variance}
    def objective(self,b,p):
        loss=region(p['probability'],b['y'].float())+F.l1_loss(p['sdf'],b['sdf'].float())
        if self.c['variant'] not in {'baseline','no_likelihood'}:
            count=min(64,p['sdf'][0].numel());index=torch.linspace(0,p['sdf'][0].numel()-1,count,device=p['sdf'].device).long();mean=p['sdf'].flatten(1)[:,index];basis=p['basis'].flatten(2)[:,:,index].transpose(1,2)
            covariance=basis@basis.transpose(1,2)+self.c.get('annotation_noise',.8)**2*torch.eye(count,device=mean.device);distribution=torch.distributions.MultivariateNormal(mean,covariance_matrix=covariance);loss=loss-.1*distribution.log_prob(b['sdf'].flatten(1)[:,index]).mean()/count
        return loss


def demo(seed,n,c):
    from scipy.ndimage import distance_transform_edt
    d=image_demo(seed,n,1);d['sdf']=np.array([[distance_transform_edt(1-y[0])-distance_transform_edt(y[0])] for y in d['y']],np.float32);return d


def posterior(problem,adapter):
    from scipy.special import logsumexp
    rng=np.random.default_rng(problem.get('seed',13));bounds=np.asarray(problem['bounds']);draws=np.asarray(problem['boundary_draws']);cov=np.asarray(problem['covariance']);inverse=np.linalg.inv(cov);sign,logdet=np.linalg.slogdet(cov)
    if sign<=0:raise ValueError('Geometric covariance must be positive definite')
    def density(theta):
        if np.any(theta<bounds[:,0]) or np.any(theta>bounds[:,1]):return -np.inf
        difference=draws-np.asarray(adapter.geometry(theta));log=-.5*np.einsum('mi,ij,mj->m',difference,inverse,difference)-.5*(logdet+draws.shape[1]*np.log(2*np.pi));return logsumexp(log)-np.log(len(draws))
    chains=[];acceptance=[]
    for chain in range(problem.get('chains',4)):
        current=rng.uniform(bounds[:,0],bounds[:,1]);value=density(current);step=.05;states=[];accepted=0
        for iteration in range(problem.get('warmup',1000)+problem.get('retained',2000)):
            proposed=current+rng.normal(size=len(current))*step*(bounds[:,1]-bounds[:,0]);score=density(proposed)
            yes=np.log(rng.random())<score-value
            if yes:current,value=proposed,score;accepted+=1
            if iteration<problem.get('warmup',1000):step*=np.exp((float(yes)-.25)/np.sqrt(iteration+10))
            else:states.append(current.copy())
        chains.append(states);acceptance.append(accepted/(problem.get('warmup',1000)+problem.get('retained',2000)))
    chains=np.asarray(chains);flat=chains.reshape(-1,chains.shape[-1]);index=np.linspace(0,len(flat)-1,problem.get('propagate',128)).astype(int);curves=np.stack([adapter.device(theta) for theta in flat[index]])
    voltage=np.asarray(problem['voltage']);metrics=[]
    for curve in curves:
        log=np.log10(np.maximum(abs(curve),1e-30));gradient=np.gradient(log,voltage);metrics.append([float(np.interp(problem.get('threshold_current',1e-7),curve,voltage)),float(1000/max(gradient.max(),1e-8)),float(np.interp(problem.get('on_voltage',.7),voltage,curve))])
    metrics=np.asarray(metrics);within=chains.var(1,ddof=1).mean(0);between=chains.shape[1]*chains.mean(1).var(0,ddof=1);variance=(chains.shape[1]-1)/chains.shape[1]*within+between/chains.shape[1]
    import arviz as az
    diagnostic=az.from_dict(posterior={'theta':chains});rhat=az.rhat(diagnostic,method='rank')['theta'].values;bulk=az.ess(diagnostic,method='bulk')['theta'].values;tail=az.ess(diagnostic,method='tail')['theta'].values
    return chains,curves,{'rank_normalized_rhat':rhat.tolist(),'bulk_ess':bulk.tolist(),'tail_ess':tail.tolist(),'diagnostic_pass':bool(np.all(rhat<1.01) and np.all(bulk>400) and np.all(tail>400)),'parameter_mean':flat.mean(0).tolist(),'acceptance':acceptance,'classical_rhat':np.sqrt(variance/np.maximum(within,1e-12)).tolist(),'electrical_mean':metrics.mean(0).tolist(),'electrical_lower':np.quantile(metrics,.05,axis=0).tolist(),'electrical_upper':np.quantile(metrics,.95,axis=0).tolist()}


def execute(args):
    import json
    from pathlib import Path
    problem=json.loads(Path(args.input).read_text());spec=importlib.util.spec_from_file_location('device_adapter',problem['adapter']);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    chains,curves,result=posterior(problem,adapter)
    if 'reference_curve' in problem:
        reference=np.asarray(problem['reference_curve']);result['iv_nrmse_percent']=float(100*np.sqrt(np.sum((curves.mean(0)-reference)**2)/max(np.sum(reference**2),1e-30)))
    if 'reference_metrics' in problem:
        target=np.asarray(problem['reference_metrics']);low=np.asarray(result['electrical_lower']);high=np.asarray(result['electrical_upper']);result['electrical_coverage90']=((target>=low)&(target<=high)).tolist();result['interval_score90']=(high-low+20*np.maximum(low-target,0)+20*np.maximum(target-high,0)).tolist()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'posterior.npz',chains=chains,iv_curves=curves);(out/'electrical.json').write_text(json.dumps(result,indent=2))
