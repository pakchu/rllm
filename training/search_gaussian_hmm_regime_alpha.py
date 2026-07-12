"""Leak-safe Gaussian-HMM BTC regime alpha search.

The HMM is fit once on pre-2024 hourly observations.  Trading states use the
causal forward filter (never smoothed posteriors).  State direction is labelled
only from pre-2024 forward returns; 2024 ranks fixed variants and 2025/2026 are
report-only diagnostics.
"""
from __future__ import annotations
import argparse,itertools,json,sys
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
if __package__ is None or __package__=="":sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config,sim

SPLITS={"train":("2020-01-01","2024-01-01"),"test2024":("2024-01-01","2025-01-01"),"eval2025":("2025-01-01","2026-01-01"),"ytd2026":("2026-01-01","2026-06-02")}

def hourly_features(m):
 x=m.set_index(pd.to_datetime(m.date)).sort_index();q=x.quote_asset_volume.astype(float);b=x.taker_buy_quote.astype(float)
 h=pd.DataFrame({"open":x.open.resample('1h',closed='right',label='right').first(),"high":x.high.resample('1h',closed='right',label='right').max(),"low":x.low.resample('1h',closed='right',label='right').min(),"close":x.close.resample('1h',closed='right',label='right').last(),"quote":q.resample('1h',closed='right',label='right').sum(),"buy":b.resample('1h',closed='right',label='right').sum()}).dropna()
 r=np.log(h.close).diff();flow=2*h.buy/h.quote.replace(0,np.nan)-1
 f=pd.DataFrame(index=h.index);f['ret1']=r;f['trend24']=np.log(h.close/h.close.shift(24));f['trend72']=np.log(h.close/h.close.shift(72));f['vol24']=r.rolling(24).std();f['vol168']=r.rolling(168).std();f['volterm']=f.vol24/f.vol168.replace(0,np.nan);f['range24']=(h.high.rolling(24).max()-h.low.rolling(24).min())/h.close;f['flow24']=flow.rolling(24).mean();f['volume_z']=(np.log1p(h.quote)-np.log1p(h.quote).rolling(168).mean())/np.log1p(h.quote).rolling(168).std().replace(0,np.nan)
 return h,f.replace([np.inf,-np.inf],np.nan)
def emission(x,mu,var):return -.5*(np.log(2*np.pi*var).sum(1)[None,:]+(((x[:,None,:]-mu[None,:,:])**2)/var[None,:,:]).sum(2))
def emission_prob(x,mu,var):
 lb=emission(x,mu,var);lb-=lb.max(1,keepdims=True);return np.exp(lb)
def forward_prob(B,pi,A):
 n,k=B.shape;alpha=np.empty((n,k));scale=np.empty(n);alpha[0]=pi*B[0];scale[0]=alpha[0].sum()+1e-300;alpha[0]/=scale[0]
 for t in range(1,n):alpha[t]=(alpha[t-1]@A)*B[t];scale[t]=alpha[t].sum()+1e-300;alpha[t]/=scale[t]
 return alpha,scale
def fit_hmm(x,k,seed=712,max_iter=12):
 labels=KMeans(k,n_init=10,random_state=seed).fit_predict(x);mu=np.array([x[labels==j].mean(0) for j in range(k)]);var=np.array([x[labels==j].var(0)+1e-3 for j in range(k)]);A=np.ones((k,k))
 for a,b in zip(labels[:-1],labels[1:]):A[a,b]+=1
 A/=A.sum(1,keepdims=True);pi=np.bincount(labels[:min(500,len(labels))],minlength=k)+1;pi=pi/pi.sum();last=None
 for _ in range(max_iter):
  B=emission_prob(x,mu,var);alpha,scale=forward_prob(B,pi,A);n=len(x);beta=np.ones_like(alpha)
  for t in range(n-2,-1,-1):beta[t]=A@(B[t+1]*beta[t+1]);beta[t]/=beta[t].sum()+1e-300
  g=alpha*beta;g/=g.sum(1,keepdims=True)+1e-300;xi=np.zeros_like(A)
  for t in range(n-1):
   z=alpha[t][:,None]*A*(B[t+1]*beta[t+1])[None,:];xi+=z/(z.sum()+1e-300)
  pi=g[0]+1e-6;pi/=pi.sum();A=xi+1e-4;A/=A.sum(1,keepdims=True);w=g.sum(0)+1e-9;mu=(g.T@x)/w[:,None];var=np.array([((g[:,j,None]*(x-mu[j])**2).sum(0)/w[j]).clip(1e-3,None) for j in range(k)])
  ll=float(np.log(scale+1e-300).sum())
  if last is not None and abs(ll-last)<1e-3:break
  last=ll
 return {"pi":pi,"A":A,"mu":mu,"var":var,"loglik":last}
def filtered(x,model):
 p,_=forward_prob(emission_prob(x,model['mu'],model['var']),model['pi'],model['A']);return p
def run(cfg):
 m=_load_market(cfg);h,f=hourly_features(m);train=(f.index>=SPLITS['train'][0])&(f.index<SPLITS['train'][1]);feature_sets={"core":['ret1','trend24','vol24','volterm'],"flow":['ret1','trend24','trend72','vol24','volterm','flow24'],"liquidity":['ret1','trend24','vol24','range24','flow24','volume_z']};rows=[];signal_cache={}
 for fs,cols in feature_sets.items():
  good=f[cols].notna().all(1);fitmask=train&good;raw=f.loc[fitmask,cols].to_numpy();mean=raw.mean(0);std=raw.std(0);std[std<1e-8]=1;allx=((f.loc[good,cols].to_numpy()-mean)/std).clip(-8,8)
  for k in (3,4,5):
   model=fit_hmm(((raw-mean)/std).clip(-8,8),k);prob=filtered(allx,model);state=prob.argmax(1);conf=prob.max(1);dates=f.index[good];close=h.close.reindex(dates);train_good=(dates>=SPLITS['train'][0])&(dates<SPLITS['train'][1])
   for horizon in (6,12,24,48):
    future=np.log(close.shift(-horizon)/close).to_numpy();state_edge=[];state_n=[]
    for j in range(k):
     use=train_good&(state==j)&np.isfinite(future);state_edge.append(float(np.mean(future[use])) if use.sum() else 0.0);state_n.append(int(use.sum()))
    for min_conf,min_edge,hold in itertools.product((.4,.5,.6,.7),(0,.0005,.001,.002),(72,144,288)):
     sides=np.array([1 if e>=min_edge else -1 if e<=-min_edge else 0 for e in state_edge]);pred=sides[state];hour_long=(pred>0)&(conf>=min_conf);hour_short=(pred<0)&(conf>=min_conf)
     posmap=pd.Series(np.arange(len(m)),index=pd.to_datetime(m.date));idx=posmap.reindex(dates).dropna().astype(int);la=np.zeros(len(m),bool);sa=np.zeros(len(m),bool);valid=idx.index;la[idx.to_numpy()]=hour_long[np.isin(dates,valid)];sa[idx.to_numpy()]=hour_short[np.isin(dates,valid)]
     key=f'{fs}|{k}|{horizon}|{min_conf}|{min_edge}';signal_cache[key]=(la,sa);t=sim(m,pd.to_datetime(m.date),la,sa,cfg,hold,1,.025,.015,'test2024')
     if t['trades']>=30 and t['longs']>=5 and t['shorts']>=5:rows.append({'feature_set':fs,'states':k,'label_horizon_hours':horizon,'min_confidence':min_conf,'min_state_edge':min_edge,'hold_bars':hold,'signal_key':key,'state_train_edge':state_edge,'state_train_n':state_n,'transition_matrix':model['A'].tolist(),'stats':{'test2024':t}})
 rows.sort(key=lambda r:(r['stats']['test2024']['ratio'],r['stats']['test2024']['return_pct']),reverse=True);sel=rows[:250]
 for r in sel:
  la,sa=signal_cache[r.pop('signal_key')]
  for sp in ('train','eval2025','ytd2026'):r['stats'][sp]=sim(m,pd.to_datetime(m.date),la,sa,cfg,r['hold_bars'],1,.025,.015,sp)
  e=r['stats']['eval2025'];y=r['stats']['ytd2026'];r['passes_alpha_pool']=bool(e['trades']>=20 and r['stats']['test2024']['ratio']>=2.5 and e['ratio']>=2.5);r['passes_live_grade']=bool(r['passes_alpha_pool'] and y['trades']>=8 and y['ratio']>=5)
 out={'as_of':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'protocol':'Gaussian diagonal HMM fit pre-2024; causal filtered posterior; train-only state direction; test2024 rank; sealed eval/2026; 6bp/side; strict MDD','tested':len(rows),'selected':sel,'alpha_pool_qualifiers':[r for r in sel if r['passes_alpha_pool']],'live_grade':[r for r in sel if r['passes_live_grade']]};Path(cfg.output).write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-csv',required=True);p.add_argument('--output',required=True);p.add_argument('--funding-csv',default='');p.add_argument('--premium-csv',default='');p.add_argument('--exclude-from',default='2026-06-02');a=p.parse_args();o=run(Config(**vars(a)));print(json.dumps({'tested':o['tested'],'qualifiers':len(o['alpha_pool_qualifiers']),'live':len(o['live_grade']),'top':o['selected'][:3]},indent=2,ensure_ascii=False,default=str))
if __name__=='__main__':main()
