"""Deterministic NumPy histogram Newton hazard boosting.

Cuts use the risk-trial matrix; no XGBoost or new fitting convention."""

import numpy as np


# Deterministic NumPy histogram Newton boosting

def sigmoid(x):
    x=np.asarray(x,float);return np.exp(-np.logaddexp(0,-x))


def quantize(x,cuts):return np.column_stack([np.searchsorted(v,x[:,j],side='left') for j,v in enumerate(cuts)]).astype(np.uint8)


def tree_predict(tree,bins):
    out=np.empty(len(bins))
    def walk(node,ix):
        if 'leaf' in node:out[ix]=node['leaf'];return
        left=bins[ix,node['feature']]<=node['bin'];walk(node['left'],ix[left]);walk(node['right'],ix[~left])
    walk(tree,np.arange(len(bins)));return out


def fit_tree(bins,g,h,depth, *, histogram_bins=32, leaf_l2=5.0, minimum_leaf_trials=200, minimum_split_gain=0.0):
    B=histogram_bins;P=bins.shape[1];minimum=minimum_leaf_trials
    def node(ix,remaining):
        G=float(g[ix].sum());H=float(h[ix].sum());leaf={'leaf':-G/(H+leaf_l2),'n':len(ix),'gradient_sum':G,'hessian_sum':H}
        if remaining==0 or len(ix)<2*minimum:return leaf
        best=None
        for j in range(P):
            z=bins[ix,j];count=np.bincount(z,minlength=B);gl=np.cumsum(np.bincount(z,weights=g[ix],minlength=B))[:-1];hl=np.cumsum(np.bincount(z,weights=h[ix],minlength=B))[:-1];nl=np.cumsum(count)[:-1]
            gain=.5*(gl**2/(hl+leaf_l2)+(G-gl)**2/(H-hl+leaf_l2)-G**2/(H+leaf_l2));gain[(nl<minimum)|((len(ix)-nl)<minimum)]=-np.inf;k=int(np.argmax(gain))
            if np.isfinite(gain[k]) and gain[k]>minimum_split_gain and (best is None or gain[k]>best[0]+1e-12):best=(float(gain[k]),j,k)
        if best is None:return leaf
        gain,j,k=best;left=bins[ix,j]<=k
        return {'feature':j,'bin':k,'gain':gain,'n':len(ix),'left':node(ix[left],remaining-1),'right':node(ix[~left],remaining-1)}
    return node(np.arange(len(bins)),depth)


def train(x,y,depth,learning_rate,max_rounds=None, *, histogram_bins=32, minimum_leaf_trials=200, leaf_l2=5.0, minimum_split_gain=0.0, minimum_fit_trials=500, minimum_fit_class_trials=20):
    x=np.asarray(x,float);y=np.asarray(y,float);maximum=max_rounds or 80
    if len(y)<minimum_fit_trials or min(y.sum(),len(y)-y.sum())<minimum_fit_class_trials:return {'status':'unsupported_training','n':len(y),'events':int(y.sum())}
    if not np.isfinite(x).all():raise ValueError('No missing/imputed tree inputs')
    cuts=[np.unique(np.quantile(x[:,j],np.arange(1,histogram_bins)/histogram_bins)).tolist() for j in range(x.shape[1])];bins=quantize(x,cuts)
    intercept=float(np.log(y.mean()/(1-y.mean())));eta=np.full(len(y),intercept);trees=[];history=[]
    for iteration in range(maximum):
        p=sigmoid(eta);tree=fit_tree(bins,p-y,p*(1-p),depth, histogram_bins=histogram_bins, leaf_l2=leaf_l2, minimum_leaf_trials=minimum_leaf_trials, minimum_split_gain=minimum_split_gain);eta+=learning_rate*tree_predict(tree,bins);trees.append(tree)
        history.append(float(np.mean(np.logaddexp(0,eta)-y*eta)))
    return {'status':'fit','n':len(y),'events':int(y.sum()),'depth':depth,'learning_rate':learning_rate,'maximum_rounds':maximum,'cuts':cuts,'intercept':intercept,'trees':trees,'unweighted_training_log_loss':history,'algorithm':'local_numpy_histogram_Newton_boosting'}


def predict(model,x,rounds=None):
    x=np.asarray(x,float)
    if model['status']!='fit':return np.full(len(x),np.nan)
    good=np.isfinite(x).all(axis=1);out=np.full(len(x),np.nan);b=quantize(x[good],model['cuts']);eta=np.full(len(b),model['intercept'])
    for tree in model['trees'][:rounds]:eta+=model['learning_rate']*tree_predict(tree,b)
    out[good]=sigmoid(eta);return out


def survival(model,x,rounds=None, *, maximum_hazard_hour=16):
    x=np.asarray(x,float);n=len(x);K=maximum_hazard_hour;design=np.c_[np.repeat(x,K,axis=0),np.tile(np.arange(1,K+1),n)];hazard=predict(model,design,rounds).reshape(n,K)
    return np.cumprod(1-hazard,axis=1),hazard


def importance(model,names):
    gain=np.zeros(len(names))
    def visit(t):
        if 'leaf' in t:return
        gain[t['feature']]+=t['gain'];visit(t['left']);visit(t['right'])
    if model['status']=='fit':
        for t in model['trees']:visit(t)
    return {n:float(v/gain.sum()) if gain.sum()>0 else 0. for n,v in zip(names,gain)}
