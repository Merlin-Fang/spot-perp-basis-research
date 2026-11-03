"""Registered hourly and coarse unweighted logistic hazard estimators.

fit/predict are the matched hourly model: predict returns convergence then
hazard. logistic_fit/hazard_predict retain the original coarse model, whose
first prediction is survival. Their support gates and optimizers differ."""

import numpy as np


# Matched hourly logistic: cumulative event probability

def sigmoid(x):return np.exp(-np.logaddexp(0,-np.asarray(x,float)))


def fit(x,bins,y,alpha,K,mean=None,scale=None, *, maximum_iterations=80, gradient_tolerance=1e-7):
    x=np.asarray(x,float);y=np.asarray(y,float);bins=np.asarray(bins,int)
    if len(y)<500 or min(y.sum(),len(y)-y.sum())<20:return {'status':'unsupported_training_support','n':len(y),'events':int(y.sum()),'alpha':alpha,'K':K}
    if not np.isfinite(x).all():raise ValueError('No missing/imputed training inputs')
    mean=x.mean(axis=0) if mean is None else np.asarray(mean,float);scale=x.std(axis=0) if scale is None else np.asarray(scale,float);scale=np.where(scale>1e-12,scale,1.)
    Z=np.c_[np.ones(len(y)),np.eye(K)[bins,1:],(x-mean)/scale];penalty=np.r_[0.,np.full(Z.shape[1]-1,alpha)];b=np.zeros(Z.shape[1]);b[0]=np.log(y.mean()/(1-y.mean()))
    def objective(beta):
        eta=Z@beta;return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty*beta**2))
    converged=False
    for iteration in range(maximum_iterations):
        p=sigmoid(Z@b);gradient=Z.T@(p-y)/len(y)+penalty*b
        if np.max(np.abs(gradient))<gradient_tolerance:converged=True;break
        H=Z.T@((p*(1-p))[:,None]*Z)/len(y)+np.diag(penalty);step=np.linalg.solve(H+np.eye(len(b))*1e-12,gradient);fraction=1.;previous=objective(b)
        while fraction>1e-8 and objective(b-fraction*step)>previous+1e-12:fraction*=.5
        b-=fraction*step
    return {'status':'fit' if converged else 'unsupported_numerical_nonconvergence','n':len(y),'events':int(y.sum()),'alpha':alpha,'K':K,'mean':mean.tolist(),'scale':scale.tolist(),'beta':b.tolist(),'maximum_gradient':float(np.max(np.abs(gradient))),'iterations':iteration+1,'objective':objective(b),'unweighted_likelihood':True}


def predict(model,x):
    x=np.asarray(x,float);K=model['K'];out=np.full((len(x),K),np.nan);hazard=out.copy()
    if model['status']!='fit':return out,hazard
    good=np.isfinite(x).all(axis=1);z=(x[good]-np.array(model['mean']))/np.array(model['scale']);b=np.array(model['beta']);eta=b[0]+np.r_[0.,b[1:K]][None,:]+z@b[K:][:,None];q=sigmoid(eta);hazard[good]=q;out[good]=1-np.cumprod(1-q,axis=1);return out,hazard


# Original coarse logistic: survival probability; distinct support rules

def logistic_fit(x,bin_index,y,alpha, *, score_grid_hours=(1,2,4,8,16,32,64,128), maximum_iterations=60):
    x=np.asarray(x,float);y=np.asarray(y,float);bins=np.asarray(bin_index,int);J=len(score_grid_hours)
    if len(y)<100 or min(y.sum(),len(y)-y.sum())<10:return {'status':'unsupported_training_support','n':len(y),'events':int(y.sum())}
    mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
    z=np.c_[np.ones(len(y)),np.eye(J)[bins,1:],(x-mean)/scale]
    penalty=np.r_[0.,np.full(z.shape[1]-1,alpha)];b=np.zeros(z.shape[1]);b[0]=np.log(y.mean()/(1-y.mean()))
    def objective(beta):
        eta=z@beta;return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty*beta**2))
    converged=False
    for iteration in range(maximum_iterations):
        eta=z@b;p=1/(1+np.exp(-np.clip(eta,-700,700)));w=p*(1-p)
        gradient=z.T@(p-y)/len(y)+penalty*b
        if np.max(np.abs(gradient))<1e-7:converged=True;break
        hessian=z.T@(w[:,None]*z)/len(y)+np.diag(penalty)
        step=np.linalg.solve(hessian+np.eye(len(b))*1e-12,gradient);old=objective(b);fraction=1.
        while fraction>1e-6 and objective(b-fraction*step)>old+1e-12:fraction*=.5
        b-=fraction*step
    return {'status':'fit' if converged else 'unsupported_numerical_nonconvergence','n':len(y),'events':int(y.sum()),'features_mean':mean.tolist(),
        'features_scale':scale.tolist(),'beta':b.tolist(),'alpha':alpha,'iterations':iteration+1,'maximum_gradient':float(np.max(np.abs(gradient))),'objective':objective(b)}


def hazard_predict(model,x, *, score_grid_hours=(1,2,4,8,16,32,64,128)):
    x=np.asarray(x,float);J=len(score_grid_hours)
    if model['status']!='fit':return np.full((len(x),J),np.nan),np.full((len(x),J),np.nan)
    z=(x-np.asarray(model['features_mean']))/np.asarray(model['features_scale']);beta=np.asarray(model['beta'])
    intercept=np.r_[beta[0],beta[0]+beta[1:J]]
    value=intercept[None,:]+z@beta[J:][:,None]
    hazard=1/(1+np.exp(-np.clip(value,-700,700)))
    return np.cumprod(1-hazard,axis=1),hazard
