"""Registered monotone Platt numerical optimization and application."""

import numpy as np
from .trees import sigmoid


# Monotone Platt optimization

def logit_input(p, epsilon=1e-8):
    v=np.asarray(p,float);eps=epsilon;z=np.clip(v,eps,1-eps);return np.log(z)-np.log1p(-z)


def platt_fit(p,y,minimum=None,minimum_class=None,monotone=True, *, epsilon=1e-8, l2=0.001):
    p=np.asarray(p,float);y=np.asarray(y,float);good=np.isfinite(p)&np.isfinite(y);p=p[good];y=y[good]
    minimum=minimum if minimum is not None else 200;minimum_class=minimum_class if minimum_class is not None else 20
    if len(y)<minimum or min(y.sum(),len(y)-y.sum())<minimum_class:return {'status':'unsupported_calibration_support','n':len(y),'events':int(y.sum())}
    x=np.c_[np.ones(len(y)),logit_input(p, epsilon)];b=np.array([0.,1.]);prior=np.array([0.,1.])
    def objective(v):
        eta=x@v;return float(np.mean(np.logaddexp(0,eta)-y*eta)+.5*l2*np.sum((v-prior)**2))
    converged=False
    for iteration in range(100):
        pp=sigmoid(x@b);g=x.T@(pp-y)/len(y)+l2*(b-prior);projected=g.copy()
        if monotone and b[1]<=1e-12 and g[1]>0:projected[1]=0
        if np.max(np.abs(projected))<1e-8:converged=True;break
        H=x.T@((pp*(1-pp))[:,None]*x)/len(y)+l2*np.eye(2);step=np.linalg.solve(H,g);fraction=1.;old=objective(b)
        while fraction>1e-10:
            candidate=b-fraction*step
            if monotone:candidate[1]=max(candidate[1],0)
            if objective(candidate)<=old+1e-13:break
            fraction*=.5
        if np.max(np.abs(candidate-b))<1e-10:converged=True;b=candidate;break
        b=candidate
    return {'status':'fit' if converged else 'unsupported_calibration_numerical','n':len(y),'events':int(y.sum()),'intercept':float(b[0]),'slope':float(b[1]),'objective':objective(b),'iterations':iteration+1,
        'monotone':monotone,'finite_logit_clipping_inputs':int(((p<epsilon)|(p>1-epsilon)).sum())}


def calibrate(model,p, *, epsilon=1e-8):
    p=np.asarray(p,float)
    if model['status']!='fit':return np.full(len(p),np.nan)
    return np.where(np.isfinite(p),sigmoid(model['intercept']+model['slope']*logit_input(p, epsilon)),np.nan)
