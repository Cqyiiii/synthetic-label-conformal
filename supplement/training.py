"""V11 full-batch learner with immutable intermediate checkpoints."""
from __future__ import annotations

import copy

import common
import numpy as np
from synrcp.v11_neural import _prepare_torch, _network, _make_predictor, _parameter_norm


def fit_candidates(x,s,xp,sp,sd,*,powers,checkpoints,seed,tau,hidden,learning_rate,bound):
    candidates=[]
    for power in powers:
        torch,nn=_prepare_torch(seed)
        model=_network(nn,x.shape[1],1,hidden)
        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
        tx=torch.from_numpy(np.asarray(x,dtype=np.float32))
        ts=torch.from_numpy(np.asarray(s,dtype=np.float32).reshape(-1,1))
        txp=torch.from_numpy(np.asarray(xp,dtype=np.float32))
        tsp=torch.from_numpy(np.asarray(sp,dtype=np.float32).reshape(-1,1))
        tsd=torch.from_numpy(np.asarray(sd,dtype=np.float32).reshape(-1,1))
        bound=float(max(bound,1e-6))
        def transform(raw):
            return bound*torch.sigmoid(raw)
        def lossfn(pred,target):
            residual=target-pred
            return torch.maximum(tau*residual,(tau-1)*residual)
        curve=[]
        for epoch in range(1,max(checkpoints)+1):
            optimizer.zero_grad(set_to_none=True)
            pt=transform(model(tx))
            lt=lossfn(pt,ts).mean()
            lp=torch.zeros((),dtype=lt.dtype)
            ld=torch.zeros((),dtype=lt.dtype)
            if power:
                lp=lossfn(transform(model(txp)),tsp).mean()
                ld=lossfn(pt,tsd).mean()
            loss=lt+power*(lp-ld) if power else lt
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite objective lambda={power}, epoch={epoch}')
            loss.backward()
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('Nonfinite model parameters')
            if epoch==1 or epoch%5==0:
                curve.append({'epoch':epoch,'objective':float(loss.detach()),'trusted':float(lt.detach()),'pool':float(lp.detach()),'paired':float(ld.detach())})
            if epoch in checkpoints:
                frozen=copy.deepcopy(model).eval()
                predictor=_make_predictor(torch,frozen,transform)
                def predict(values,p=predictor):
                    return p(values).reshape(-1)
                candidates.append({'id':f'lambda{power:g}_epoch{epoch}', 'power':float(power),'epoch':epoch,
                                   'predict':predict,'state':copy.deepcopy(frozen.state_dict()),
                                   'diagnostics':{'curve':copy.deepcopy(curve),'parameter_norm':_parameter_norm(torch,frozen),
                                                  'finite':True,'seed':seed,'bound':bound,'hidden':hidden}})
    return candidates
