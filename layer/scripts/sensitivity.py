# Run: PYTHONPATH=. python scripts/sensitivity.py
import numpy as np
from cognitive_layer import *
from cognitive_layer.sim import stream, declare_all
from cognitive_layer.evaluation import run_arm, oracle_instance, paired_bootstrap
for v in (0.7,0.3):
  U=UtilityTable(verify=v)
  def make(mode="scoped", scoped=True, shuffle=False):
    def f():
        reg=CalibrationRegistry()
        if scoped: declare_all(reg)
        return CognitiveLayer(reg, ExpectedUtilityPolicy(U), mode=mode, shuffle_evidence=shuffle)
    return f
  arms={"scoped":make(),"global":make(scoped=False),"raw":make("raw"),"base_rate_only":make("base_rate_only"),"shuffled":make(shuffle=True)}
  u={k:[] for k in arms}; o=[]
  for seed in range(3):
    learn=stream(1500,seed); hold=stream(1000,1000+seed)
    for k,f in arms.items(): u[k].append(run_arm(f,learn,hold,U)["holdout_utility"])
    o.append(oracle_instance(hold,U))
  u={k:np.concatenate(x) for k,x in u.items()}; o=np.concatenate(o)
  print("verify",v,{k:round(x.mean(),3) for k,x in u.items()},"oracle",round(o.mean(),3))
  for k in arms:
    if k!="scoped":
      b=paired_bootstrap(u["scoped"],u[k]); print("  scoped-",k,round(b["mean"],3),[round(b["lower"],3),round(b["upper"],3)])
