import os, yaml
from typing import Dict, Tuple, Any, List
from .parsing import slugify

def load_empirical_ranges(base_dir: str, exercise_name: str) -> Dict[str, Tuple[float,float]]:
    path = os.path.join(base_dir, "data", "exercises", f"{slugify(exercise_name)}.yaml")
    if not os.path.isfile(path): return {}
    cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
    out={}
    for a in (cfg.get("angles") or []):
        rng=a.get("target_range_deg")
        if a.get("id") and isinstance(rng,list) and len(rng)==2: out[a["id"]]=(float(rng[0]),float(rng[1]))
    return out

def evaluate_against_ranges(measured: Dict[str,float], ranges: Dict[str,Tuple[float,float]]):
    rows=[]
    for aid,val in measured.items():
        if val is None: rows.append({"id":aid,"value_deg":None,"status":"unknown"}); continue
        lo,hi = ranges.get(aid,(None,None))
        if lo is None: rows.append({"id":aid,"value_deg":round(val,1),"status":"no_range"}); continue
        if val<lo: status,delta="below",round(lo-val,1)
        elif val>hi: status,delta="above",round(val-hi,1)
        else: status,delta="in_range",0.0
        rows.append({"id":aid,"value_deg":round(val,1),"target_range_deg":[lo,hi],"status":status,"delta_deg":delta})
    return rows
