"""Aggregate checkpointed R1 rule evidence; never zero-impute undefined fidelity."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from src.experiments.run_r1_rules import clean,write_json

def stats(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return {"mean":float(a.mean()) if len(a) else None,"std":float(a.std()) if len(a) else None,
            "valid":len(a),"total":len(values),"missing":len(values)-len(a)}

def main():
    out=Path("results/rules_r1")
    groups=defaultdict(list)
    for p in sorted(out.glob("*_seed*.json")):
        d=json.loads(p.read_text());groups[(d["experiment"],d["dataset"],d["preprocessing"])].append(d)
    summary={};lines=["# R1 rule computation summary","","Only completed checkpoints are included. Expected: E1 5 seeds per skeleton; Hillstrom 5 per preprocessing; MegaFon 10 train-only; IHDP main 10.",""]
    for key,rows in groups.items():
        name="/".join(key)
        rec={"seeds":sorted(r["seed"] for r in rows),"changed_rule_sets":sum(not r["pair"]["same_conditions"] for r in rows),"versions":{}}
        lines += [f"## {name}",f"Seeds: {rec['seeds']}; changed legacy/revised rule sets: {rec['changed_rule_sets']}",""]
        for version in ("legacy","hard"):
            vr=[r["versions"][version] for r in rows]
            report={"normalized_test_fidelity":stats([v["test_fidelity"]["value"] for v in vr]),
                    "legacy_curve_mean":stats([v["test_fidelity"]["legacy_coverage_count_curve_mean"] for v in vr]),
                    "modifier_hits":stats([v["modifier_hits"] for v in vr]),
                    "union_coverage":stats([v["test_fidelity"]["coverage"] for v in vr]),
                    "n_rules":stats([v["n_rules"] for v in vr]),
                    "degeneracy":[{"seed":r["seed"],"status":v["test_fidelity"]["status"]} for r,v in zip(rows,vr) if v["test_fidelity"]["status"]!="ok"],
                    "selection_violations":sum(not all(q["selection"][k] for k in ["support_ok","overlap_ok","max_coverage_ok"]) for v in vr for q in v["rules"]),
                    "test_support_failures":sum(not (q["test"]["support_ok"] and q["test"]["overlap_ok"]) for v in vr for q in v["rules"])}
            if "outcomes" in vr[0]:
                names=sorted(set().union(*(v["outcomes"] for v in vr)))
                report["outcomes"]={n:{m:stats([v["outcomes"].get(n,{}).get(m) for v in vr]) for m in
                    ["difference_in_means","ratio_to_population","coverage","n_treated","n_control"]} for n in names}
            rec["versions"][version]=report
            lines += [f"- {version}: "+json.dumps(clean(report),ensure_ascii=False),""]
        if "causal_tree" in rows[0]:
            rec["causal_tree"]={m:stats([r["causal_tree"][m]["value"] for r in rows]) for m in ["top5","all_leaves"]}
            rec["causal_tree"]["modifier_hits"]=stats([r["causal_tree"]["modifier_hits"] for r in rows])
            lines += ["- Causal tree: "+json.dumps(rec["causal_tree"]),""]
        if "curls" in rows[0]:rec["curls"]=stats([r["curls"]["test_fidelity"]["value"] for r in rows])
        summary[name]=rec
    write_json(out/"summary.json",summary)
    (Path("audit")/"rule_computation_summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:{"seeds":v["seeds"],"changed":v["changed_rule_sets"]} for k,v in summary.items()},indent=2))

if __name__=="__main__":main()
