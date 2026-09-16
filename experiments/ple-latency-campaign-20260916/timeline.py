"""Summarize PLE overlap from one bounded Nsight SQLite export."""
import argparse
import json
import sqlite3
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
args = parser.parse_args()
c = sqlite3.connect(args.database)
ranges = c.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'ple_in_process.%' ORDER BY start").fetchall()
kernels = c.execute("SELECT k.start,k.end,s.value,k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start").fetchall()
launches = c.execute("SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE 'cudaGraphLaunch%' ORDER BY r.start").fetchall()
steps = []
prepares = [(a,b) for a,b,t in ranges if t == "ple_in_process.prepare"]
for index,(a,b) in enumerate(prepares):
    boundary = prepares[index+1][0] if index+1 < len(prepares) else 2**63-1
    launch = next(((x,y) for x,y in launches if b <= x < boundary), None)
    wait = next(((x,y) for x,y,name,_ in kernels if b <= x < boundary and "wait_ready" in name), None)
    if launch is None or wait is None:
        continue
    first = next((x for x,y,name,node in kernels if launch[0] <= x <= wait[0] and node), None)
    if first is None:
        continue
    row = {"prepare_start_ns": a, "prepare_ms": (b-a)/1e6, "graph_launch_cpu_ms": (launch[1]-launch[0])/1e6, "gpu_early_layers_ms": (wait[0]-first)/1e6, "gpu_wait_us": (wait[1]-wait[0])/1e3}
    # Union of kernel intervals: concurrent streams must not count twice.
    intervals = sorted((max(x,a),min(y,b)) for x,y,_,_ in kernels if x < b and y > a)
    busy = 0
    end = a
    for x,y in intervals:
        busy += max(0,y-max(x,end))
        end = max(end,y)
    row["prepare_gpu_kernel_idle_ms"] = (b-a-busy)/1e6
    complete = next(((x,y) for x,y,t in ranges if b <= x < boundary and t == "ple_in_process.complete"), None)
    if complete:
        row.update(complete_ms=(complete[1]-complete[0])/1e6, complete_end_to_gpu_need_ms=(wait[0]-complete[1])/1e6, prepare_to_complete_ms=(complete[1]-a)/1e6)
    steps.append(row)
summary = {"steps":len(steps),"metrics":{}}
for key in steps[0] if steps else []:
    if key.endswith("_ns"):
        continue
    values = sorted(row[key] for row in steps if key in row)
    summary["metrics"][key] = {"median": statistics.median(values), "mean": statistics.mean(values), "p95": values[min(len(values)-1,int(.95*len(values)))], "max":max(values)}
output = args.database.with_suffix(".ple.json")
output.write_text(json.dumps({"summary":summary,"steps":steps},indent=2)+"\n")
print(json.dumps(summary,indent=2))
