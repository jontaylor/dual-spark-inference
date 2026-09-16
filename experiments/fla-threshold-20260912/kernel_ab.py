"""Paired FLA threshold probe; only this process imports the modified source."""
import importlib.util, json, pathlib, statistics, sys, time, types
import torch
from vllm.third_party.flash_linear_attention.ops import utils, chunk, chunk_o, cumsum
root=pathlib.Path(sys.argv[1]); root.mkdir(exist_ok=True,parents=True)
pkg='vllm.third_party.flash_linear_attention.ops'
def variant(name, source, replacement=None):
    spec=importlib.util.spec_from_file_location(pkg+'.'+name,source)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    code=pathlib.Path(source).read_text()
    if replacement:
        assert code.count(replacement[0])==1
        code=code.replace(*replacement)
    exec(compile(code,str(source),'exec'),mod.__dict__)
    return mod
u99=variant('utils99',utils.__file__,('DEFAULT = 102400','DEFAULT = 101376'))
original=utils.check_shared_mem
try:
    utils.check_shared_mem=u99.check_shared_mem
    o99=variant('chunk_o99',chunk_o.__file__)
    c99=variant('cumsum99',cumsum.__file__)
finally: utils.check_shared_mem=original
base=chunk.chunk_gated_delta_rule_fwd
newglobals=dict(base.__globals__,chunk_fwd_o=o99.chunk_fwd_o,chunk_local_cumsum=c99.chunk_local_cumsum)
patched=types.FunctionType(base.__code__,newglobals,'patched',base.__defaults__)
print(json.dumps({'shared_bytes':utils.get_all_max_shared_mem(),'baseline_gate':original(),'patched_gate':u99.check_shared_mem(),'baseline_tiles':chunk_o.BKV_LIST,'patched_tiles':o99.BKV_LIST}),flush=True)
results=[]
for lengths in ([128],[2048],[8192],[511,1025,513]):
    T=sum(lengths);H=24;Hg=8;K=V=128
    torch.manual_seed(731)
    q=torch.nn.functional.normalize(torch.randn(1,T,Hg,K,device='cuda',dtype=torch.bfloat16),dim=-1)
    k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.randn(1,T,H,V,device='cuda',dtype=torch.bfloat16)
    g=-torch.rand(1,T,H,device='cuda',dtype=torch.float32)*0.1
    beta=torch.rand(1,T,H,device='cuda',dtype=torch.bfloat16)
    h0=torch.randn(len(lengths),H,V,K,device='cuda',dtype=torch.float32)*0.05
    bounds=[0]
    for n in lengths:bounds.append(bounds[-1]+n)
    cu=torch.tensor(bounds,device='cuda',dtype=torch.int32)
    def run(fn):return fn(q,k,v,g,beta,K**-0.5,h0,True,cu_seqlens=cu)
    print('warmup '+str(lengths),flush=True)
    a=run(base);b=run(patched);torch.cuda.synchronize()
    diffs={}
    for i,name in [(1,'output'),(3,'state')]:
        aa=a[i].float();bb=b[i].float();d=aa-bb
        diffs[name]={'finite':bool(torch.isfinite(bb).all()),'equal':bool(torch.equal(aa,bb)),'relative_l2':float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(aa)),'max_abs':float(d.abs().max())}
    times={'baseline':[],'patched':[]}
    for repeat in range(12):
        for name,fn in ([('baseline',base),('patched',patched)] if repeat%2==0 else [('patched',patched),('baseline',base)]):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):run(fn)
            end.record();end.synchronize();times[name].append(start.elapsed_time(end)/3)
    row={'lengths':lengths,'ms':times,'median_ms':{k:statistics.median(v) for k,v in times.items()},'difference':diffs}
    row['speedup']=row['median_ms']['baseline']/row['median_ms']['patched']
    results.append(row);print(json.dumps(row),flush=True)
    (root/'kernel-results.json').write_text(json.dumps(results,indent=2))
print('DONE',flush=True)
