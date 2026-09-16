exec(open('/e/head_projection_check.py').read().split('records=[]')[0])
from triton.testing import do_bench
rows=[]
for M in [1,4,8]:
 a=x.repeat(M,1)
 for name,fn in [('stock',lambda:torch.nn.functional.linear(a,w)),('fixed128',lambda:fixed_bf16_matmul(a,w.t())),('fixed16',lambda:small(a))]:
  out=fn()
  if name=='fixed16':assert torch.all(out==small(x))
  ms=do_bench(fn,warmup=20,rep=100)
  rows.append({'M':M,'method':name,'ms':ms})
Path('/e/head-benchmark.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
