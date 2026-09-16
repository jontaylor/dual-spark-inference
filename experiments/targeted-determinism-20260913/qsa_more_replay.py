exec(open('/e/qsa_projection_replay.py').read().split('parts=[]')[0])
records=[]
for name in ('o_proj','indexer.index_qk_proj'):
 w=weight(prefix+name+'.weight')
 if name=='o_proj':w=w.chunk(2,dim=1)[0]
 w=w.cuda();print(name,w.shape,w.dtype,flush=True)
 if name=='o_proj':
  torch.manual_seed(4);row=torch.randn(1,w.shape[1],device='cuda',dtype=torch.bfloat16)*0.01
  scope='synthetic BF16 input; checkpoint weights'
 else:row=x;scope='captured final prefill input row repeated'
 for small,large in [(1,4),(330,991)]:
  a=row.repeat(small,1);b=row.repeat(large,1)
  for kind,fn in [('stock',lambda v:torch.nn.functional.linear(v,w)),('fixed',lambda v:fixed_bf16_matmul(v,w.t()))]:
   u=fn(a)[-1];v=fn(b)[-1];d=(u.float()-v.float()).abs()
   records.append({'projection':'layer3.'+name,'input':scope,'M':[small,large],'method':kind,'equal':torch.equal(u,v),'changed':int((d!=0).sum()),'max_abs':float(d.max())})
Path('/e/qsa-more-replay.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
