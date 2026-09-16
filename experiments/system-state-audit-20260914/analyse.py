import json,ast,difflib,hashlib,shlex
from pathlib import Path
root=Path.cwd();out=Path(__file__).parent;hist=root/'experiments/performance-until-10am-20260914'
a=json.loads((hist/'baseline-r0.json').read_text());j=json.loads((hist/'candidate-J-reservation-r0.json').read_text());n=json.loads((root/'deploy_config.json').read_text())
def flat(v,p=''):
 d={}
 for k,x in v.items():
  if k.startswith('runtime_override'):continue
  q=p+k
  if isinstance(x,dict):d.update(flat(x,q+'.'))
  else:d[q]=x
 return d
fa,fj,fn=map(flat,[a,j,n]);rows=[{'setting':k,'before':fa.get(k,'NOT SET'),'J':fj.get(k,'NOT SET'),'now':fn.get(k,'NOT SET')} for k in sorted(fa.keys()|fj.keys()|fn.keys())];(out/'config-delta.json').write_text(json.dumps(rows,indent=2))
for label,d in [('A',a),('J',j),('now',n)]: (out/f'config-{label}.json').write_text(json.dumps(d,indent=2))
for rank in [0,1]:
 d=json.loads((out/f'live-r{rank}.json').read_text());c=d['container'];p1=next(x for x in d['processes'] if x['pid']==1)
 txt=f"Captured UTC: {d['captured_utc']}\nContainer: {c['Name']}\nID: {c['Id']}\nImage ID: {c['Image']}\nStarted: {c['State']['StartedAt']}\n\nDocker Path/Args:\n{shlex.join([c['Path']]+c['Args'])}\n\nActual PID1 argv:\n{shlex.join(p1['argv'])}\n\nDocker Config.Env:\n"+'\n'.join(c['Config']['Env'])+'\n\nActual PID1 environ:\n'+'\n'.join(p1['environment'])+'\n'
 for proc in d['processes']:
  if any('VLLM::' in x for x in proc['argv']):txt+=f"\nPID {proc['pid']} argv: {shlex.join(proc['argv'])}\nEnvironment:\n"+'\n'.join(proc['environment'])+'\n'
 (out/f'live-r{rank}.txt').write_text(txt)
 overrides=[]
 for f,s in n['runtime_overrides'].items():
  actual=d['mounted_python_hashes'].get('/usr/local/lib/python3.12/dist-packages/vllm/'+f)
  overrides.append({'file':f,'actual':actual,'expected':n['runtime_override_sha256'][f],'match':actual is not None and actual['sha256']==n['runtime_override_sha256'][f]})
 (out/f'override-verification-r{rank}.json').write_text(json.dumps(overrides,indent=2));print('rank',rank,'verified',sum(x['match'] for x in overrides),'/',len(overrides))
# source diffs baseline->now; retain full changed function inventory
inv=[]
for i,(f,s) in enumerate(n['runtime_overrides'].items(),1):
 old=a['runtime_overrides'].get(f);new=Path(s).read_text();baseline=Path(old).read_text() if old else ''
 diff=''.join(difflib.unified_diff(baseline.splitlines(True),new.splitlines(True),fromfile=old or 'NOT AN OVERRIDE IN A',tofile=s))
 (out/f'override-{i:02d}-A-now.diff').write_text(diff)
 def symbols(src):
  t=ast.parse(src);d={}
  def walk(node,path=''):
   for ch in ast.iter_child_nodes(node):
    if isinstance(ch,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
     name=path+ch.name;d[name]=ast.dump(ch,include_attributes=False);walk(ch,name+'.')
  walk(t);return d
 before=symbols(baseline);after=symbols(new)
 inv.append({'number':i,'file':f,'A_source':old,'J_source':j['runtime_overrides'].get(f),'now_source':s,'introduced_after_A':old is None,'changed_symbols_A_now':[k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)],'all_symbols':list(after)})
(out/'override-inventory.json').write_text(json.dumps(inv,indent=2))
# cold/warm exact first divergence
r=root/'experiments/aggregate-throughput-20260914/after-J';w=json.loads((r/'warm.json').read_text())['response'];c=json.loads((r/'cold.json').read_text())['response'];cw=w['choices'][0];cc=c['choices'][0]
wi=cw['token_ids'];ci=cc['token_ids'];pos=next((i for i,(x,y) in enumerate(zip(wi,ci)) if x!=y),None)
d={'prompt_equal':cw.get('prompt_token_ids')==cc.get('prompt_token_ids'),'warm_usage':w['usage'],'cold_usage':c['usage'],'warm_ids':wi,'cold_ids':ci,'warm_text':cw.get('text'),'cold_text':cc.get('text'),'first_different_position_1based':None if pos is None else pos+1,'positions_different':sum(x!=y for x,y in zip(wi,ci)),'warm_logprobs':cw['logprobs'],'cold_logprobs':cc['logprobs']}
(out/'cold-warm-difference.json').write_text(json.dumps(d,indent=2));print('COLD',json.dumps({k:v for k,v in d.items() if 'logprobs' not in k},indent=2))
print('CHANGED SETTINGS',json.dumps([r for r in rows if r['before']!=r['now'] or r['before']!=r['J']],indent=2))
