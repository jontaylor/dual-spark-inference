"""Analyze exact-row strata and same-process publication phases."""
import collections,csv,json,pathlib,statistics,sys
p=pathlib.Path(sys.argv[1]);result={}
def quantile(a,q):
 a=sorted(a);x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def summarize(a):
 out={'n':len(a)}
 for k in ['gpu_hash_to_gate_ms','next_period_ms','gpu_wait_us','gpu_wait_polls','reader_ms','submit_ms','complete_ms','between_submit_complete_ms','misses','hits','post_staging_total_ms','hash_ms','before_hash_ms','hash_to_reader_ms','reader_to_done_ms','period_ms','gpu_launch_ms','gpu_launch_to_expected_ms','gpu_launch_to_done_ms']:
  v=[x[k] for x in a if k in x and x[k] is not None]
  if v:out[k]={'p50':quantile(v,.5),'p95':quantile(v,.95),'mean':statistics.mean(v),'max':max(v)}
 return out
for rank in (0,1):
 policies={};cpus={};deferred={};gates=[];waits=[];readers=[];submits={};completes=[];hashes=[];gpu_hashes=[];releases=collections.defaultdict(list)
 for line in (p/f'rank{rank}.txt').read_text().splitlines():
  s=line.split()
  if not s:continue
  if s[0]=='policy' and len(s) in (3,4,5):
   policies[int(s[1])]='inline'
   if len(s)>=4:cpus[int(s[1])]=int(s[3])
   if len(s)==5:deferred[int(s[1])]=1-int(s[4])
  if s[0]=='prefetch' and len(s)==3:policies[int(s[1])]='async' if int(s[2])==1 else 'inline'
  if s[0]=='gpu_gate' and len(s)==7:gates.append(tuple(map(int,s[1:])))
  if s[0]=='gpu_wait' and len(s)==7:waits.append(tuple(map(int,s[1:])))
  if s[0]=='reader' and len(s)==7:
   start,rows,ns,miss,hits,code=map(int,s[1:]);readers.append(dict(start_ns=start,end_ns=start+ns,rows=rows,reader_ms=ns/1e6,misses=miss,hits=hits,code=code))
  if s[0]=='submit' and len(s)==4:
   t,ns,code=map(int,s[1:]);submits[t]=(ns,code)
  if s[0]=='complete' and len(s)==4:
   t,ns,code=map(int,s[1:]);completes.append((t,ns,code))
  if s[0]=='hash' and len(s)==6:hashes.append(tuple(map(int,s[1:])))
  if s[0]=='hash_gpu' and len(s)==6:gpu_hashes.append(tuple(map(int,s[1:])))
  if s[0]=='release' and len(s)==4:
   t,addr,seq=map(int,s[1:]);releases[seq].append((t,addr))
 for r in readers:
  if r['start_ns'] in policies:r['policy']=policies[r['start_ns']]
  if r['start_ns'] in cpus:r['submit_cpu']=cpus[r['start_ns']]
  if r['start_ns'] in deferred:r['deferred']=deferred[r['start_ns']]
  if r['start_ns'] in submits:
   ns,code=submits[r['start_ns']];r['submit_ms']=ns/1e6
   cc=[c for c in completes if r['start_ns']<=c[0]<=r['end_ns']]
   if len(cc)==1:
    t,ns2,code2=cc[0];r['complete_ms']=ns2/1e6;r['between_submit_complete_ms']=(t-r['start_ns']-ns)/1e6
 matched=[];last=None;previous_done=0
 for seq,events in sorted(releases.items()):
  events.sort()
  if len(events)!=2 or events[0][1]!=events[1][1]+64:continue
  start,done=events[0][0],events[1][0];rr=[r for r in readers if start<=r['start_ns']<=r['end_ns']<=done];hh=[h for h in hashes if start<=h[0]<=h[1]<=done]
  gh=[h for h in gpu_hashes if previous_done<=h[0]<=h[1]<=start]
  previous_done=done
  if len(rr)!=1:continue
  r=rr[0].copy()
  if len(hh)==1:
   h=hh[0]
   assert h[1]<=r['start_ns'] and h[4]==0 and r['code']==0
   r.update(seq=seq,tokens=h[2],requests=h[3],post_staging_total_ms=(done-start)/1e6,hash_ms=(h[1]-h[0])/1e6,before_hash_ms=(h[0]-start)/1e6,hash_to_reader_ms=(r['start_ns']-h[1])/1e6,reader_to_done_ms=(done-r['end_ns'])/1e6)
  elif not hh and len(gh)==1:
   h=gh[0]
   assert h[4]==0 and r['code']==0
   r.update(seq=seq,tokens=h[2],requests=h[3],post_staging_total_ms=(done-start)/1e6,gpu_launch_ms=(h[1]-h[0])/1e6,gpu_launch_to_expected_ms=(start-h[0])/1e6,gpu_launch_to_done_ms=(done-h[0])/1e6,reader_to_done_ms=(done-r['end_ns'])/1e6)
  else:continue
  if last and last[0]+1==seq:r['period_ms']=(start-last[1])/1e6
  if matched and matched[-1]['seq']+1==seq:
   matched[-1].update(next_period_ms=r.get('period_ms'),next_rows=r['rows'],next_requests=r['requests'],next_tokens=r['tokens'],next_deferred=r.get('deferred'))
  last=seq,start;matched.append(r)
 by_seq={r['seq']:r for r in matched}
 for t,addr,current,previous,ns,polls in waits:
  events=releases.get(current,[])
  if len(events)==2 and addr==max(e[1] for e in events) and previous==current-1:
   if previous in by_seq:by_seq[previous].update(gpu_wait_us=ns/1000,gpu_wait_polls=polls)
 for t,addr,current,previous,start,end in gates:
  events=releases.get(current,[])
  if (len(events)==2 and addr==max(e[1] for e in events) and previous==current-1
      and 0<start<=end and end-start<10000000000 and previous in by_seq):
   by_seq[previous]['gpu_hash_to_gate_ms']=(end-start)/1e6
 strata=collections.defaultdict(list)
 for r in readers:strata[r['rows']].append(r)
 result[rank]={'all':summarize(readers),'errors':sum(r['code']!=0 for r in readers),'rows':{n:summarize(a) for n,a in sorted(strata.items())},'matched':summarize(matched),'decode':summarize([r for r in matched if r.get('deferred',r['tokens']<=4*r['requests'])]),'gpu_wait_samples':sum('gpu_wait_us' in r for r in matched),'gpu_wait_nonzero_polls':sum(r.get('gpu_wait_polls',0)>0 for r in matched)}
 if matched:
  fields=list(dict.fromkeys(k for r in matched for k in r))
  with (p/f'rank{rank}-steps.csv').open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(matched)
 top=sorted(strata,key=lambda n:len(strata[n]),reverse=True)[:3]
 print(rank,'top strata:',{n:result[rank]['rows'][n] for n in top},'matched:',result[rank]['matched'])
(p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
