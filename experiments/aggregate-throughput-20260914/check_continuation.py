import json,urllib.request,time,concurrent.futures
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/'continuation-equivalence-v2';out.mkdir(exist_ok=False);key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001'
first=json.loads((p/'after-cache-v2/first.json').read_text())['response'];known=first['choices'][0]['token_ids'];prompt=first['prompt_token_ids'];salt='aligned-check-after-cache-v2'
def call(name,endpoint,payload):
 payload=dict(payload,request_id='continuationcheck-'+name)
 req=urllib.request.Request(base+endpoint,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'});start=time.monotonic()
 with urllib.request.urlopen(req,timeout=300) as r:d=json.load(r)
 (out/(name+'.json')).write_text(json.dumps({'seconds':time.monotonic()-start,'response':d},indent=2));print(name,d.get('usage'),flush=True);return d
common={'model':'qwen3.8-flash-next','temperature':0,'seed':917352,'ignore_eos':True,'return_token_ids':True}
# A single uninterrupted generation is the independent state-continuation oracle.
ref=call('uninterrupted','/v1/completions',dict(common,prompt=prompt,max_tokens=144,logprobs=5,cache_salt=salt+'-reference'))
warm=call('restored','/v1/completions',dict(common,prompt=prompt+known,max_tokens=16,logprobs=5,cache_salt=salt))
r=ref['choices'][0];w=warm['choices'][0]
result={'first128_match':r['token_ids'][:128]==known,'restored_tokens_match_uninterrupted':w['token_ids']==r['token_ids'][128:],'restored_selected_scores_match_uninterrupted':w['logprobs']['token_logprobs']==r['logprobs']['token_logprobs'][128:],'cached_tokens':warm['usage']['prompt_tokens_details']['cached_tokens']}
# Independently capture the same boundary as an exact finish, using the
# existing accepted-state capture path rather than the retained shadow.
boundary=(len(prompt)+len(known)-1)//64*64
count=boundary-len(prompt)+1
exact=call('exact-finish','/v1/completions',dict(common,prompt=prompt,max_tokens=count,logprobs=5,cache_salt=salt+'-exact'))
exact_warm=call('exact-restored','/v1/completions',dict(common,prompt=prompt+known,max_tokens=16,logprobs=5,cache_salt=salt+'-exact'))
result['exact_finish_prefix_matches']=exact['choices'][0]['token_ids']==known[:count]
result['shadow_matches_exact_checkpoint_tokens']=exact_warm['choices'][0]['token_ids']==w['token_ids']
result['shadow_matches_exact_checkpoint_scores']=exact_warm['choices'][0]['logprobs']==w['logprobs']
result['exact_checkpoint_cached_tokens']=exact_warm['usage']['prompt_tokens_details']['cached_tokens']
# Independent salts give a cache-history-independent deterministic path.
payload=dict(common,prompt=prompt+known+[198],max_tokens=16,logprobs=5)
refcold=call('cold-serial','/v1/completions',dict(payload,cache_salt=salt+'-strict-serial'))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(lambda i:call('cold-parallel-'+str(i),'/v1/completions',dict(payload,cache_salt=salt+'-strict-'+str(i))),range(4)))
result['cold_C1_C4_tokens_equal']=all(d['choices'][0]['token_ids']==refcold['choices'][0]['token_ids'] for d in rows)
result['cold_C1_C4_scores_equal']=all(d['choices'][0]['logprobs']==refcold['choices'][0]['logprobs'] for d in rows)
result['all_strict_cached_zero']=all(d['usage']['prompt_tokens_details']['cached_tokens']==0 for d in [refcold]+rows)
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert all(result[k] for k in ['first128_match','exact_finish_prefix_matches','shadow_matches_exact_checkpoint_tokens','shadow_matches_exact_checkpoint_scores','cold_C1_C4_tokens_equal','cold_C1_C4_scores_equal','all_strict_cached_zero'])
assert result['cached_tokens']==result['exact_checkpoint_cached_tokens']==boundary
